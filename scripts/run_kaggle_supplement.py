"""
Kaggle 데이터 보강 파이프라인.

TMDB API로 수집되지 않은 ~44,000건의 영화를 Kaggle CSV에서 변환하여
Qdrant/Neo4j/ES에 추가 적재한다.

체크포인트 기반으로 중단/재개 가능하다.

사용법:
    # Kaggle 보강 실행 (중단 시 이어서 진행)
    uv run python scripts/run_kaggle_supplement.py --kaggle-dir ../theMoviesDataset/archive

    # 배치 크기 조정 (임베딩 API Rate Limit 고려)
    uv run python scripts/run_kaggle_supplement.py --kaggle-dir ../theMoviesDataset/archive --batch-size 1000

    # 현재 체크포인트 상태만 확인
    uv run python scripts/run_kaggle_supplement.py --status
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog  # noqa: E402

from monglepick.data_pipeline.checkpoint import PipelineCheckpoint  # noqa: E402
from monglepick.data_pipeline.embedder import embed_texts  # noqa: E402
from monglepick.data_pipeline.es_loader import load_to_elasticsearch  # noqa: E402
from monglepick.data_pipeline.kaggle_enricher import load_kaggle_movies  # noqa: E402
from monglepick.data_pipeline.neo4j_loader import load_to_neo4j  # noqa: E402
from monglepick.data_pipeline.qdrant_loader import load_to_qdrant  # noqa: E402
from monglepick.db.clients import init_all_clients, close_all_clients  # noqa: E402

logger = structlog.get_logger()


def init_checkpoint_from_qdrant(checkpoint: PipelineCheckpoint) -> None:
    """
    기존 Qdrant 데이터에서 체크포인트를 초기화한다.

    첫 실행 시 이미 적재된 TMDB API 데이터를 체크포인트에 반영한다.
    """
    from qdrant_client import QdrantClient
    from monglepick.config import settings

    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)

    # 모든 포인트 ID와 source 필드를 가져온다
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_vectors=False,
            with_payload=["source"],
        )
        if not points:
            break

        for p in points:
            source = (p.payload or {}).get("source", "tmdb")
            if source == "kaggle":
                checkpoint.kaggle_loaded_ids.add(p.id)
            else:
                checkpoint.tmdb_api_loaded_ids.add(p.id)
            checkpoint.embedded_ids.add(p.id)

        if next_offset is None:
            break
        offset = next_offset

    client.close()
    logger.info(
        "checkpoint_initialized_from_qdrant",
        tmdb_api=len(checkpoint.tmdb_api_loaded_ids),
        kaggle=len(checkpoint.kaggle_loaded_ids),
    )


async def run_kaggle_supplement(
    kaggle_dir: str,
    batch_size: int = 2000,
    embedding_batch_size: int = 50,
) -> None:
    """
    Kaggle 데이터 보강 파이프라인.

    1. 체크포인트 로드 (또는 Qdrant에서 초기화)
    2. Kaggle CSV → MovieDocument 변환 (이미 적재된 ID 제외)
    3. 배치 단위로: 임베딩 → Qdrant/Neo4j/ES 적재
    4. 배치마다 체크포인트 저장 (중단 대응)

    Args:
        kaggle_dir: Kaggle 데이터 디렉토리
        batch_size: 적재 배치 크기 (기본 2000건씩)
        embedding_batch_size: Upstage 임베딩 API 배치 크기
    """
    # 1. 체크포인트 로드
    checkpoint = PipelineCheckpoint()
    checkpoint.load()

    # 체크포인트가 비어있으면 Qdrant에서 초기화
    if not checkpoint.all_loaded_ids:
        logger.info("checkpoint_empty_initializing_from_qdrant")
        init_checkpoint_from_qdrant(checkpoint)
        checkpoint.save()

    logger.info("current_checkpoint", summary=checkpoint.summary())

    # 2. DB 초기화
    await init_all_clients()

    try:
        # 3. Kaggle → MovieDocument 변환 (이미 적재된 ID 제외)
        logger.info("loading_kaggle_movies", exclude_count=len(checkpoint.all_loaded_ids))
        documents = load_kaggle_movies(
            kaggle_dir=kaggle_dir,
            exclude_ids=checkpoint.all_loaded_ids,
        )

        if not documents:
            logger.info("no_new_kaggle_movies_to_load")
            print("보강할 새 영화가 없습니다.")
            return

        checkpoint.kaggle_total_available = len(documents)
        logger.info("kaggle_movies_to_load", count=len(documents))

        # 4. 배치 단위로 임베딩 → 적재 → 체크포인트 저장
        total_loaded = 0

        for batch_start in range(0, len(documents), batch_size):
            batch_end = min(batch_start + batch_size, len(documents))
            batch = documents[batch_start:batch_end]

            logger.info(
                "batch_processing",
                batch=f"{batch_start}~{batch_end}",
                batch_size=len(batch),
                total=len(documents),
            )

            # 4a. 임베딩 생성 (Upstage API)
            texts = [doc.embedding_text for doc in batch]
            embeddings = embed_texts(texts, batch_size=embedding_batch_size)

            # 4b. Qdrant 적재
            qdrant_count = await load_to_qdrant(batch, embeddings)

            # 4c. Neo4j 적재
            await load_to_neo4j(batch)

            # 4d. ES 적재
            es_count = await load_to_elasticsearch(batch)

            # 4e. 체크포인트 업데이트 및 저장
            for doc in batch:
                doc_id = int(doc.id)
                checkpoint.kaggle_loaded_ids.add(doc_id)
                checkpoint.embedded_ids.add(doc_id)

            checkpoint.save()
            total_loaded += len(batch)

            logger.info(
                "batch_complete",
                batch_loaded=len(batch),
                total_loaded=total_loaded,
                remaining=len(documents) - total_loaded,
                qdrant=qdrant_count,
                es=es_count,
            )

        # 5. 완료
        logger.info(
            "kaggle_supplement_complete",
            total_loaded=total_loaded,
            summary=checkpoint.summary(),
        )
        print(f"\nKaggle 보강 완료! {total_loaded}건 추가 적재")
        print(f"체크포인트: {checkpoint.summary()}")

    finally:
        await close_all_clients()


def show_status() -> None:
    """현재 체크포인트 상태를 출력한다."""
    checkpoint = PipelineCheckpoint()
    checkpoint.load()

    if not checkpoint.all_loaded_ids:
        # Qdrant에서 초기화
        init_checkpoint_from_qdrant(checkpoint)
        checkpoint.save()

    print("=" * 60)
    print("  파이프라인 체크포인트 상태")
    print("=" * 60)
    print(f"  TMDB API 적재:  {len(checkpoint.tmdb_api_loaded_ids):>10,}건")
    print(f"  Kaggle 적재:    {len(checkpoint.kaggle_loaded_ids):>10,}건")
    print(f"  총 적재:        {len(checkpoint.all_loaded_ids):>10,}건")
    print(f"  임베딩 완료:    {len(checkpoint.embedded_ids):>10,}건")
    print(f"  실패:           {len(checkpoint.failed_ids):>10,}건")
    print(f"  최종 업데이트:  {checkpoint.last_updated}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle 데이터 보강 파이프라인")
    parser.add_argument(
        "--kaggle-dir",
        type=str,
        default="../theMoviesDataset/archive",
        help="Kaggle 데이터 디렉토리 경로",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="적재 배치 크기 (기본: 2000)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=50,
        help="Upstage 임베딩 API 배치 크기 (기본: 50, Rate Limit 100RPM)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="현재 체크포인트 상태만 확인",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        asyncio.run(
            run_kaggle_supplement(
                kaggle_dir=args.kaggle_dir,
                batch_size=args.batch_size,
                embedding_batch_size=args.embedding_batch_size,
            )
        )
