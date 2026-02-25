"""
Qdrant 벡터를 TensorFlow Embedding Projector 형식으로 내보내는 스크립트.

https://projector.tensorflow.org/ 에서 시각화할 수 있는 두 파일을 생성한다:
1. vectors.tsv  — 벡터 값 (탭 구분, 헤더 없음)
2. metadata.tsv — 메타데이터 (제목, 장르, 감독 등)

사용법:
    uv run python scripts/export_embeddings_for_projector.py

    # 출력 디렉토리 지정
    uv run python scripts/export_embeddings_for_projector.py --output-dir data/projector

시각화 방법:
    1. https://projector.tensorflow.org/ 접속
    2. 좌측 "Load" 클릭
    3. "Choose file" → vectors.tsv 업로드
    4. "Choose file" → metadata.tsv 업로드
    5. PCA / t-SNE / UMAP 선택하여 시각화
"""

import argparse
import asyncio
import csv
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monglepick.config import settings  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402


def export_embeddings(output_dir: str, limit: int = 0) -> None:
    """
    Qdrant에서 벡터와 메타데이터를 TSV 파일로 내보낸다.

    Args:
        output_dir: 출력 디렉토리
        limit: 내보낼 최대 벡터 수 (0 = 전체)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Qdrant 동기 클라이언트 (스크립트용)
    client = QdrantClient(
        url=settings.QDRANT_URL,
        check_compatibility=False,
    )

    # 컬렉션 정보 확인
    collection_info = client.get_collection(settings.QDRANT_COLLECTION)
    total_points = collection_info.points_count
    print(f"컬렉션: {settings.QDRANT_COLLECTION}, 총 벡터: {total_points}")

    if limit > 0:
        total_points = min(total_points, limit)
        print(f"내보낼 벡터 수: {total_points}")

    # 모든 포인트를 스크롤하여 가져오기
    vectors_file = output_path / "vectors.tsv"
    metadata_file = output_path / "metadata.tsv"

    # 메타데이터 헤더
    metadata_headers = ["title", "genres", "director", "mood_tags", "release_year", "rating"]

    exported = 0
    offset = None  # 첫 페이지

    with open(vectors_file, "w", newline="") as vf, \
         open(metadata_file, "w", newline="", encoding="utf-8") as mf:

        vec_writer = csv.writer(vf, delimiter="\t")
        meta_writer = csv.writer(mf, delimiter="\t")

        # 메타데이터 헤더 작성
        meta_writer.writerow(metadata_headers)

        while True:
            # Qdrant scroll API로 배치 가져오기
            points, next_offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=100,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )

            if not points:
                break

            for point in points:
                # 벡터 TSV 행 작성 (탭 구분 실수 값)
                vector = point.vector
                vec_writer.writerow(vector)

                # 메타데이터 TSV 행 작성
                payload = point.payload or {}
                title = payload.get("title", "")
                genres = ", ".join(payload.get("genres", []))
                director = payload.get("director", "")
                mood_tags = ", ".join(payload.get("mood_tags", []))
                release_year = str(payload.get("release_year", ""))
                rating = str(payload.get("rating", ""))

                meta_writer.writerow([title, genres, director, mood_tags, release_year, rating])

                exported += 1
                if limit > 0 and exported >= limit:
                    break

            if limit > 0 and exported >= limit:
                break

            if next_offset is None:
                break
            offset = next_offset

            if exported % 500 == 0:
                print(f"  내보내기 진행: {exported}/{total_points}")

    print(f"\n내보내기 완료!")
    print(f"  벡터 파일:     {vectors_file} ({exported}건)")
    print(f"  메타데이터 파일: {metadata_file} ({exported}건)")
    print(f"\n시각화 방법:")
    print(f"  1. https://projector.tensorflow.org/ 접속")
    print(f"  2. 좌측 'Load' 클릭")
    print(f"  3. 'Step 1: Load a TSV file of vectors' → {vectors_file.name} 업로드")
    print(f"  4. 'Step 2: Load a TSV file of metadata' → {metadata_file.name} 업로드")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant 벡터를 TF Projector용 TSV로 내보내기")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/projector",
        help="출력 디렉토리 (기본: data/projector)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="내보낼 최대 벡터 수 (0 = 전체, 기본: 0)",
    )
    args = parser.parse_args()
    export_embeddings(args.output_dir, args.limit)
