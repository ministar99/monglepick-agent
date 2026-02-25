"""
Qdrant 벡터DB 유사도 검색 결과 Export 스크립트.

동일 벡터에 대해 Cosine / Euclidean 두 가지 거리 방식으로 유사도 검색을 수행하고
결과를 CSV 파일로 저장한다.

Qdrant의 기존 movies 컬렉션(Cosine)을 활용하고,
Euclidean 검색을 위해 임시 컬렉션을 생성한 뒤 자동 정리한다.

사용법:
    # 기본 실행 (인기 상위 10개 영화 기준, 각 top-20 유사 영화)
    uv run python scripts/export_similarity_search.py

    # 시드 영화 수 / 유사 영화 수 조정
    uv run python scripts/export_similarity_search.py --seed-count 20 --top-k 30

    # 특정 영화 ID로 검색
    uv run python scripts/export_similarity_search.py --seed-ids 11 12 120 155 597

출력 파일:
    data/similarity_search/cosine_results.csv
    data/similarity_search/euclidean_results.csv
    data/similarity_search/comparison_summary.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    Distance,
    PointStruct,
    VectorParams,
)
from monglepick.config import settings  # noqa: E402


# ── 상수 ──
COSINE_COLLECTION = settings.QDRANT_COLLECTION  # 기존 "movies" 컬렉션 (Cosine)
EUCLIDEAN_COLLECTION = "movies_euclidean_temp"   # 임시 Euclidean 컬렉션
OUTPUT_DIR = Path("data/similarity_search")


def get_seed_movies(
    client: QdrantClient,
    seed_ids: list[int] | None = None,
    seed_count: int = 10,
) -> list[dict]:
    """
    유사도 검색의 시드(seed) 영화를 선정한다.

    seed_ids가 지정되면 해당 ID의 영화를 사용하고,
    아니면 인기도(popularity_score) 상위 seed_count개를 선택한다.

    Args:
        client: Qdrant 동기 클라이언트
        seed_ids: 특정 영화 TMDB ID 리스트
        seed_count: 시드 영화 수 (seed_ids 미지정 시)

    Returns:
        list[dict]: 시드 영화 정보 (id, title, vector, payload)
    """
    if seed_ids:
        # 특정 ID로 포인트 조회
        points = client.retrieve(
            collection_name=COSINE_COLLECTION,
            ids=seed_ids,
            with_vectors=True,
            with_payload=True,
        )
        print(f"  지정 시드 영화 {len(points)}건 조회 완료")
    else:
        # 전체 영화에서 인기도 상위 seed_count개 선택
        # Qdrant scroll로 전체 가져온 뒤 Python에서 정렬
        all_points = []
        offset = None
        while True:
            points_batch, next_offset = client.scroll(
                collection_name=COSINE_COLLECTION,
                limit=500,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            if not points_batch:
                break
            all_points.extend(points_batch)
            if next_offset is None:
                break
            offset = next_offset

        # 인기도(popularity_score) 기준 내림차순 정렬
        all_points.sort(
            key=lambda p: (p.payload or {}).get("popularity_score", 0),
            reverse=True,
        )
        points = all_points[:seed_count]
        print(f"  인기도 상위 {len(points)}건을 시드 영화로 선정")

    # 결과 변환
    seeds = []
    for p in points:
        payload = p.payload or {}
        seeds.append({
            "id": p.id,
            "title": payload.get("title", "N/A"),
            "genres": payload.get("genres", []),
            "director": payload.get("director", "N/A"),
            "release_year": payload.get("release_year", 0),
            "rating": payload.get("rating", 0),
            "popularity_score": payload.get("popularity_score", 0),
            "vector": p.vector,
        })

    return seeds


def create_euclidean_collection(client: QdrantClient) -> int:
    """
    기존 Cosine 컬렉션의 벡터를 Euclidean 거리 컬렉션으로 복사한다.

    동일한 벡터를 Euclidean 거리로 검색하기 위해 임시 컬렉션을 생성한다.

    Args:
        client: Qdrant 동기 클라이언트

    Returns:
        int: 복사된 벡터 수
    """
    # 기존 컬렉션 정보에서 벡터 차원 가져오기
    cosine_info = client.get_collection(COSINE_COLLECTION)
    vector_size = cosine_info.config.params.vectors.size

    # 임시 Euclidean 컬렉션이 이미 있으면 삭제
    existing = [c.name for c in client.get_collections().collections]
    if EUCLIDEAN_COLLECTION in existing:
        client.delete_collection(EUCLIDEAN_COLLECTION)
        print(f"  기존 임시 컬렉션 '{EUCLIDEAN_COLLECTION}' 삭제")

    # Euclidean 거리 기반 새 컬렉션 생성
    client.create_collection(
        collection_name=EUCLIDEAN_COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.EUCLID,
        ),
    )
    print(f"  임시 컬렉션 '{EUCLIDEAN_COLLECTION}' 생성 (Euclidean, {vector_size}차원)")

    # 기존 Cosine 컬렉션에서 벡터+페이로드 전체를 복사
    copied = 0
    offset = None
    while True:
        points_batch, next_offset = client.scroll(
            collection_name=COSINE_COLLECTION,
            limit=100,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not points_batch:
            break

        # 배치 upsert
        client.upsert(
            collection_name=EUCLIDEAN_COLLECTION,
            points=[
                PointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload=p.payload or {},
                )
                for p in points_batch
            ],
        )
        copied += len(points_batch)

        if next_offset is None:
            break
        offset = next_offset

        if copied % 500 == 0:
            print(f"    복사 진행: {copied}건")

    print(f"  벡터 복사 완료: {copied}건 → {EUCLIDEAN_COLLECTION}")
    return copied


def search_similar(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int = 20,
) -> list[dict]:
    """
    벡터 유사도 검색을 수행한다.

    Args:
        client: Qdrant 동기 클라이언트
        collection_name: 검색 대상 컬렉션
        query_vector: 쿼리 벡터
        top_k: 반환 개수

    Returns:
        list[dict]: 유사 영화 리스트 (rank, id, title, score, ...)
    """
    # query_points 사용 (Qdrant v1.17+)
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k + 1,  # 자기 자신 포함될 수 있으므로 +1
    )

    similar = []
    rank = 0
    for point in results.points:
        payload = point.payload or {}
        rank += 1
        similar.append({
            "rank": rank,
            "id": point.id,
            "title": payload.get("title", "N/A"),
            "genres": ", ".join(payload.get("genres", [])),
            "director": payload.get("director", "N/A"),
            "release_year": payload.get("release_year", 0),
            "rating": payload.get("rating", 0),
            "score": round(point.score, 6),
        })

    return similar


def export_results(
    seeds: list[dict],
    cosine_results: dict[int, list[dict]],
    euclidean_results: dict[int, list[dict]],
) -> None:
    """
    유사도 검색 결과를 CSV 파일로 저장한다.

    3개 파일을 생성한다:
    1. cosine_results.csv     — Cosine 유사도 검색 결과
    2. euclidean_results.csv  — Euclidean 유사도 검색 결과
    3. comparison_summary.csv — 두 방식 비교 요약 (시드별 Top-10 겹침 비율 등)

    Args:
        seeds: 시드 영화 리스트
        cosine_results: {seed_id: [유사영화 리스트]} (Cosine)
        euclidean_results: {seed_id: [유사영화 리스트]} (Euclidean)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV 공통 헤더
    detail_headers = [
        "seed_id", "seed_title", "seed_genres", "seed_year",
        "rank", "similar_id", "similar_title", "similar_genres",
        "similar_director", "similar_year", "similar_rating", "score",
    ]

    # ── 1. Cosine 결과 CSV ──
    cosine_file = OUTPUT_DIR / "cosine_results.csv"
    with open(cosine_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(detail_headers)
        for seed in seeds:
            sid = seed["id"]
            for item in cosine_results.get(sid, []):
                writer.writerow([
                    sid,
                    seed["title"],
                    ", ".join(seed["genres"]),
                    seed["release_year"],
                    item["rank"],
                    item["id"],
                    item["title"],
                    item["genres"],
                    item["director"],
                    item["release_year"],
                    item["rating"],
                    item["score"],
                ])
    print(f"  Cosine 결과: {cosine_file}")

    # ── 2. Euclidean 결과 CSV ──
    euclidean_file = OUTPUT_DIR / "euclidean_results.csv"
    with open(euclidean_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(detail_headers)
        for seed in seeds:
            sid = seed["id"]
            for item in euclidean_results.get(sid, []):
                writer.writerow([
                    sid,
                    seed["title"],
                    ", ".join(seed["genres"]),
                    seed["release_year"],
                    item["rank"],
                    item["id"],
                    item["title"],
                    item["genres"],
                    item["director"],
                    item["release_year"],
                    item["rating"],
                    item["score"],
                ])
    print(f"  Euclidean 결과: {euclidean_file}")

    # ── 3. 비교 요약 CSV ──
    # 시드별로 Cosine Top-10과 Euclidean Top-10의 겹치는 영화 비율 계산
    comparison_file = OUTPUT_DIR / "comparison_summary.csv"
    comparison_headers = [
        "seed_id", "seed_title", "seed_genres", "seed_year",
        "cosine_top1", "euclidean_top1",
        "top5_overlap_count", "top5_overlap_ratio",
        "top10_overlap_count", "top10_overlap_ratio",
        "top20_overlap_count", "top20_overlap_ratio",
        "cosine_top10_titles", "euclidean_top10_titles",
    ]

    with open(comparison_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(comparison_headers)

        for seed in seeds:
            sid = seed["id"]
            cos_list = cosine_results.get(sid, [])
            euc_list = euclidean_results.get(sid, [])

            # 자기 자신 제외한 결과에서 ID 추출
            cos_ids = [r["id"] for r in cos_list if r["id"] != sid]
            euc_ids = [r["id"] for r in euc_list if r["id"] != sid]

            # 각 top-K에서 겹치는 비율 계산
            def overlap(a: list, b: list, k: int) -> tuple[int, float]:
                """두 리스트의 상위 k개 요소 겹침 수와 비율을 반환한다."""
                set_a = set(a[:k])
                set_b = set(b[:k])
                common = len(set_a & set_b)
                ratio = common / k if k > 0 else 0
                return common, round(ratio, 3)

            top5_cnt, top5_ratio = overlap(cos_ids, euc_ids, 5)
            top10_cnt, top10_ratio = overlap(cos_ids, euc_ids, 10)
            top20_cnt, top20_ratio = overlap(cos_ids, euc_ids, 20)

            # Top-1 영화 제목
            cos_top1 = cos_list[0]["title"] if cos_list else "N/A"
            euc_top1 = euc_list[0]["title"] if euc_list else "N/A"

            # Top-10 제목 리스트 (자기 자신 제외)
            cos_top10_titles = " | ".join(
                r["title"] for r in cos_list if r["id"] != sid
            )[:200]
            euc_top10_titles = " | ".join(
                r["title"] for r in euc_list if r["id"] != sid
            )[:200]

            writer.writerow([
                sid,
                seed["title"],
                ", ".join(seed["genres"]),
                seed["release_year"],
                cos_top1,
                euc_top1,
                top5_cnt, top5_ratio,
                top10_cnt, top10_ratio,
                top20_cnt, top20_ratio,
                cos_top10_titles,
                euc_top10_titles,
            ])

    print(f"  비교 요약: {comparison_file}")


def main(
    seed_ids: list[int] | None = None,
    seed_count: int = 10,
    top_k: int = 20,
) -> None:
    """
    전체 유사도 검색 + Export 파이프라인.

    1. Qdrant 연결 + 시드 영화 선정
    2. 임시 Euclidean 컬렉션 생성 (벡터 복사)
    3. 각 시드 영화에 대해 Cosine / Euclidean 검색 수행
    4. 결과를 CSV 파일로 저장
    5. 임시 컬렉션 정리

    Args:
        seed_ids: 특정 시드 영화 TMDB ID 리스트 (None이면 인기도 상위)
        seed_count: 시드 영화 수 (seed_ids가 None일 때)
        top_k: 유사 영화 반환 수
    """
    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)

    try:
        # ── 1. 시드 영화 선정 ──
        print("\n[1/5] 시드 영화 선정")
        seeds = get_seed_movies(client, seed_ids, seed_count)
        for s in seeds:
            print(f"  #{s['id']:>6} | {s['title']} ({s['release_year']}) | "
                  f"장르: {', '.join(s['genres'])} | 인기도: {s['popularity_score']:.1f}")

        # ── 2. Euclidean 임시 컬렉션 생성 ──
        print("\n[2/5] Euclidean 임시 컬렉션 생성 (벡터 복사)")
        start_time = time.time()
        create_euclidean_collection(client)
        elapsed = time.time() - start_time
        print(f"  소요 시간: {elapsed:.1f}초")

        # ── 3. Cosine 유사도 검색 ──
        print(f"\n[3/5] Cosine 유사도 검색 (top-{top_k})")
        cosine_results = {}
        for seed in seeds:
            similar = search_similar(
                client, COSINE_COLLECTION, seed["vector"], top_k,
            )
            # 자기 자신을 제외한 결과만 유지
            similar = [s for s in similar if s["id"] != seed["id"]][:top_k]
            cosine_results[seed["id"]] = similar
            print(f"  #{seed['id']:>6} {seed['title']}: "
                  f"1위={similar[0]['title']} (score={similar[0]['score']:.4f})")

        # ── 4. Euclidean 유사도 검색 ──
        print(f"\n[4/5] Euclidean 유사도 검색 (top-{top_k})")
        euclidean_results = {}
        for seed in seeds:
            similar = search_similar(
                client, EUCLIDEAN_COLLECTION, seed["vector"], top_k,
            )
            # 자기 자신을 제외한 결과만 유지
            similar = [s for s in similar if s["id"] != seed["id"]][:top_k]
            euclidean_results[seed["id"]] = similar
            print(f"  #{seed['id']:>6} {seed['title']}: "
                  f"1위={similar[0]['title']} (score={similar[0]['score']:.4f})")

        # ── 5. CSV 파일로 Export ──
        print(f"\n[5/5] 결과 CSV 파일 저장")
        export_results(seeds, cosine_results, euclidean_results)

        # ── 콘솔에 비교 요약 출력 ──
        print("\n" + "=" * 80)
        print("  Cosine vs Euclidean 유사도 검색 비교 요약")
        print("=" * 80)
        print(f"  {'시드 영화':<30} {'Cosine Top-1':<25} {'Euclidean Top-1':<25} {'Top-10 겹침'}")
        print("-" * 80)

        for seed in seeds:
            sid = seed["id"]
            cos_list = cosine_results.get(sid, [])
            euc_list = euclidean_results.get(sid, [])

            cos_top1 = cos_list[0]["title"] if cos_list else "N/A"
            euc_top1 = euc_list[0]["title"] if euc_list else "N/A"

            # Top-10 겹침 비율
            cos_ids_10 = set(r["id"] for r in cos_list[:10])
            euc_ids_10 = set(r["id"] for r in euc_list[:10])
            overlap_10 = len(cos_ids_10 & euc_ids_10)

            print(f"  {seed['title'][:28]:<30} {cos_top1[:23]:<25} {euc_top1[:23]:<25} {overlap_10}/10")

        print("=" * 80)

        # ── 거리 메트릭 설명 출력 ──
        print("\n[참고] 거리 메트릭 해석:")
        print("  Cosine:    score가 1에 가까울수록 유사 (방향 유사도, 벡터 크기 무관)")
        print("  Euclidean: score가 0에 가까울수록 유사 (절대 거리, 벡터 크기 영향)")
        print(f"\n출력 디렉토리: {OUTPUT_DIR.resolve()}")

    finally:
        # ── 임시 Euclidean 컬렉션 정리 ──
        try:
            existing = [c.name for c in client.get_collections().collections]
            if EUCLIDEAN_COLLECTION in existing:
                client.delete_collection(EUCLIDEAN_COLLECTION)
                print(f"\n임시 컬렉션 '{EUCLIDEAN_COLLECTION}' 정리 완료")
        except Exception as e:
            print(f"\n[경고] 임시 컬렉션 정리 실패: {e}")

        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qdrant 벡터DB Cosine/Euclidean 유사도 검색 결과 Export"
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=10,
        help="시드 영화 수 — 인기도 상위 N개 (기본: 10)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="유사 영화 반환 수 (기본: 20)",
    )
    parser.add_argument(
        "--seed-ids",
        type=int,
        nargs="+",
        default=None,
        help="특정 시드 영화 TMDB ID 리스트 (예: --seed-ids 11 120 155)",
    )
    args = parser.parse_args()

    main(
        seed_ids=args.seed_ids,
        seed_count=args.seed_count,
        top_k=args.top_k,
    )
