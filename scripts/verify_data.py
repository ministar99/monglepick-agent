"""
Phase 1 데이터 적재 검증 스크립트.

모든 DB에 데이터가 올바르게 적재되었는지 검증한다:
1. Qdrant (벡터 DB): 벡터 차원, 페이로드 필드, 샘플 검색
2. Neo4j (그래프 DB): 노드/관계 수, 관계 정확도
3. Elasticsearch (BM25): 인덱스 문서 수, 한국어 검색
4. Redis (CF 캐시): 유사 유저, 평점 캐시

사용법:
    uv run python scripts/verify_data.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monglepick.config import settings  # noqa: E402


async def verify_qdrant() -> dict:
    """Qdrant 벡터 DB 검증."""
    from qdrant_client import AsyncQdrantClient

    print("\n" + "=" * 60)
    print("1. QDRANT (벡터 DB) 검증")
    print("=" * 60)

    client = AsyncQdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
    results = {}

    # 컬렉션 정보
    info = await client.get_collection(settings.QDRANT_COLLECTION)
    results["total_points"] = info.points_count
    results["vector_dimension"] = info.config.params.vectors.size
    results["distance"] = info.config.params.vectors.distance.name
    results["status"] = info.status.name

    print(f"  컬렉션: {settings.QDRANT_COLLECTION}")
    print(f"  총 벡터 수: {info.points_count}")
    print(f"  벡터 차원: {info.config.params.vectors.size}")
    print(f"  거리 메트릭: {info.config.params.vectors.distance.name}")
    print(f"  상태: {info.status.name}")

    # 샘플 데이터 확인 (3건)
    points, _ = await client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        limit=3,
        with_vectors=False,
        with_payload=True,
    )

    print(f"\n  [샘플 데이터 (3건)]")
    required_fields = ["title", "genres", "director", "release_year", "rating", "mood_tags"]
    missing_fields_count = 0

    for p in points:
        payload = p.payload or {}
        title = payload.get("title", "N/A")
        genres = payload.get("genres", [])
        director = payload.get("director", "N/A")
        year = payload.get("release_year", "N/A")
        rating = payload.get("rating", "N/A")
        mood = payload.get("mood_tags", [])
        print(f"    [{p.id}] {title} ({year}) | 장르: {genres} | 감독: {director} | 평점: {rating} | 무드: {mood}")

        # 필수 필드 존재 확인
        for field in required_fields:
            if field not in payload or not payload[field]:
                missing_fields_count += 1

    results["missing_fields"] = missing_fields_count

    # 벡터 품질 확인: 임의 벡터로 유사도 검색
    sample_point, _ = await client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        limit=1,
        with_vectors=True,
        with_payload=True,
    )
    if sample_point:
        from qdrant_client.models import models
        search_results = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=sample_point[0].vector,
            limit=5,
        )
        print(f"\n  [벡터 유사도 검색 테스트] '{sample_point[0].payload.get('title')}'와 유사한 영화:")
        for sr in search_results.points[1:]:  # 자기 자신 제외
            print(f"    score={sr.score:.4f} | {sr.payload.get('title')} ({sr.payload.get('release_year')}) | 장르: {sr.payload.get('genres')}")
        results["search_test"] = "PASS"

    await client.close()

    # 판정
    passed = info.points_count > 3000 and info.config.params.vectors.size == 4096
    results["verdict"] = "PASS" if passed else "FAIL"
    print(f"\n  ✅ 판정: {results['verdict']} (벡터 {info.points_count}건, {info.config.params.vectors.size}차원)")
    return results


async def verify_neo4j() -> dict:
    """Neo4j 그래프 DB 검증."""
    from neo4j import AsyncGraphDatabase

    print("\n" + "=" * 60)
    print("2. NEO4J (그래프 DB) 검증")
    print("=" * 60)

    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    results = {}

    async with driver.session() as session:
        # 노드 수 확인 (6종)
        node_labels = ["Movie", "Person", "Genre", "Keyword", "MoodTag", "OTTPlatform"]
        print(f"  [노드 수]")
        for label in node_labels:
            result = await session.run(f"MATCH (n:{label}) RETURN count(n) as cnt")
            record = await result.single()
            count = record["cnt"]
            results[f"node_{label}"] = count
            print(f"    {label}: {count}")

        # 관계 수 확인 (6종)
        rel_types = ["DIRECTED", "ACTED_IN", "HAS_GENRE", "HAS_KEYWORD", "HAS_MOOD", "AVAILABLE_ON"]
        print(f"\n  [관계 수]")
        for rel in rel_types:
            result = await session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) as cnt")
            record = await result.single()
            count = record["cnt"]
            results[f"rel_{rel}"] = count
            print(f"    {rel}: {count}")

        # 관계 정확도 검증: 영화-감독 관계 샘플
        print(f"\n  [영화-감독 관계 샘플 (5건)]")
        result = await session.run("""
            MATCH (p:Person)-[:DIRECTED]->(m:Movie)
            RETURN m.title AS title, m.release_year AS year, p.name AS director
            LIMIT 5
        """)
        records = [record async for record in result]
        for r in records:
            print(f"    {r['title']} ({r['year']}) — 감독: {r['director']}")

        # 영화당 관계 수 통계
        print(f"\n  [영화당 평균 관계 수]")
        result = await session.run("""
            MATCH (m:Movie)
            OPTIONAL MATCH (m)-[r]-()
            WITH m, count(r) as rel_count
            RETURN avg(rel_count) as avg_rels, min(rel_count) as min_rels, max(rel_count) as max_rels
        """)
        record = await result.single()
        avg_rels = record["avg_rels"]
        print(f"    평균: {avg_rels:.1f}, 최소: {record['min_rels']}, 최대: {record['max_rels']}")
        results["avg_relations_per_movie"] = float(avg_rels)

        # 고립 노드 확인 (관계 없는 영화)
        result = await session.run("""
            MATCH (m:Movie)
            WHERE NOT (m)-[]-()
            RETURN count(m) as cnt
        """)
        record = await result.single()
        orphan_count = record["cnt"]
        results["orphan_movies"] = orphan_count
        print(f"    고립 영화 (관계 없음): {orphan_count}")

    await driver.close()

    # 판정
    movie_count = results.get("node_Movie", 0)
    passed = movie_count > 3000 and results.get("rel_DIRECTED", 0) > 0 and results.get("rel_HAS_GENRE", 0) > 0
    results["verdict"] = "PASS" if passed else "FAIL"
    print(f"\n  ✅ 판정: {results['verdict']} (Movie {movie_count}건, 6종 관계 구축)")
    return results


async def verify_elasticsearch() -> dict:
    """Elasticsearch 검증."""
    from elasticsearch import AsyncElasticsearch

    print("\n" + "=" * 60)
    print("3. ELASTICSEARCH (BM25 검색) 검증")
    print("=" * 60)

    client = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
    results = {}

    # 인덱스 정보
    count_resp = await client.count(index="movies_bm25")
    doc_count = count_resp["count"]
    results["total_docs"] = doc_count
    print(f"  인덱스: movies_bm25")
    print(f"  총 문서 수: {doc_count}")

    # Nori 분석기 테스트
    try:
        analyze_resp = await client.indices.analyze(
            index="movies_bm25",
            body={"analyzer": "korean_analyzer", "text": "감동적인 한국 영화"},
        )
        tokens = [t["token"] for t in analyze_resp["tokens"]]
        print(f"  Nori 분석기 테스트: '감동적인 한국 영화' → {tokens}")
        results["nori_analyzer"] = "PASS"
    except Exception as e:
        print(f"  Nori 분석기 오류: {e}")
        results["nori_analyzer"] = "FAIL"

    # 한국어 검색 테스트
    test_queries = [
        ("액션 영화", "장르별 검색"),
        ("크리스토퍼 놀란", "감독 검색"),
        ("우주 모험", "키워드 검색"),
    ]
    print(f"\n  [BM25 검색 테스트]")
    for query, desc in test_queries:
        resp = await client.search(
            index="movies_bm25",
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "director^2.5", "overview", "keywords^1.5", "genres"],
                    }
                },
                "size": 3,
            },
        )
        hits = resp["hits"]["hits"]
        print(f"    '{query}' ({desc}):")
        for hit in hits:
            src = hit["_source"]
            print(f"      score={hit['_score']:.2f} | {src.get('title')} ({src.get('release_year')}) | 장르: {src.get('genres')}")
        results[f"search_{query}"] = len(hits)

    await client.close()

    # 판정
    passed = doc_count > 3000 and results.get("nori_analyzer") == "PASS"
    results["verdict"] = "PASS" if passed else "FAIL"
    print(f"\n  ✅ 판정: {results['verdict']} ({doc_count}건, Nori 분석기 {results.get('nori_analyzer')})")
    return results


async def verify_redis() -> dict:
    """Redis CF 캐시 검증."""
    from redis.asyncio import Redis

    print("\n" + "=" * 60)
    print("4. REDIS (CF 캐시) 검증")
    print("=" * 60)

    client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    results = {}

    # 전체 키 수
    info = await client.info("keyspace")
    db_info = info.get("db0", {})
    total_keys = db_info.get("keys", 0)
    results["total_keys"] = total_keys
    print(f"  총 키 수: {total_keys}")

    # CF 키 유형별 수
    key_types = {
        "cf:similar_users:*": "유사 유저",
        "cf:user_ratings:*": "유저 평점",
        "cf:movie_avg_rating:*": "영화 평균 평점",
    }
    for pattern, desc in key_types.items():
        # SCAN으로 샘플 카운트 (전체 스캔은 비용이 크므로 제한)
        count = 0
        async for key in client.scan_iter(match=pattern, count=1000):
            count += 1
            if count >= 10000:
                break
        results[f"keys_{pattern}"] = count
        suffix = "+" if count >= 10000 else ""
        print(f"  {desc} ({pattern}): {count}{suffix}건")

    # 매트릭스 버전 확인
    version = await client.get("cf:matrix_version")
    results["matrix_version"] = version
    print(f"  매트릭스 버전: {version}")

    # 샘플 유사 유저 확인
    sample_key = None
    async for key in client.scan_iter(match="cf:similar_users:*", count=10):
        sample_key = key
        break

    if sample_key:
        similar = await client.zrevrange(sample_key, 0, 4, withscores=True)
        user_id = sample_key.split(":")[-1]
        print(f"\n  [유사 유저 샘플] user_id={user_id}")
        for sim_uid, score in similar:
            print(f"    유사유저={sim_uid}, 유사도={score:.4f}")
        results["sample_similar_users"] = len(similar)

    # 샘플 영화 평균 평점
    avg_key = None
    async for key in client.scan_iter(match="cf:movie_avg_rating:*", count=10):
        avg_key = key
        break

    if avg_key:
        avg_rating = await client.get(avg_key)
        movie_id = avg_key.split(":")[-1]
        print(f"\n  [영화 평균 평점 샘플] movie_id={movie_id} → 평균 평점: {avg_rating}")

    # 메모리 사용량
    mem_info = await client.info("memory")
    used_mb = mem_info.get("used_memory_human", "N/A")
    results["memory_used"] = used_mb
    print(f"\n  Redis 메모리 사용량: {used_mb}")

    await client.close()

    # 판정
    passed = total_keys > 100000 and version is not None
    results["verdict"] = "PASS" if passed else "FAIL"
    print(f"\n  ✅ 판정: {results['verdict']} ({total_keys}건, version={version})")
    return results


async def main():
    """전체 검증 실행."""
    print("=" * 60)
    print("  Phase 1 데이터 적재 검증")
    print("  대상: Qdrant / Neo4j / Elasticsearch / Redis")
    print("=" * 60)

    all_results = {}

    all_results["qdrant"] = await verify_qdrant()
    all_results["neo4j"] = await verify_neo4j()
    all_results["elasticsearch"] = await verify_elasticsearch()
    all_results["redis"] = await verify_redis()

    # 최종 요약
    print("\n" + "=" * 60)
    print("  최종 검증 결과 요약")
    print("=" * 60)

    all_pass = True
    for db, result in all_results.items():
        verdict = result.get("verdict", "N/A")
        icon = "✅" if verdict == "PASS" else "❌"
        print(f"  {icon} {db.upper()}: {verdict}")
        if verdict != "PASS":
            all_pass = False

    print(f"\n  {'✅ 전체 PASS — Phase 1 적재 검증 완료!' if all_pass else '❌ 일부 실패 — 위 결과를 확인하세요.'}")

    # 결과 파일 저장
    output_path = Path("data/verification_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  상세 결과 저장: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
