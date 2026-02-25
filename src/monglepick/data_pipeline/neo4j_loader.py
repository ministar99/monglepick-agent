"""
Neo4j 지식그래프 적재기.

§11-7 Neo4j 그래프 구축 명세:
- 노드: Movie, Person, Genre, Keyword, MoodTag, OTTPlatform
- 관계: DIRECTED, ACTED_IN, HAS_GENRE, HAS_KEYWORD, HAS_MOOD, AVAILABLE_ON, SIMILAR_TO

§11-7-2 배치 적재 패턴:
- UNWIND 배치로 노드/관계 생성 (500건/배치)
- 모든 노드/관계에 MERGE 사용 (중복 방지)
- 적재 순서: 노드 → 관계 → SIMILAR_TO → 인덱스 확인
"""

from __future__ import annotations

import structlog

from monglepick.data_pipeline.models import MovieDocument
from monglepick.db.clients import get_neo4j

logger = structlog.get_logger()

# 배치 크기 (§11-7-2)
NODE_BATCH_SIZE = 500
RELATION_BATCH_SIZE = 500


async def _run_batch(cypher: str, params: dict) -> None:
    """Neo4j Cypher 쿼리를 실행한다."""
    driver = await get_neo4j()
    async with driver.session() as session:
        await session.run(cypher, params)


async def _batch_execute(cypher: str, data: list[dict], batch_size: int, label: str) -> None:
    """데이터를 배치로 나누어 UNWIND Cypher를 실행한다."""
    if not data:
        logger.info(f"neo4j_{label}_skipped", count=0)
        return

    # batch_size가 0이면 기본값 사용 (OTT 등 빈 데이터 방어)
    if batch_size <= 0:
        batch_size = NODE_BATCH_SIZE

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        await _run_batch(cypher, {"batch": batch})

    logger.info(f"neo4j_{label}_loaded", count=len(data))


# ============================================================
# Step 1: 노드 생성
# ============================================================

async def load_movie_nodes(documents: list[MovieDocument]) -> None:
    """
    Movie 노드를 배치 생성한다.

    §11-7-2: 500건/배치, MERGE 사용
    """
    data = [
        {
            "id": doc.id,
            "title": doc.title,
            "title_en": doc.title_en,
            "release_year": doc.release_year,
            "rating": doc.rating,
            "popularity_score": doc.popularity_score,
        }
        for doc in documents
    ]

    cypher = """
    UNWIND $batch AS m
    MERGE (movie:Movie {id: m.id})
    SET movie.title = m.title,
        movie.title_en = m.title_en,
        movie.release_year = m.release_year,
        movie.rating = m.rating,
        movie.popularity_score = m.popularity_score
    """
    await _batch_execute(cypher, data, NODE_BATCH_SIZE, "movie_nodes")


async def load_genre_nodes(documents: list[MovieDocument]) -> None:
    """Genre 노드를 생성한다. (~20개, 배치 불필요)"""
    genres = list({genre for doc in documents for genre in doc.genres})
    data = [{"name": g} for g in genres]

    cypher = """
    UNWIND $batch AS g
    MERGE (:Genre {name: g.name})
    """
    await _batch_execute(cypher, data, len(data), "genre_nodes")


async def load_mood_tag_nodes(documents: list[MovieDocument]) -> None:
    """MoodTag 노드를 생성한다. (~25개, 배치 불필요)"""
    tags = list({tag for doc in documents for tag in doc.mood_tags})
    data = [{"name": t} for t in tags]

    cypher = """
    UNWIND $batch AS t
    MERGE (:MoodTag {name: t.name})
    """
    await _batch_execute(cypher, data, len(data), "mood_tag_nodes")


async def load_ott_nodes(documents: list[MovieDocument]) -> None:
    """OTTPlatform 노드를 생성한다. (~10개, 배치 불필요)"""
    platforms = list({p for doc in documents for p in doc.ott_platforms})
    data = [{"name": p} for p in platforms]

    cypher = """
    UNWIND $batch AS p
    MERGE (:OTTPlatform {name: p.name})
    """
    await _batch_execute(cypher, data, len(data), "ott_nodes")


async def load_person_nodes(documents: list[MovieDocument]) -> None:
    """Person 노드를 생성한다 (감독 + 배우 합집합)."""
    persons = set()
    for doc in documents:
        if doc.director:
            persons.add(doc.director)
        persons.update(doc.cast)

    data = [{"name": p} for p in persons if p]

    cypher = """
    UNWIND $batch AS p
    MERGE (:Person {name: p.name})
    """
    await _batch_execute(cypher, data, NODE_BATCH_SIZE, "person_nodes")


async def load_keyword_nodes(documents: list[MovieDocument]) -> None:
    """Keyword 노드를 배치 생성한다."""
    keywords = list({kw for doc in documents for kw in doc.keywords})
    data = [{"name": k} for k in keywords if k]

    cypher = """
    UNWIND $batch AS k
    MERGE (:Keyword {name: k.name})
    """
    await _batch_execute(cypher, data, NODE_BATCH_SIZE, "keyword_nodes")


# ============================================================
# Step 2: 관계 생성
# ============================================================

async def load_directed_relations(documents: list[MovieDocument]) -> None:
    """DIRECTED 관계를 배치 생성한다. §11-7: director 필드가 존재하고 비어있지 않을 때."""
    data = [
        {"movie_id": doc.id, "name": doc.director}
        for doc in documents
        if doc.director
    ]

    cypher = """
    UNWIND $batch AS d
    MATCH (p:Person {name: d.name})
    MATCH (m:Movie {id: d.movie_id})
    MERGE (p)-[:DIRECTED]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "directed_relations")


async def load_acted_in_relations(documents: list[MovieDocument]) -> None:
    """ACTED_IN 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": actor}
        for doc in documents
        for actor in doc.cast
        if actor
    ]

    cypher = """
    UNWIND $batch AS a
    MATCH (p:Person {name: a.name})
    MATCH (m:Movie {id: a.movie_id})
    MERGE (p)-[:ACTED_IN]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "acted_in_relations")


async def load_has_genre_relations(documents: list[MovieDocument]) -> None:
    """HAS_GENRE 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": genre}
        for doc in documents
        for genre in doc.genres
    ]

    cypher = """
    UNWIND $batch AS g
    MATCH (m:Movie {id: g.movie_id})
    MATCH (genre:Genre {name: g.name})
    MERGE (m)-[:HAS_GENRE]->(genre)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "has_genre_relations")


async def load_has_keyword_relations(documents: list[MovieDocument]) -> None:
    """HAS_KEYWORD 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": kw}
        for doc in documents
        for kw in doc.keywords
        if kw
    ]

    cypher = """
    UNWIND $batch AS k
    MATCH (m:Movie {id: k.movie_id})
    MATCH (kw:Keyword {name: k.name})
    MERGE (m)-[:HAS_KEYWORD]->(kw)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "has_keyword_relations")


async def load_has_mood_relations(documents: list[MovieDocument]) -> None:
    """HAS_MOOD 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": tag}
        for doc in documents
        for tag in doc.mood_tags
    ]

    cypher = """
    UNWIND $batch AS t
    MATCH (m:Movie {id: t.movie_id})
    MATCH (mt:MoodTag {name: t.name})
    MERGE (m)-[:HAS_MOOD]->(mt)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "has_mood_relations")


async def load_available_on_relations(documents: list[MovieDocument]) -> None:
    """AVAILABLE_ON 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": platform}
        for doc in documents
        for platform in doc.ott_platforms
    ]

    cypher = """
    UNWIND $batch AS o
    MATCH (m:Movie {id: o.movie_id})
    MATCH (ott:OTTPlatform {name: o.name})
    MERGE (m)-[:AVAILABLE_ON]->(ott)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "available_on_relations")


# ============================================================
# 전체 적재 오케스트레이션
# ============================================================

async def load_to_neo4j(documents: list[MovieDocument]) -> None:
    """
    MovieDocument 리스트를 Neo4j 그래프에 적재한다.

    §11-7-2 적재 순서: 노드 → 관계
    SIMILAR_TO는 별도 배치(§10-3-2)에서 처리하므로 여기서는 생략한다.
    """
    logger.info("neo4j_load_started", count=len(documents))

    # Step 1: 노드 생성 (병렬 가능한 것은 병렬)
    await load_movie_nodes(documents)
    await load_person_nodes(documents)
    await load_genre_nodes(documents)
    await load_keyword_nodes(documents)
    await load_mood_tag_nodes(documents)
    await load_ott_nodes(documents)

    # Step 2: 관계 생성 (노드 생성 완료 후)
    await load_directed_relations(documents)
    await load_acted_in_relations(documents)
    await load_has_genre_relations(documents)
    await load_has_keyword_relations(documents)
    await load_has_mood_relations(documents)
    await load_available_on_relations(documents)

    logger.info("neo4j_load_complete", count=len(documents))
