# 몽글픽 DB 예시 쿼리

5개 DB(Qdrant, Neo4j, Elasticsearch, Redis, MySQL)의 주요 쿼리 예시.

---

## 1. Qdrant (벡터 DB)

### 컬렉션 정보 조회
```bash
curl http://localhost:6333/collections/movies
```

### 벡터 유사도 검색 (영화 추천)
```bash
# 특정 영화(인터스텔라, ID: 157336)와 유사한 영화 5편
curl -X POST http://localhost:6333/collections/movies/points/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": 157336,
    "limit": 5,
    "with_payload": ["title", "genres", "rating", "release_year"]
  }'
```

### 필터 검색 (장르 + 평점)
```bash
curl -X POST http://localhost:6333/collections/movies/points/scroll \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {
      "must": [
        {"key": "genres", "match": {"value": "SF"}},
        {"key": "rating", "range": {"gte": 8.0}}
      ]
    },
    "limit": 10,
    "with_payload": ["title", "rating", "release_year", "genres"]
  }'
```

### 제목으로 영화 검색
```bash
curl -X POST http://localhost:6333/collections/movies/points/scroll \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {
      "must": [
        {"key": "title", "match": {"value": "인터스텔라"}}
      ]
    },
    "limit": 1,
    "with_payload": true
  }'
```

### 포인트 수 확인
```bash
curl http://localhost:6333/collections/movies | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])"
```

---

## 2. Neo4j (그래프 DB)

> 접속: `http://localhost:7474` (Browser UI)
> 인증: `neo4j` / `monglepick_dev`

### 영화 노드 조회
```cypher
-- 인터스텔라 상세 정보
MATCH (m:Movie {title: "인터스텔라"})
RETURN m.title, m.rating, m.release_year, m.certification, m.runtime

-- 전체 영화 수
MATCH (m:Movie) RETURN count(m) AS total_movies
```

### 감독/배우 관계 조회
```cypher
-- 봉준호 감독의 영화
MATCH (p:Person {name: "봉준호"})-[:DIRECTED]->(m:Movie)
RETURN m.title, m.release_year, m.rating
ORDER BY m.rating DESC

-- 특정 영화의 출연진
MATCH (m:Movie {title: "기생충"})<-[:ACTED_IN]-(a:Person)
RETURN a.name
```

### 그래프 탐색 (추천용)
```cypher
-- 인터스텔라와 같은 장르의 영화 (평점순)
MATCH (m:Movie {title: "인터스텔라"})-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(rec:Movie)
WHERE rec <> m
RETURN rec.title, rec.rating, collect(DISTINCT g.name) AS shared_genres
ORDER BY rec.rating DESC
LIMIT 10

-- 같은 감독 + 같은 장르 영화
MATCH (m:Movie {title: "기생충"})-[:HAS_GENRE]->(g:Genre),
      (m)<-[:DIRECTED]-(d:Person)-[:DIRECTED]->(rec:Movie)-[:HAS_GENRE]->(g)
WHERE rec <> m
RETURN rec.title, d.name AS director, collect(DISTINCT g.name) AS genres
```

### 노드/관계 통계
```cypher
-- 전체 노드 라벨별 수
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC

-- 전체 관계 타입별 수
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC
```

---

## 3. Elasticsearch (한국어 검색)

### 인덱스 정보
```bash
# 문서 수
curl http://localhost:9200/movies/_count

# 인덱스 매핑 확인
curl http://localhost:9200/movies/_mapping
```

### 한국어 전문 검색 (BM25)
```bash
# "우주 탐험" 검색
curl -X POST http://localhost:9200/movies/_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "multi_match": {
        "query": "우주 탐험 SF",
        "fields": ["title^3", "overview^2", "genres", "keywords"],
        "type": "best_fields"
      }
    },
    "size": 5,
    "_source": ["title", "overview", "genres", "rating"]
  }'
```

### 필터 + 정렬
```bash
# 2020년 이후, 평점 8.0 이상, 평점순 정렬
curl -X POST http://localhost:9200/movies/_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "bool": {
        "must": [{"match": {"genres": "액션"}}],
        "filter": [
          {"range": {"release_year": {"gte": 2020}}},
          {"range": {"rating": {"gte": 8.0}}}
        ]
      }
    },
    "sort": [{"rating": "desc"}],
    "size": 10,
    "_source": ["title", "rating", "release_year", "genres"]
  }'
```

### 감독/배우 검색
```bash
# 봉준호 감독 영화
curl -X POST http://localhost:9200/movies/_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {"match": {"director": "봉준호"}},
    "size": 10,
    "_source": ["title", "director", "rating", "release_year"]
  }'
```

### 집계 (Aggregation)
```bash
# 장르별 영화 수
curl -X POST http://localhost:9200/movies/_search \
  -H "Content-Type: application/json" \
  -d '{
    "size": 0,
    "aggs": {
      "genres": {
        "terms": {"field": "genres.keyword", "size": 20}
      }
    }
  }'
```

---

## 4. Redis (CF 캐시)

### 기본 상태 확인
```bash
docker exec monglepick-redis redis-cli INFO keyspace
docker exec monglepick-redis redis-cli DBSIZE
docker exec monglepick-redis redis-cli INFO memory | grep used_memory_human
```

### CF 캐시 조회

```bash
# 유저의 유사 유저 Top-50 조회
docker exec monglepick-redis redis-cli GET "cf:similar_users:1"

# 유저의 평점 데이터 조회
docker exec monglepick-redis redis-cli GET "cf:user_ratings:1"

# 키 패턴 검색 (유사유저 키 샘플)
docker exec monglepick-redis redis-cli SCAN 0 MATCH "cf:similar_users:*" COUNT 10

# 키 패턴 검색 (평점 키 샘플)
docker exec monglepick-redis redis-cli SCAN 0 MATCH "cf:user_ratings:*" COUNT 10
```

### 키 타입/TTL 확인
```bash
docker exec monglepick-redis redis-cli TYPE "cf:similar_users:1"
docker exec monglepick-redis redis-cli TTL "cf:similar_users:1"
```

---

## 5. MySQL (RDB)

> 접속: `docker exec -it monglepick-mysql mysql -u monglepick -pmonglepick_dev --default-character-set=utf8mb4 monglepick`

### 테이블 목록 및 건수
```sql
-- 전체 테이블 목록
SHOW TABLES;

-- 주요 테이블 건수
SELECT 'movies' AS tbl, COUNT(*) AS cnt FROM movies
UNION ALL SELECT 'users', COUNT(*) FROM users
UNION ALL SELECT 'watch_history', COUNT(*) FROM watch_history
UNION ALL SELECT 'tables', COUNT(*) FROM information_schema.tables WHERE table_schema='monglepick';
```

### 영화 조회
```sql
-- 인터스텔라 조회
SELECT movie_id, title, title_en, rating, release_year, genres, director, runtime
FROM movies WHERE title = '인터스텔라';

-- 평점 높은 영화 Top 10
SELECT title, rating, vote_count, release_year, genres
FROM movies
WHERE vote_count > 1000
ORDER BY rating DESC
LIMIT 10;

-- 장르별 영화 수
SELECT
  JSON_UNQUOTE(j.genre) AS genre,
  COUNT(*) AS cnt
FROM movies,
  JSON_TABLE(genres, '$[*]' COLUMNS(genre VARCHAR(50) PATH '$')) AS j
GROUP BY genre
ORDER BY cnt DESC
LIMIT 10;
```

### 시청 이력 조회
```sql
-- 특정 유저의 시청 이력 (최근 10건)
SELECT wh.movie_id, m.title, wh.rating, wh.watched_at
FROM watch_history wh
JOIN movies m ON wh.movie_id = m.movie_id
WHERE wh.user_id = 'kaggle_1'
ORDER BY wh.watched_at DESC
LIMIT 10;

-- 가장 많이 본 영화 Top 10
SELECT m.title, COUNT(*) AS view_count, AVG(wh.rating) AS avg_rating
FROM watch_history wh
JOIN movies m ON wh.movie_id = m.movie_id
GROUP BY wh.movie_id, m.title
ORDER BY view_count DESC
LIMIT 10;

-- 활발한 유저 Top 10
SELECT user_id, COUNT(*) AS watch_count, ROUND(AVG(rating), 2) AS avg_rating
FROM watch_history
GROUP BY user_id
ORDER BY watch_count DESC
LIMIT 10;
```

### 연도별 통계
```sql
-- 연도별 영화 수
SELECT release_year, COUNT(*) AS cnt
FROM movies
WHERE release_year IS NOT NULL
GROUP BY release_year
ORDER BY release_year DESC
LIMIT 20;
```

---

## DB 접속 정보 요약

| DB | 포트 | 접속 방법 |
|---|---|---|
| Qdrant | 6333 (REST), 6334 (gRPC) | `curl http://localhost:6333` |
| Neo4j | 7474 (Browser), 7687 (Bolt) | `http://localhost:7474` |
| Elasticsearch | 9200 | `curl http://localhost:9200` |
| Redis | 6379 | `docker exec monglepick-redis redis-cli` |
| MySQL | 3306 | `docker exec -it monglepick-mysql mysql -u monglepick -pmonglepick_dev monglepick` |

## 현재 데이터 현황

| DB | 건수 |
|---|---|
| Qdrant | 157,194 벡터 |
| Neo4j | 157,194 Movie + 222,302 Person |
| Elasticsearch | 157,194 문서 |
| Redis | 586,647 CF 캐시 키 |
| MySQL | 157,194 movies + 26,010,786 watch_history + 270,888 users |
