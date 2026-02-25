"""
협업 필터링(CF) 매트릭스 구축 및 Redis 캐시.

§11-7-3 CF 매트릭스 구축 및 Redis 캐시 구조:
1. ratings.csv → TMDB ID 매핑 (links.csv)
2. scipy.sparse.csr_matrix (138K × 10K) 구축
3. sklearn cosine_similarity → 유저별 Top-50 유사 유저
4. Redis 캐시 저장

Redis 키 구조:
- cf:similar_users:{user_id}  → Sorted Set (유사 유저 Top-50, score=유사도)
- cf:user_ratings:{user_id}   → Hash (영화별 평점)
- cf:movie_avg_rating:{movie_id} → String (영화 평균 평점)
- cf:matrix_version           → String (매트릭스 버전 타임스탬프)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import structlog
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from monglepick.db.clients import get_redis

logger = structlog.get_logger()

# Redis 키 접두사
KEY_SIMILAR_USERS = "cf:similar_users:{user_id}"
KEY_USER_RATINGS = "cf:user_ratings:{user_id}"
KEY_MOVIE_AVG_RATING = "cf:movie_avg_rating:{movie_id}"
KEY_MATRIX_VERSION = "cf:matrix_version"

# TTL (§11-7-3: 24시간)
CF_TTL_SECONDS = 86400  # 24h


async def build_cf_matrix(
    ratings_df: pd.DataFrame,
    top_k: int = 50,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, dict[int, float]], dict[int, float]]:
    """
    CF 유사 유저 매트릭스를 구축한다.

    §11-7-3 매트릭스 구축 흐름:
    [1] TMDB ID 매핑 완료된 ratings_df 입력
    [2] 스파스 매트릭스 구축 (user × movie)
    [3] 유사 유저 계산 (cosine_similarity)
    [4] Top-K 유사 유저 추출

    Args:
        ratings_df: columns=['userId', 'tmdbId', 'rating', 'timestamp']
        top_k: 유사 유저 수 (기본 50)

    Returns:
        similar_users: {user_id: [(similar_user_id, similarity_score), ...]}
        user_ratings: {user_id: {movie_id: rating, ...}}
        movie_avg_ratings: {movie_id: avg_rating}
    """
    logger.info("cf_matrix_build_started", ratings_count=len(ratings_df))

    # 유저/영화 인덱스 매핑 생성
    user_ids = ratings_df["userId"].unique()
    movie_ids = ratings_df["tmdbId"].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    # §11-7-3 [2]: 스파스 매트릭스 구축
    row = ratings_df["userId"].map(user_to_idx).values
    col = ratings_df["tmdbId"].map(movie_to_idx).values
    data = ratings_df["rating"].values

    matrix = csr_matrix(
        (data, (row, col)),
        shape=(len(user_ids), len(movie_ids)),
    )

    logger.info("cf_sparse_matrix_built", shape=matrix.shape, nnz=matrix.nnz)

    # §11-7-3 [3]: 유저별 코사인 유사도 계산
    # 메모리 절약을 위해 청크 단위로 계산
    similar_users: dict[int, list[tuple[int, float]]] = {}
    chunk_size = 1000

    for start in range(0, len(user_ids), chunk_size):
        end = min(start + chunk_size, len(user_ids))
        chunk_matrix = matrix[start:end]

        # 해당 청크의 유저 vs 전체 유저 유사도 계산
        sim = cosine_similarity(chunk_matrix, matrix)

        for local_idx in range(end - start):
            global_idx = start + local_idx
            uid = int(user_ids[global_idx])

            # 자기 자신 제외, Top-K 추출
            sim_scores = sim[local_idx]
            sim_scores[global_idx] = -1  # 자기 자신 제외

            top_indices = np.argsort(sim_scores)[-top_k:][::-1]
            top_users = [
                (int(user_ids[idx]), float(sim_scores[idx]))
                for idx in top_indices
                if sim_scores[idx] > 0
            ]

            similar_users[uid] = top_users

        if (end) % 10000 == 0 or end == len(user_ids):
            logger.info("cf_similarity_progress", completed=end, total=len(user_ids))

    # 유저별 평점 딕셔너리
    user_ratings: dict[int, dict[int, float]] = {}
    for _, row_data in ratings_df.iterrows():
        uid = int(row_data["userId"])
        mid = int(row_data["tmdbId"])
        rating = float(row_data["rating"])
        if uid not in user_ratings:
            user_ratings[uid] = {}
        user_ratings[uid][mid] = rating

    # 영화별 평균 평점
    movie_avg_ratings: dict[int, float] = (
        ratings_df.groupby("tmdbId")["rating"]
        .mean()
        .to_dict()
    )
    # 키를 int로 변환
    movie_avg_ratings = {int(k): float(v) for k, v in movie_avg_ratings.items()}

    logger.info(
        "cf_matrix_build_complete",
        users=len(similar_users),
        movies=len(movie_avg_ratings),
    )

    return similar_users, user_ratings, movie_avg_ratings


async def cache_cf_to_redis(
    similar_users: dict[int, list[tuple[int, float]]],
    user_ratings: dict[int, dict[int, float]],
    movie_avg_ratings: dict[int, float],
) -> None:
    """
    CF 매트릭스 결과를 Redis에 캐시한다.

    §11-7-3 Redis 캐시 키 구조:
    - cf:similar_users:{user_id} → Sorted Set (TTL 24h)
    - cf:user_ratings:{user_id}  → Hash (TTL 24h)
    - cf:movie_avg_rating:{movie_id} → String (TTL 24h)
    - cf:matrix_version → 현재 타임스탬프
    """
    redis = await get_redis()

    logger.info("cf_redis_cache_started")

    # 배치 크기 (메모리 안전을 위해 분할 실행)
    batch_size = 5000

    # 1. 유사 유저 Sorted Set (배치 분할)
    user_items = list(similar_users.items())
    for i in range(0, len(user_items), batch_size):
        batch = user_items[i : i + batch_size]
        pipe = redis.pipeline()
        for uid, sim_list in batch:
            key = KEY_SIMILAR_USERS.format(user_id=uid)
            pipe.delete(key)
            if sim_list:
                # zadd: {member: score} 형식
                mapping = {str(sim_uid): score for sim_uid, score in sim_list}
                pipe.zadd(key, mapping)
                pipe.expire(key, CF_TTL_SECONDS)
        await pipe.execute()

    logger.info("cf_similar_users_cached", count=len(similar_users))

    # 2. 유저별 평점 Hash (배치 분할)
    rating_items = list(user_ratings.items())
    for i in range(0, len(rating_items), batch_size):
        batch = rating_items[i : i + batch_size]
        pipe = redis.pipeline()
        for uid, ratings in batch:
            key = KEY_USER_RATINGS.format(user_id=uid)
            pipe.delete(key)
            if ratings:
                mapping = {str(mid): str(rating) for mid, rating in ratings.items()}
                pipe.hset(key, mapping=mapping)
                pipe.expire(key, CF_TTL_SECONDS)
        await pipe.execute()

    logger.info("cf_user_ratings_cached", count=len(user_ratings))

    # 3. 영화 평균 평점 String
    pipe = redis.pipeline()
    for mid, avg in movie_avg_ratings.items():
        key = KEY_MOVIE_AVG_RATING.format(movie_id=mid)
        pipe.set(key, str(round(avg, 2)), ex=CF_TTL_SECONDS)

    await pipe.execute()
    logger.info("cf_movie_avg_cached", count=len(movie_avg_ratings))

    # 4. 매트릭스 버전 타임스탬프
    version = datetime.now().strftime("%Y%m%d_%H%M")
    await redis.set(KEY_MATRIX_VERSION, version)

    logger.info("cf_redis_cache_complete", version=version)
