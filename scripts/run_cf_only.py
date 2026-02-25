"""
CF 매트릭스 구축 + Redis 캐시만 실행하는 스크립트.

파이프라인에서 TMDB/임베딩/적재가 이미 완료된 상태에서
Redis 메모리 부족 등으로 CF 캐시만 재실행할 때 사용한다.

사용법:
    uv run python scripts/run_cf_only.py --kaggle-dir ../theMoviesDataset/archive
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monglepick.data_pipeline.cf_builder import build_cf_matrix, cache_cf_to_redis  # noqa: E402
from monglepick.data_pipeline.kaggle_loader import KaggleLoader  # noqa: E402
from monglepick.db.clients import get_redis, close_all_clients  # noqa: E402


async def main(kaggle_dir: str) -> None:
    """CF 매트릭스 구축 후 Redis 캐시."""
    kaggle = KaggleLoader(kaggle_dir)

    # Kaggle 데이터 로드
    id_map = kaggle.load_links()
    ratings_df = kaggle.load_ratings(id_map)

    # CF 매트릭스 구축 (유사 유저 계산)
    similar_users, user_ratings, movie_avg = await build_cf_matrix(ratings_df)

    # Redis 캐시 저장
    await cache_cf_to_redis(similar_users, user_ratings, movie_avg)

    await close_all_clients()
    print("CF 캐시 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CF 매트릭스 + Redis 캐시 실행")
    parser.add_argument(
        "--kaggle-dir",
        type=str,
        default="data/kaggle_movies",
        help="Kaggle 데이터 디렉토리 경로",
    )
    args = parser.parse_args()
    asyncio.run(main(args.kaggle_dir))
