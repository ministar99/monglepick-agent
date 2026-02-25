"""
TMDB API 영화 데이터 수집기.

§11-5 TMDB 수집기 상세:
- collect_popular_movies, collect_top_rated_movies, collect_now_playing, collect_korean_movies
- collect_movie_details (상세 + 크레딧 + 키워드)
- collect_watch_providers (OTT 정보)
- Rate Limiting: asyncio.Semaphore(35) — TMDB 10초당 40회 제한에 여유분 확보
- 10,000편 수집 시 약 1~1.5시간 소요
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from monglepick.config import settings
from monglepick.data_pipeline.models import TMDBRawMovie

logger = structlog.get_logger()

# TMDB API Rate Limit: 10초당 40회 → 여유있게 35로 제한
_semaphore = asyncio.Semaphore(35)


class TMDBCollector:
    """
    TMDB API 비동기 수집기.

    사용 예:
        async with TMDBCollector() as collector:
            movies = await collector.collect_popular_movies(pages=50)
            for movie in movies:
                detail = await collector.collect_movie_details(movie["id"])
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url = settings.TMDB_BASE_URL
        self._api_key = settings.TMDB_API_KEY

    async def __aenter__(self) -> TMDBCollector:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            params={"api_key": self._api_key, "language": "ko-KR"},
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    # ── 내부 HTTP 호출 (Rate Limit 적용) ──

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
    async def _get(self, path: str, params: dict | None = None) -> dict:
        """Rate-limited GET 요청. 최대 3회 재시도 (지수 백오프)."""
        async with _semaphore:
            assert self._client is not None, "TMDBCollector must be used as async context manager"
            resp = await self._client.get(path, params=params or {})
            resp.raise_for_status()
            return resp.json()

    # ── 목록 수집 메서드 ──

    async def _collect_list(self, endpoint: str, pages: int, extra_params: dict | None = None) -> list[dict]:
        """페이지네이션 목록 API에서 영화 ID 목록을 수집한다."""
        results: list[dict] = []
        for page in range(1, pages + 1):
            params = {"page": page, **(extra_params or {})}
            data = await self._get(endpoint, params)
            results.extend(data.get("results", []))
            total_pages = data.get("total_pages", 1)
            if page >= total_pages:
                break
        logger.info("tmdb_list_collected", endpoint=endpoint, count=len(results))
        return results

    async def collect_popular_movies(self, pages: int = 50) -> list[dict]:
        """인기 영화 수집 (~1,000편). §11-5: /movie/popular"""
        return await self._collect_list("/movie/popular", pages)

    async def collect_top_rated_movies(self, pages: int = 50) -> list[dict]:
        """높은 평점 영화 수집 (~1,000편). §11-5: /movie/top_rated"""
        return await self._collect_list("/movie/top_rated", pages)

    async def collect_now_playing(self) -> list[dict]:
        """현재 상영 중 영화 수집 (~40편). §11-5: /movie/now_playing"""
        return await self._collect_list("/movie/now_playing", pages=2)

    async def collect_korean_movies(self, pages: int = 100) -> list[dict]:
        """한국 영화 수집 (~2,000편). §11-5: /discover/movie?region=KR"""
        return await self._collect_list(
            "/discover/movie",
            pages,
            extra_params={"region": "KR", "with_original_language": "ko", "sort_by": "popularity.desc"},
        )

    # ── 상세 수집 메서드 ──

    async def collect_movie_details(self, movie_id: int) -> TMDBRawMovie:
        """
        영화 상세정보 수집 (크레딧 + 키워드 포함).

        §11-5: /movie/{id}?append_to_response=credits,keywords
        """
        data = await self._get(
            f"/movie/{movie_id}",
            params={"append_to_response": "credits,keywords"},
        )
        return TMDBRawMovie(
            id=data.get("id", movie_id),
            title=data.get("title", ""),
            original_title=data.get("original_title", ""),
            overview=data.get("overview", ""),
            release_date=data.get("release_date", ""),
            vote_average=data.get("vote_average", 0.0),
            popularity=data.get("popularity", 0.0),
            poster_path=data.get("poster_path"),
            runtime=data.get("runtime"),
            genres=data.get("genres", []),
            credits=data.get("credits", {}),
            keywords=data.get("keywords", {}),
        )

    async def collect_watch_providers(self, movie_id: int) -> dict:
        """
        OTT 제공 정보 수집.

        §11-5: /movie/{id}/watch/providers → 한국(KR) flatrate 기준
        """
        data = await self._get(f"/movie/{movie_id}/watch/providers")
        return data.get("results", {})

    # ── 대량 수집 오케스트레이션 ──

    async def collect_all_movie_ids(self) -> list[int]:
        """
        4개 목록 API에서 중복 제거된 전체 영화 ID를 수집한다.

        §11-5: popular + top_rated + now_playing + korean → ~10,000편 (중복 제거 후)
        """
        # 4개 목록 동시 수집
        popular, top_rated, now_playing, korean = await asyncio.gather(
            self.collect_popular_movies(pages=50),
            self.collect_top_rated_movies(pages=50),
            self.collect_now_playing(),
            self.collect_korean_movies(pages=100),
        )

        # 중복 제거 (TMDB ID 기준)
        seen: set[int] = set()
        unique_ids: list[int] = []
        for movie in popular + top_rated + now_playing + korean:
            mid = movie.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                unique_ids.append(mid)

        logger.info("tmdb_all_ids_collected", total=len(unique_ids))
        return unique_ids

    async def collect_full_details(self, movie_ids: list[int]) -> list[TMDBRawMovie]:
        """
        영화 ID 목록에 대해 상세정보 + OTT 정보를 수집한다.

        §11-5: 영화당 상세 API + OTT API 호출.
        개별 실패 시 로그 기록 후 건너뛴다 (§11-9: 부분 실패 허용).
        """
        results: list[TMDBRawMovie] = []

        for i, mid in enumerate(movie_ids):
            try:
                # 상세 정보 수집
                movie = await self.collect_movie_details(mid)

                # OTT 정보 수집 및 병합
                providers = await self.collect_watch_providers(mid)
                movie.watch_providers = providers

                results.append(movie)

                # 진행률 로깅 (500건마다)
                if (i + 1) % 500 == 0:
                    logger.info("tmdb_detail_progress", completed=i + 1, total=len(movie_ids))

            except Exception as e:
                logger.warning("tmdb_detail_failed", movie_id=mid, error=str(e))
                continue

        logger.info("tmdb_full_details_collected", success=len(results), total=len(movie_ids))
        return results
