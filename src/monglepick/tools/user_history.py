"""
사용자 시청 이력 MySQL 조회 도구 (Phase 6 Tool 6).

MySQL watch_history 테이블에서 사용자의 시청 이력을 조회한다.
"내가 본 영화 알려줘" 또는 "내가 본 거 빼고 추천해줘" 등의
사용자 요청에 응답하거나, 추천 제외 목록 구성에 활용된다.

MySQL 스키마 (context_loader의 쿼리 기준):
    watch_history (user_id, movie_id, rating, watched_at)
    movies (movie_id, title, genres, director, cast_members, mood_tags, ...)

aiomysql.DictCursor를 사용해 결과를 dict로 반환한다.
"""

from __future__ import annotations

import asyncio
import json

import aiomysql
import structlog
from langchain_core.tools import tool

from monglepick.db.clients import get_mysql

logger = structlog.get_logger()

# MySQL 쿼리 타임아웃 (초) — 커넥션 풀 획득 포함
_MYSQL_TIMEOUT_SEC = 5.0


@tool
async def user_history(
    user_id: str,
    limit: int = 20,
) -> list[dict]:
    """
    MySQL watch_history 테이블에서 사용자의 시청 이력을 조회한다.

    사용자가 "내가 본 영화" 또는 "이미 본 거 빼고"라고 요청할 때 활용한다.
    익명 사용자(user_id 빈 문자열)는 빈 리스트를 반환한다.

    Args:
        user_id: 사용자 ID (예: "user_abc123"). 빈 문자열이면 익명으로 처리.
        limit: 조회할 최대 시청 이력 수 (기본 20, 최대 100). 최신순 정렬.

    Returns:
        시청 이력 dict 목록 (최신 시청 순 내림차순):
        [
            {
                "movie_id": str,     # 영화 ID
                "title": str,        # 영화 제목
                "watched_at": str,   # 시청 일시 (ISO 8601 문자열)
                "rating": float,     # 사용자 평점 (0.0이면 평점 없음)
                "genres": list[str], # 장르 목록
                "director": str,     # 감독명
            }
        ]
        user_id 없음, DB 오류, 타임아웃 시: 빈 리스트 반환 (에러 전파 금지).
    """
    # 익명 사용자: DB 조회 없이 즉시 반환
    if not user_id:
        logger.info("user_history_tool_anonymous")
        return []

    # limit 범위 보정 (최소 1, 최대 100)
    safe_limit = max(1, min(int(limit), 100))

    try:
        pool = await get_mysql()

        async def _query() -> list[dict]:
            """
            MySQL 커넥션 풀에서 시청 이력을 조회하는 내부 코루틴.

            aiomysql.DictCursor를 사용하여 row를 dict로 반환한다.
            context_loader의 쿼리 패턴을 재사용하되 필요한 컬럼만 선택한다.
            """
            async with pool.acquire() as conn:
                # DictCursor: 결과 row를 컬럼명을 키로 하는 dict로 반환
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        """
                        SELECT
                            wh.movie_id,
                            m.title,
                            wh.watched_at,
                            wh.rating,
                            m.genres,
                            m.director
                        FROM watch_history wh
                        LEFT JOIN movies m ON wh.movie_id = m.movie_id
                        WHERE wh.user_id = %s
                        ORDER BY wh.watched_at DESC
                        LIMIT %s
                        """,
                        (user_id, safe_limit),
                    )
                    rows = await cursor.fetchall()
                    return list(rows)

        # 타임아웃 래핑 — 커넥션 풀 대기 + 쿼리 실행 포함
        raw_rows: list[dict] = await asyncio.wait_for(
            _query(),
            timeout=_MYSQL_TIMEOUT_SEC,
        )

        # 후처리: JSON 문자열 → list, datetime → ISO 8601 문자열
        results: list[dict] = []
        for row in raw_rows:
            # genres: DB에 JSON 문자열로 저장된 경우 파싱
            genres = row.get("genres", [])
            if isinstance(genres, str):
                try:
                    parsed = json.loads(genres)
                    genres = parsed if isinstance(parsed, list) else []
                except (json.JSONDecodeError, ValueError):
                    genres = []

            # watched_at: aiomysql이 datetime 객체로 반환 → ISO 8601 문자열 변환
            watched_at = row.get("watched_at")
            if watched_at is not None and hasattr(watched_at, "isoformat"):
                watched_at = watched_at.isoformat()
            elif watched_at is None:
                watched_at = ""

            results.append({
                "movie_id": str(row.get("movie_id", "")),
                "title": row.get("title", ""),
                "watched_at": str(watched_at),
                # rating: NULL이면 0.0으로 처리 (미평가 시청 이력)
                "rating": float(row.get("rating") or 0.0),
                "genres": genres,
                "director": row.get("director", ""),
            })

        logger.info(
            "user_history_tool_done",
            user_id=user_id,
            result_count=len(results),
            recent_titles=[r.get("title", "") for r in results[:3]],
        )
        return results

    except asyncio.TimeoutError:
        logger.error(
            "user_history_tool_timeout",
            user_id=user_id,
            timeout_sec=_MYSQL_TIMEOUT_SEC,
        )
        return []

    except Exception as e:
        # DB 연결 실패, 쿼리 오류 등 모든 예외 처리 (에러 전파 금지)
        logger.error(
            "user_history_tool_error",
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return []
