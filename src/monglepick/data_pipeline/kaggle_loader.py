"""
Kaggle The Movies Dataset 로더.

§11-4 Kaggle 데이터 로딩 명세:
- movies_metadata.csv: TMDB ID 정규화, 결측치 처리 → 보강용 DataFrame
- ratings.csv: userId·movieId·rating·timestamp → CF 매트릭스 입력
- credits.csv: JSON 파싱 (cast/crew) → 감독·배우 추출
- keywords.csv: JSON 파싱 → 키워드 목록
- links.csv: movieId → tmdbId → imdbId 매핑 테이블

§11-2: MovieLens movieId ↔ TMDB tmdbId 매핑은 links.csv로 수행.
매핑 실패 레코드는 제외하고 로그 기록 (~5% 예상).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger()

# 기본 Kaggle 데이터 경로
DEFAULT_DATA_DIR = Path("data/kaggle_movies")


class KaggleLoader:
    """
    Kaggle The Movies Dataset CSV 파일 로더.

    사용 예:
        loader = KaggleLoader("data/kaggle_movies")
        id_map = loader.load_links()
        metadata_df = loader.load_movies_metadata()
        ratings_df = loader.load_ratings(id_map)
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)

    def _path(self, filename: str) -> Path:
        return self.data_dir / filename

    # ── links.csv: ID 매핑 테이블 ──

    def load_links(self) -> dict[int, int]:
        """
        MovieLens movieId → TMDB tmdbId 매핑 딕셔너리를 생성한다.

        §11-2: links.csv의 movieId를 키로 tmdbId를 조회.
        tmdbId가 NaN이거나 숫자가 아닌 경우 제외.

        Returns:
            dict[int, int]: {movieLens_id: tmdb_id}
        """
        df = pd.read_csv(self._path("links.csv"))

        # tmdbId가 유효한 숫자인 행만 사용
        df = df.dropna(subset=["tmdbId"])
        df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")
        df = df.dropna(subset=["tmdbId"])

        id_map = dict(zip(df["movieId"].astype(int), df["tmdbId"].astype(int)))
        logger.info("kaggle_links_loaded", count=len(id_map))
        return id_map

    # ── movies_metadata.csv: 영화 메타데이터 ──

    def load_movies_metadata(self) -> pd.DataFrame:
        """
        영화 메타데이터를 로드하고 정규화한다.

        §11-4: TMDB ID 정규화 (숫자가 아닌 id 제외), 결측치 처리.

        Returns:
            DataFrame: id(int), title, overview, genres(list), release_date, vote_average, popularity 등
        """
        df = pd.read_csv(
            self._path("movies_metadata.csv"),
            low_memory=False,
        )

        # TMDB ID 정규화: 숫자가 아닌 id 제외
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.dropna(subset=["id"])
        df["id"] = df["id"].astype(int)

        # 중복 TMDB ID 제거 (첫 번째 유지)
        df = df.drop_duplicates(subset=["id"], keep="first")

        # genres 컬럼 파싱 (문자열 → 리스트)
        df["genres_parsed"] = df["genres"].apply(self._parse_json_column)

        # 결측치 처리
        df["overview"] = df["overview"].fillna("")
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0).astype(int)

        logger.info("kaggle_metadata_loaded", count=len(df))
        return df

    # ── ratings.csv: 유저-영화 평점 ──

    def load_ratings(self, id_map: dict[int, int] | None = None) -> pd.DataFrame:
        """
        평점 데이터를 로드하고 TMDB ID로 매핑한다.

        §11-4: userId·movieId·rating·timestamp 로드, links.csv로 TMDB ID 매핑.
        매핑 실패 레코드는 제외.

        Args:
            id_map: MovieLens movieId → TMDB tmdbId 매핑. None이면 자동 로드.

        Returns:
            DataFrame: userId, tmdbId, rating, timestamp
        """
        if id_map is None:
            id_map = self.load_links()

        df = pd.read_csv(self._path("ratings.csv"))

        # MovieLens movieId → TMDB tmdbId 매핑
        df["tmdbId"] = df["movieId"].map(id_map)
        original_count = len(df)

        # 매핑 실패 레코드 제외
        df = df.dropna(subset=["tmdbId"])
        df["tmdbId"] = df["tmdbId"].astype(int)

        dropped = original_count - len(df)
        logger.info(
            "kaggle_ratings_loaded",
            total=len(df),
            dropped=dropped,
            drop_rate=f"{dropped / original_count * 100:.1f}%",
        )
        return df[["userId", "tmdbId", "rating", "timestamp"]]

    # ── credits.csv: 감독/배우 정보 ──

    def load_credits(self) -> pd.DataFrame:
        """
        크레딧 데이터에서 감독과 상위 5명 배우를 추출한다.

        §11-4: JSON 파싱 (cast/crew 컬럼), 감독/상위 5 배우 추출.

        Returns:
            DataFrame: id(int), director(str), cast(list[str])
        """
        df = pd.read_csv(self._path("credits.csv"))

        # id 정규화
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.dropna(subset=["id"])
        df["id"] = df["id"].astype(int)

        # crew에서 감독 추출
        df["director"] = df["crew"].apply(self._extract_director)

        # cast에서 상위 5명 배우 추출
        df["cast_names"] = df["cast"].apply(lambda x: self._extract_cast(x, top_n=5))

        logger.info("kaggle_credits_loaded", count=len(df))
        return df[["id", "director", "cast_names"]]

    # ── keywords.csv: 키워드 목록 ──

    def load_keywords(self) -> pd.DataFrame:
        """
        키워드 데이터를 파싱한다.

        §11-4: JSON 파싱 (keywords 컬럼), 키워드 목록 추출.

        Returns:
            DataFrame: id(int), keywords(list[str])
        """
        df = pd.read_csv(self._path("keywords.csv"))

        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.dropna(subset=["id"])
        df["id"] = df["id"].astype(int)

        df["keywords_list"] = df["keywords"].apply(
            lambda x: [item["name"] for item in self._parse_json_column(x)]
        )

        logger.info("kaggle_keywords_loaded", count=len(df))
        return df[["id", "keywords_list"]]

    # ── JSON 파싱 헬퍼 ──

    @staticmethod
    def _parse_json_column(value: str) -> list[dict]:
        """문자열로 저장된 JSON/Python 리터럴을 파싱한다."""
        if pd.isna(value) or not value:
            return []
        try:
            return json.loads(value.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []

    @staticmethod
    def _extract_director(crew_str: str) -> str:
        """crew JSON에서 job='Director'인 사람의 이름을 추출한다."""
        crew = KaggleLoader._parse_json_column(crew_str)
        for person in crew:
            if person.get("job") == "Director":
                return person.get("name", "")
        return ""

    @staticmethod
    def _extract_cast(cast_str: str, top_n: int = 5) -> list[str]:
        """cast JSON에서 상위 N명의 배우 이름을 추출한다."""
        cast = KaggleLoader._parse_json_column(cast_str)
        return [person.get("name", "") for person in cast[:top_n] if person.get("name")]
