"""
데이터 전처리기.

§11-6 전처리기 상세:
1. 장르 한국어 변환 (TMDB genre ID → 한국어)
2. 무드태그 생성 (Ollama qwen2.5:14b, 25개 화이트리스트)
3. 임베딩 입력 텍스트 구성 (구조화된 텍스트)
4. OTT 플랫폼명 정규화 ("Netflix" → "넷플릭스")

§11-3 데이터 검증 규칙:
- 필수 필드: id, title, genres, release_year
- 평점 범위: 0~10
- 개봉년도: 1900~현재
"""

from __future__ import annotations

import json
from datetime import datetime

import structlog

from monglepick.config import settings
from monglepick.data_pipeline.models import MovieDocument, TMDBRawMovie

logger = structlog.get_logger()

# ============================================================
# 상수: 장르 매핑 테이블
# ============================================================

# TMDB genre_id → 한국어 (§11-6 단계 1)
GENRE_ID_TO_KR: dict[int, str] = {
    28: "액션", 12: "모험", 16: "애니메이션", 35: "코미디",
    80: "범죄", 99: "다큐멘터리", 18: "드라마", 10751: "가족",
    14: "판타지", 36: "역사", 27: "공포", 10402: "음악",
    9648: "미스터리", 10749: "로맨스", 878: "SF", 10770: "TV 영화",
    53: "스릴러", 10752: "전쟁", 37: "서부",
}

# 영문 장르명 → 한국어 (KOBIS/Kaggle 보강용)
GENRE_EN_TO_KR: dict[str, str] = {
    "Action": "액션", "Adventure": "모험", "Animation": "애니메이션",
    "Comedy": "코미디", "Crime": "범죄", "Documentary": "다큐멘터리",
    "Drama": "드라마", "Family": "가족", "Fantasy": "판타지",
    "History": "역사", "Horror": "공포", "Music": "음악",
    "Mystery": "미스터리", "Romance": "로맨스", "Science Fiction": "SF",
    "TV Movie": "TV 영화", "Thriller": "스릴러", "War": "전쟁", "Western": "서부",
}

# ============================================================
# 상수: 무드태그 관련
# ============================================================

# 25개 허용 무드태그 화이트리스트 (§11-6)
MOOD_WHITELIST: set[str] = {
    "몰입", "감동", "웅장", "긴장감", "힐링", "유쾌", "따뜻", "슬픔",
    "공포", "잔잔", "스릴", "카타르시스", "청춘", "우정", "가족애",
    "로맨틱", "미스터리", "반전", "철학적", "사회비판", "모험", "판타지",
    "레트로", "다크", "유머",
}

# §11-6-1 장르 → 무드 기본 매핑 테이블 (GPT 실패 시 fallback)
GENRE_TO_DEFAULT_MOOD: dict[str, list[str]] = {
    "액션": ["몰입", "스릴"], "모험": ["모험", "몰입"], "애니메이션": ["따뜻", "판타지"],
    "코미디": ["유쾌", "유머"], "범죄": ["긴장감", "다크"], "다큐멘터리": ["철학적", "사회비판"],
    "드라마": ["감동", "잔잔"], "가족": ["가족애", "따뜻"], "판타지": ["판타지", "모험"],
    "역사": ["웅장", "감동"], "공포": ["공포", "다크"], "음악": ["감동", "힐링"],
    "미스터리": ["미스터리", "긴장감"], "로맨스": ["로맨틱", "따뜻"], "SF": ["몰입", "웅장"],
    "TV 영화": ["잔잔"], "스릴러": ["스릴", "긴장감"], "전쟁": ["웅장", "카타르시스"],
    "서부": ["모험", "레트로"],
}

# ============================================================
# 상수: OTT 플랫폼 정규화 테이블
# ============================================================

# §11-6 단계 4: 영문 → 한국어
OTT_NORMALIZE: dict[str, str] = {
    "Netflix": "넷플릭스", "Disney Plus": "디즈니+", "Amazon Prime Video": "아마존 프라임",
    "Wavve": "웨이브", "Watcha": "왓챠", "Tving": "티빙",
    "Coupang Play": "쿠팡플레이", "Apple TV Plus": "애플TV+", "Apple TV": "애플TV+",
    "Google Play Movies": "구글플레이", "YouTube": "유튜브",
    "Naver Store": "네이버", "KakaoPage": "카카오페이지",
}

# ============================================================
# 전처리 함수
# ============================================================


def convert_genres(raw_genres: list[dict]) -> list[str]:
    """
    TMDB 장르 객체 배열을 한국어 장르 문자열 리스트로 변환한다.

    §11-6 단계 1: TMDB genre ID → 한국어.
    변환 실패 시 영문명 유지 (§11-3).

    Args:
        raw_genres: [{"id": 878, "name": "Science Fiction"}, ...]

    Returns:
        ["SF", "드라마", ...]
    """
    result: list[str] = []
    for genre in raw_genres:
        genre_id = genre.get("id")
        genre_name = genre.get("name", "")

        # 1차: ID로 변환
        if genre_id and genre_id in GENRE_ID_TO_KR:
            result.append(GENRE_ID_TO_KR[genre_id])
        # 2차: 영문명으로 변환
        elif genre_name in GENRE_EN_TO_KR:
            result.append(GENRE_EN_TO_KR[genre_name])
        # 3차: 변환 실패 시 영문명 유지
        elif genre_name:
            result.append(genre_name)

    return result


def extract_director(credits: dict) -> str:
    """credits.crew에서 job='Director'인 사람의 이름을 추출한다."""
    crew = credits.get("crew", [])
    for person in crew:
        if person.get("job") == "Director":
            return person.get("name", "")
    return ""


def extract_cast(credits: dict, top_n: int = 5) -> list[str]:
    """credits.cast에서 상위 N명 배우 이름을 추출한다."""
    cast = credits.get("cast", [])
    return [person.get("name", "") for person in cast[:top_n] if person.get("name")]


def extract_keywords(keywords_data: dict) -> list[str]:
    """keywords 객체에서 키워드 이름 리스트를 추출한다."""
    keywords = keywords_data.get("keywords", [])
    return [kw.get("name", "") for kw in keywords if kw.get("name")]


def normalize_ott_platforms(watch_providers: dict) -> list[str]:
    """
    TMDB watch/providers 응답에서 한국(KR) OTT 플랫폼 목록을 추출하고 한국어로 정규화한다.

    §11-6 단계 4: "Netflix" → "넷플릭스"
    """
    kr_providers = watch_providers.get("KR", {})
    flatrate = kr_providers.get("flatrate", [])

    result: list[str] = []
    for provider in flatrate:
        name = provider.get("provider_name", "")
        # 정규화 매핑이 있으면 적용, 없으면 원본 유지
        normalized = OTT_NORMALIZE.get(name, name)
        if normalized and normalized not in result:
            result.append(normalized)

    return result


def build_embedding_text(doc: MovieDocument) -> str:
    """
    임베딩 모델 입력용 구조화 텍스트를 생성한다.

    §11-6 단계 3: [제목] {title} [장르] {genres} [감독] {director}
                   [키워드] {keywords} [무드] {mood_tags} [줄거리] {overview[:200]}
    """
    parts = [
        f"[제목] {doc.title}",
        f"[장르] {', '.join(doc.genres)}",
    ]
    if doc.director:
        parts.append(f"[감독] {doc.director}")
    if doc.keywords:
        parts.append(f"[키워드] {', '.join(doc.keywords[:10])}")
    if doc.mood_tags:
        parts.append(f"[무드] {', '.join(doc.mood_tags)}")
    if doc.overview:
        parts.append(f"[줄거리] {doc.overview[:200]}")

    return " ".join(parts)


def get_fallback_mood_tags(genres: list[str]) -> list[str]:
    """
    장르 기반 기본 무드태그를 생성한다 (GPT 실패 시 fallback).

    §11-6-1: 장르 → 무드 기본 매핑 테이블에서 합집합, 중복 제거, 최대 5개.
    """
    mood_set: set[str] = set()
    for genre in genres:
        mood_set.update(GENRE_TO_DEFAULT_MOOD.get(genre, []))

    # 태그 0개인 경우 기본값 (§11-6)
    if not mood_set:
        return ["잔잔"]

    return list(mood_set)[:5]


# ============================================================
# Ollama (qwen2.5:14b) 무드태그 생성
# ============================================================

# §11-10 무드태그 생성 프롬프트
MOOD_TAG_PROMPT = """당신은 영화 분위기 분석 전문가입니다.
다음 영화 정보를 보고, 이 영화의 분위기를 나타내는 무드 태그를 3~5개 생성해주세요.

[영화 정보]
제목: {title}
장르: {genres}
키워드: {keywords}
줄거리: {overview}

[사용 가능한 무드 태그 (이 목록에서만 선택)]
몰입, 감동, 웅장, 긴장감, 힐링, 유쾌, 따뜻, 슬픔, 공포, 잔잔,
스릴, 카타르시스, 청춘, 우정, 가족애, 로맨틱, 미스터리, 반전,
철학적, 사회비판, 모험, 판타지, 레트로, 다크, 유머

JSON 배열로만 응답해주세요. 예: ["몰입", "감동", "웅장"]"""


async def generate_mood_tags(
    title: str,
    genres: list[str],
    keywords: list[str],
    overview: str,
) -> list[str]:
    """
    Ollama (qwen2.5:14b)로 무드태그를 생성한다.

    §11-6 단계 2: 장르+키워드+줄거리 → 3~5개 무드태그.
    §11-6 검증: 화이트리스트 필터링, 실패 시 1회 재시도 후 fallback.

    Ollama는 OpenAI 호환 API를 제공하므로 AsyncOpenAI 클라이언트를 사용한다.
    """
    from openai import AsyncOpenAI

    prompt = MOOD_TAG_PROMPT.format(
        title=title,
        genres=", ".join(genres),
        keywords=", ".join(keywords[:10]),
        overview=overview[:200],
    )

    # Ollama OpenAI 호환 API 사용
    client = AsyncOpenAI(
        base_url=f"{settings.OLLAMA_BASE_URL}/v1",
        api_key="ollama",  # Ollama는 API 키 불필요, 더미값
    )

    for attempt in range(2):  # 최대 2회 시도 (1회 재시도)
        try:
            response = await client.chat.completions.create(
                model=settings.MOOD_MODEL,  # qwen2.5:14b
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )
            content = response.choices[0].message.content or "[]"

            # JSON 배열 부분만 추출 (모델이 부가 텍스트를 출력할 수 있음)
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                content = content[start:end]

            tags = json.loads(content)

            # 화이트리스트 필터링 (§11-6)
            valid_tags = [tag for tag in tags if tag in MOOD_WHITELIST]

            if valid_tags:
                return valid_tags[:5]

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("mood_tag_generation_failed", title=title, attempt=attempt + 1, error=str(e))

    # 모든 시도 실패 시 fallback (§11-6)
    return get_fallback_mood_tags(genres)


# ============================================================
# 데이터 검증 (§11-3)
# ============================================================

CURRENT_YEAR = datetime.now().year


def validate_movie(doc: MovieDocument) -> bool:
    """
    영화 문서의 유효성을 검증한다.

    §11-3 데이터 검증 규칙:
    - 필수 필드: id, title, genres, release_year
    - 평점 범위: 0~10
    - 개봉년도: 1900~현재
    """
    # 필수 필드 존재
    if not doc.id or not doc.title or not doc.genres or not doc.release_year:
        return False

    # 평점 범위
    if doc.rating < 0 or doc.rating > 10:
        return False

    # 개봉년도 범위
    if doc.release_year < 1900 or doc.release_year > CURRENT_YEAR:
        return False

    return True


# ============================================================
# TMDBRawMovie → MovieDocument 변환
# ============================================================


async def process_raw_movie(raw: TMDBRawMovie, generate_mood: bool = True) -> MovieDocument | None:
    """
    TMDBRawMovie를 MovieDocument로 변환한다 (전체 전처리 파이프라인).

    1. 장르 한국어 변환
    2. 감독/배우 추출
    3. 키워드 추출
    4. OTT 정규화
    5. 무드태그 생성 (Ollama qwen2.5:14b)
    6. 임베딩 텍스트 구성
    7. 유효성 검증

    Args:
        raw: TMDB API 원본 데이터
        generate_mood: True이면 Ollama로 무드태그 생성, False이면 fallback 사용

    Returns:
        MovieDocument 또는 None (검증 실패 시)
    """
    # 개봉 연도 추출
    release_year = 0
    if raw.release_date and len(raw.release_date) >= 4:
        try:
            release_year = int(raw.release_date[:4])
        except ValueError:
            pass

    # 장르 변환
    genres = convert_genres(raw.genres)

    # 인물 추출
    director = extract_director(raw.credits)
    cast = extract_cast(raw.credits)

    # 키워드 추출
    keywords = extract_keywords(raw.keywords)

    # OTT 정규화
    ott_platforms = normalize_ott_platforms(raw.watch_providers)

    # MovieDocument 생성 (무드태그/임베딩텍스트 제외)
    doc = MovieDocument(
        id=str(raw.id),
        title=raw.title or raw.original_title,
        title_en=raw.original_title,
        overview=raw.overview,
        release_year=release_year,
        runtime=raw.runtime or 0,
        rating=raw.vote_average,
        popularity_score=raw.popularity,
        poster_path=raw.poster_path or "",
        genres=genres,
        keywords=keywords,
        director=director,
        cast=cast,
        ott_platforms=ott_platforms,
        source="tmdb",
    )

    # 무드태그 생성 (Ollama qwen2.5:14b 사용)
    if generate_mood and settings.OLLAMA_BASE_URL:
        doc.mood_tags = await generate_mood_tags(doc.title, genres, keywords, doc.overview)
    else:
        doc.mood_tags = get_fallback_mood_tags(genres)

    # 줄거리 빈 문자열 대체 (§11-3)
    if not doc.overview or len(doc.overview) < 10:
        doc.overview = f"{', '.join(genres)} 장르의 영화. 키워드: {', '.join(keywords[:5])}"

    # 임베딩 텍스트 구성
    doc.embedding_text = build_embedding_text(doc)

    # 유효성 검증 (§11-3)
    if not validate_movie(doc):
        logger.warning("movie_validation_failed", id=doc.id, title=doc.title)
        return None

    return doc
