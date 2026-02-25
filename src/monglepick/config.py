"""
프로젝트 설정 (pydantic-settings 기반).

§17 Phase 0: 27개 환경 변수 + Ollama 로컬 LLM 설정 추가.
.env 파일에서 환경 변수를 로드한다.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── API Keys ──
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    UPSTAGE_API_KEY: str = ""
    TMDB_API_KEY: str = ""
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    KOBIS_API_KEY: str = ""
    KAKAO_API_KEY: str = ""

    # ── Ollama (로컬 LLM 서버) ──
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ── Qdrant ──
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "movies"

    # ── Redis ──
    REDIS_URL: str = "redis://localhost:6379"

    # ── Elasticsearch ──
    ELASTICSEARCH_URL: str = "http://localhost:9200"

    # ── Neo4j ──
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "monglepick_dev"

    # ── MySQL ──
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = "monglepick"
    MYSQL_USER: str = "monglepick"
    MYSQL_PASSWORD: str = "monglepick_dev"

    # ── Embedding (Upstage Solar) ──
    EMBEDDING_MODEL: str = "Upstage/solar-embedding-1-large"
    EMBEDDING_DIMENSION: int = 4096

    # ── LLM Models (Ollama 로컬 모델) ──
    # 구조화 출력 (JSON): Qwen2.5 14B
    INTENT_MODEL: str = "qwen2.5:14b"
    EMOTION_MODEL: str = "qwen2.5:14b"
    MOOD_MODEL: str = "qwen2.5:14b"
    # 한국어 자연어 생성: EXAONE 4.0 32B (비추론 모드: temperature < 0.6)
    PREFERENCE_MODEL: str = "exaone-32b:latest"
    CONVERSATION_MODEL: str = "exaone-32b:latest"
    # 경량 한국어 생성: EXAONE 4.0 32B (1.2B 미다운로드, 32B로 대체)
    QUESTION_MODEL: str = "exaone-32b:latest"
    # Vision (포스터 분석): Qwen2.5 VL 32B
    VISION_MODEL: str = "qwen2.5vl:32b"

    # ── Session / Conversation ──
    SESSION_TTL_DAYS: int = 30
    MAX_CONVERSATION_TURNS: int = 20

    # ── Security ──
    SERVICE_API_KEY: str = ""
    DAILY_TOKEN_LIMIT: int = 1_000_000


settings = Settings()
