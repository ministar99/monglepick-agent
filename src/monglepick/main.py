"""
몽글픽 AI Agent FastAPI 앱.

Phase 3: lifespan 추가 (5개 DB 클라이언트 초기화/종료), chat_router 등록.
LangSmith: LANGCHAIN_API_KEY 설정 시 LLM 호출/그래프 실행 자동 트레이싱.
"""

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from monglepick.api.chat import chat_router
from monglepick.api.router import api_router
from monglepick.config import settings
from monglepick.db.clients import close_all_clients, init_all_clients

logger = structlog.get_logger()

# ── LangSmith 트레이싱 자동 활성화 ──
# LANGCHAIN_API_KEY가 설정되어 있으면 LangChain/LangGraph의 모든 LLM 호출과
# 그래프 노드 실행을 LangSmith 대시보드에 자동 추적한다.
# 환경변수 방식이므로 코드 내 콜백 등록 없이 자동 동작한다.
if settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    logger.info(
        "langsmith_tracing_enabled",
        project=settings.LANGCHAIN_PROJECT,
        endpoint=settings.LANGCHAIN_ENDPOINT,
    )

# 앱 버전
APP_VERSION = "0.2.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 라이프사이클 관리.

    startup: 5개 DB 클라이언트 초기화 (Qdrant/Neo4j/Redis/ES/MySQL)
    shutdown: 모든 DB 연결 정리
    """
    # ── Startup ──
    logger.info("app_startup", version=APP_VERSION)
    try:
        await init_all_clients()
        logger.info("app_startup_complete", version=APP_VERSION)
    except Exception as e:
        # DB 연결 실패 시에도 앱은 기동 (health 엔드포인트는 동작)
        logger.error("app_startup_db_error", error=str(e))

    yield

    # ── Shutdown ──
    logger.info("app_shutdown")
    await close_all_clients()
    logger.info("app_shutdown_complete")


app = FastAPI(
    title="몽글픽 AI Agent",
    description="영화 추천 AI 에이전트 API",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트."""
    return {
        "status": "ok",
        "version": APP_VERSION,
        "settings": {
            "qdrant_url": settings.QDRANT_URL,
            "redis_url": settings.REDIS_URL,
            "elasticsearch_url": settings.ELASTICSEARCH_URL,
            "neo4j_uri": settings.NEO4J_URI,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
    }
