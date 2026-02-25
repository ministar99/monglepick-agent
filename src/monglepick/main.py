from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from monglepick.api.router import api_router
from monglepick.config import settings

app = FastAPI(
    title="몽글픽 AI Agent",
    description="영화 추천 AI 에이전트 API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "settings": {
            "qdrant_url": settings.QDRANT_URL,
            "redis_url": settings.REDIS_URL,
            "elasticsearch_url": settings.ELASTICSEARCH_URL,
            "neo4j_uri": settings.NEO4J_URI,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
    }
