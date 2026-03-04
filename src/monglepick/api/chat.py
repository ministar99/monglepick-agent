"""
Chat Agent SSE/동기 엔드포인트 (§6 Phase 3).

2개 엔드포인트:
- POST /api/v1/chat       — SSE 스트리밍 (EventSourceResponse)
- POST /api/v1/chat/sync  — 동기 JSON (디버그/테스트용)

요청 모델: ChatRequest (user_id, session_id, message)
응답 모델: ChatSyncResponse (동기 전용)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from monglepick.agents.chat.graph import run_chat_agent, run_chat_agent_sync

logger = structlog.get_logger()

# APIRouter 생성 (prefix는 main.py에서 설정)
chat_router = APIRouter(tags=["chat"])


# ============================================================
# 요청/응답 모델
# ============================================================

class ChatRequest(BaseModel):
    """
    채팅 요청 모델.

    user_id: 사용자 ID (빈 문자열이면 익명)
    session_id: 세션 ID (빈 문자열이면 신규 세션)
    message: 사용자 입력 메시지 (1~2000자, 필수)
    """

    user_id: str = Field(
        default="",
        description="사용자 ID (빈 문자열이면 익명)",
    )
    session_id: str = Field(
        default="",
        description="세션 ID (빈 문자열이면 신규 세션)",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="사용자 입력 메시지 (1~2000자)",
    )


class ChatSyncResponse(BaseModel):
    """
    동기 채팅 응답 모델 (디버그/테스트용).

    response: 최종 응답 텍스트
    intent: 분류된 의도
    emotion: 감지된 감정 (None이면 미감지)
    movie_count: 추천된 영화 수
    """

    response: str = Field(default="", description="최종 응답 텍스트")
    intent: str = Field(default="", description="분류된 의도")
    emotion: str | None = Field(default=None, description="감지된 감정")
    movie_count: int = Field(default=0, description="추천된 영화 수")


# ============================================================
# SSE 스트리밍 엔드포인트
# ============================================================

@chat_router.post("/chat")
async def chat_sse(request: ChatRequest):
    """
    SSE 스트리밍 채팅 엔드포인트.

    Chat Agent 그래프를 실행하며, 각 노드 완료 시 SSE 이벤트를 발행한다.
    Content-Type: text/event-stream

    SSE 이벤트:
    - status: 현재 처리 단계 (phase, message)
    - movie_card: 추천 영화 데이터 (RankedMovie JSON)
    - token: 응답 텍스트 (delta)
    - done: 완료 신호
    - error: 에러 메시지

    Args:
        request: ChatRequest (user_id, session_id, message)

    Returns:
        EventSourceResponse (SSE 스트리밍)
    """
    logger.info(
        "chat_sse_request",
        user_id=request.user_id or "(anonymous)",
        message_preview=request.message[:50],
    )

    async def event_generator():
        """SSE 이벤트 생성기 — run_chat_agent()의 이벤트를 relay한다."""
        async for sse_event in run_chat_agent(
            user_id=request.user_id,
            session_id=request.session_id,
            message=request.message,
        ):
            yield sse_event

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ============================================================
# 동기 엔드포인트 (디버그/테스트용)
# ============================================================

@chat_router.post("/chat/sync", response_model=ChatSyncResponse)
async def chat_sync(request: ChatRequest):
    """
    동기 JSON 채팅 엔드포인트 (디버그/테스트용).

    Chat Agent 그래프를 동기 실행하고, 최종 State에서 주요 정보를 추출하여 JSON으로 반환한다.

    Args:
        request: ChatRequest (user_id, session_id, message)

    Returns:
        ChatSyncResponse (response, intent, emotion, movie_count)
    """
    logger.info(
        "chat_sync_request",
        user_id=request.user_id or "(anonymous)",
        message_preview=request.message[:50],
    )

    state = await run_chat_agent_sync(
        user_id=request.user_id,
        session_id=request.session_id,
        message=request.message,
    )

    # State에서 응답 정보 추출
    intent = state.get("intent")
    emotion = state.get("emotion")
    ranked = state.get("ranked_movies", [])

    return ChatSyncResponse(
        response=state.get("response", ""),
        intent=intent.intent if intent else "",
        emotion=emotion.emotion if emotion else None,
        movie_count=len(ranked),
    )
