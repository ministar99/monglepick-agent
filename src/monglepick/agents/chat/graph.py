"""
Chat Agent LangGraph StateGraph 구성 (§6-2).

11노드 + 2개 조건부 라우팅 함수로 구성된 Chat Agent 그래프.
SSE 스트리밍과 동기 실행 인터페이스를 제공한다.

그래프 흐름:
    START → context_loader → intent_classifier
      → route_after_intent (조건부 분기)
          │
          ├─ recommend/search → emotion_analyzer → preference_refiner
          │   → route_after_preference (조건부 분기)
          │       ├─ needs_clarification=True  → question_generator → response_formatter → END
          │       └─ needs_clarification=False → query_builder → rag_retriever
          │            → recommendation_ranker → explanation_generator → response_formatter → END
          │
          ├─ general → general_responder → response_formatter → END
          │
          └─ info/theater/booking → tool_executor_node → response_formatter → END

공개 인터페이스:
- run_chat_agent(user_id, session_id, message) → AsyncGenerator[str, None]  (SSE 스트리밍)
- run_chat_agent_sync(user_id, session_id, message) → ChatAgentState  (동기 테스트용)
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import structlog
from langgraph.graph import END, START, StateGraph

from monglepick.agents.chat.models import ChatAgentState
from monglepick.agents.chat.nodes import (
    context_loader,
    emotion_analyzer,
    error_handler,
    explanation_generator,
    general_responder,
    intent_classifier,
    preference_refiner,
    query_builder,
    question_generator,
    rag_retriever,
    recommendation_ranker,
    response_formatter,
    tool_executor_node,
)

logger = structlog.get_logger()


# ============================================================
# 라우팅 함수
# ============================================================

def route_after_intent(state: ChatAgentState) -> str:
    """
    intent_classifier 이후 분기 결정.

    - recommend/search → emotion_analyzer (추천 흐름)
    - general → general_responder (일반 대화)
    - info/theater/booking → tool_executor_node (도구 실행)
    - unknown/None → error_handler (에러 처리)

    Args:
        state: ChatAgentState (intent 필요)

    Returns:
        다음 노드 이름 문자열
    """
    intent = state.get("intent")

    if intent is None:
        return "error_handler"

    intent_type = intent.intent

    if intent_type in ("recommend", "search"):
        return "emotion_analyzer"
    elif intent_type == "general":
        return "general_responder"
    elif intent_type in ("info", "theater", "booking"):
        return "tool_executor_node"
    else:
        return "error_handler"


def route_after_preference(state: ChatAgentState) -> str:
    """
    preference_refiner 이후 분기 결정.

    - needs_clarification=True → question_generator (후속 질문)
    - needs_clarification=False → query_builder (추천 진행)

    Args:
        state: ChatAgentState (needs_clarification 필요)

    Returns:
        다음 노드 이름 문자열
    """
    needs_clarification = state.get("needs_clarification", True)

    if needs_clarification:
        return "question_generator"
    else:
        return "query_builder"


# ============================================================
# 그래프 빌드
# ============================================================

def build_chat_graph() -> StateGraph:
    """
    Chat Agent StateGraph를 구성하고 컴파일한다.

    11개 노드와 2개 조건부 분기를 등록하여 영화 추천 대화 흐름을 정의한다.

    Returns:
        컴파일된 StateGraph (CompiledGraph)
    """
    # StateGraph 생성 (ChatAgentState TypedDict 기반)
    graph = StateGraph(ChatAgentState)

    # ── 노드 등록 (13개) ──
    graph.add_node("context_loader", context_loader)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("emotion_analyzer", emotion_analyzer)
    graph.add_node("preference_refiner", preference_refiner)
    graph.add_node("question_generator", question_generator)
    graph.add_node("query_builder", query_builder)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("recommendation_ranker", recommendation_ranker)
    graph.add_node("explanation_generator", explanation_generator)
    graph.add_node("response_formatter", response_formatter)
    graph.add_node("error_handler", error_handler)
    graph.add_node("general_responder", general_responder)
    graph.add_node("tool_executor_node", tool_executor_node)

    # ── 엣지 정의 ──

    # START → context_loader → intent_classifier
    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "intent_classifier")

    # intent_classifier → 조건부 분기 (route_after_intent)
    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "emotion_analyzer": "emotion_analyzer",
            "general_responder": "general_responder",
            "tool_executor_node": "tool_executor_node",
            "error_handler": "error_handler",
        },
    )

    # 추천 흐름: emotion_analyzer → preference_refiner → 조건부 분기
    graph.add_edge("emotion_analyzer", "preference_refiner")

    graph.add_conditional_edges(
        "preference_refiner",
        route_after_preference,
        {
            "question_generator": "question_generator",
            "query_builder": "query_builder",
        },
    )

    # 후속 질문 흐름: question_generator → response_formatter → END
    graph.add_edge("question_generator", "response_formatter")

    # 추천 진행 흐름: query_builder → rag_retriever → recommendation_ranker → explanation_generator → response_formatter
    graph.add_edge("query_builder", "rag_retriever")
    graph.add_edge("rag_retriever", "recommendation_ranker")
    graph.add_edge("recommendation_ranker", "explanation_generator")
    graph.add_edge("explanation_generator", "response_formatter")

    # 일반 대화: general_responder → response_formatter
    graph.add_edge("general_responder", "response_formatter")

    # 도구 실행: tool_executor_node → response_formatter
    graph.add_edge("tool_executor_node", "response_formatter")

    # 에러 처리: error_handler → response_formatter
    graph.add_edge("error_handler", "response_formatter")

    # response_formatter → END
    graph.add_edge("response_formatter", END)

    # 그래프 컴파일
    compiled = graph.compile()
    logger.info("chat_graph_compiled", node_count=13)
    return compiled


# ── 모듈 레벨 싱글턴: 컴파일 1회 ──
chat_graph = build_chat_graph()


# ============================================================
# 노드 이름 → 한국어 상태 메시지 매핑 (SSE status 이벤트용)
# ============================================================

NODE_STATUS_MESSAGES: dict[str, str] = {
    "context_loader": "사용자 정보를 불러오고 있어요...",
    "intent_classifier": "말씀을 이해하고 있어요...",
    "emotion_analyzer": "감정을 분석하고 있어요...",
    "preference_refiner": "취향을 파악하고 있어요...",
    "question_generator": "질문을 준비하고 있어요...",
    "query_builder": "검색 조건을 구성하고 있어요...",
    "rag_retriever": "영화를 검색하고 있어요... 🔍",
    "recommendation_ranker": "최적의 영화를 고르고 있어요...",
    "explanation_generator": "추천 이유를 작성하고 있어요...",
    "response_formatter": "응답을 정리하고 있어요...",
    "error_handler": "문제를 처리하고 있어요...",
    "general_responder": "답변을 준비하고 있어요...",
    "tool_executor_node": "기능을 확인하고 있어요...",
}


# ============================================================
# SSE 스트리밍 인터페이스
# ============================================================

async def run_chat_agent(
    user_id: str,
    session_id: str,
    message: str,
) -> AsyncGenerator[str, None]:
    """
    Chat Agent를 SSE 스트리밍 모드로 실행한다.

    LangGraph astream(stream_mode="updates")로 각 노드 완료 시 이벤트를 발행한다.

    SSE 이벤트 형식:
    - {"event": "status", "data": {"phase": "노드명", "message": "한국어 상태"}}
    - {"event": "movie_card", "data": {RankedMovie JSON}}
    - {"event": "token", "data": {"delta": "응답 텍스트"}}
    - {"event": "done", "data": {}}
    - {"event": "error", "data": {"message": "에러 메시지"}}

    Args:
        user_id: 사용자 ID (빈 문자열이면 익명)
        session_id: 세션 ID
        message: 사용자 입력 메시지

    Yields:
        SSE 이벤트 JSON 문자열 (줄바꿈 포함)
    """
    # 초기 State 구성
    initial_state: ChatAgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "current_input": message,
        "messages": [],
    }

    try:
        # LangGraph astream: 각 노드 완료 시 업데이트 수신
        async for event in chat_graph.astream(
            initial_state,
            stream_mode="updates",
        ):
            # event 형식: {"node_name": {updates_dict}}
            for node_name, updates in event.items():
                # status 이벤트 발행
                status_msg = NODE_STATUS_MESSAGES.get(node_name, f"{node_name} 처리 중...")
                yield _format_sse_event("status", {
                    "phase": node_name,
                    "message": status_msg,
                })

                # response_formatter 완료 시 결과 발행
                if node_name == "response_formatter":
                    # movie_card 이벤트 (추천 영화가 있는 경우)
                    ranked_movies = updates.get("ranked_movies") or []
                    if not ranked_movies:
                        # state에서 ranked_movies 확인 (response_formatter가 전달하지 않을 수 있음)
                        pass

                    # token 이벤트: 응답 텍스트 발행 (MVP: 전체 텍스트를 단일 이벤트로)
                    response_text = updates.get("response", "")
                    if response_text:
                        yield _format_sse_event("token", {"delta": response_text})

                # recommendation_ranker 완료 시 movie_card 이벤트 발행
                if node_name == "recommendation_ranker":
                    ranked_movies = updates.get("ranked_movies", [])
                    for movie in ranked_movies:
                        movie_data = movie.model_dump() if hasattr(movie, "model_dump") else movie
                        yield _format_sse_event("movie_card", movie_data)

        # 완료 이벤트
        yield _format_sse_event("done", {})

    except Exception as e:
        logger.error("chat_agent_stream_error", error=str(e))
        yield _format_sse_event("error", {"message": str(e)})
        yield _format_sse_event("done", {})


def _format_sse_event(event_type: str, data: dict) -> str:
    """
    SSE 이벤트를 JSON 문자열로 포맷한다.

    Args:
        event_type: 이벤트 타입 (status, movie_card, token, done, error)
        data: 이벤트 데이터 dict

    Returns:
        SSE 포맷 문자열 ("event: {type}\ndata: {json}\n\n")
    """
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ============================================================
# 동기 실행 인터페이스 (테스트용)
# ============================================================

async def run_chat_agent_sync(
    user_id: str,
    session_id: str,
    message: str,
) -> ChatAgentState:
    """
    Chat Agent를 동기 모드로 실행하여 최종 State를 반환한다 (테스트/디버그용).

    Args:
        user_id: 사용자 ID
        session_id: 세션 ID
        message: 사용자 입력 메시지

    Returns:
        실행 완료된 ChatAgentState
    """
    initial_state: ChatAgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "current_input": message,
        "messages": [],
    }

    result = await chat_graph.ainvoke(initial_state)
    return result
