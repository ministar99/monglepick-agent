"""
Tool Executor 체인 스텁 (Phase 6 예정).

§6-2 Node 11의 ReAct 도구 실행 체인 플레이스홀더.
Phase 6에서 LangChain Tools (get_movie_details, search_theaters,
get_booking_link, get_similar_movies, analyze_poster)를
구현할 때 이 체인을 완성한다.

현재는 NotImplementedError를 발생시킨다.
"""

from __future__ import annotations


async def execute_tool(intent: str, current_input: str) -> dict:
    """
    사용자 의도에 맞는 도구를 실행한다 (Phase 6 예정).

    Args:
        intent: 사용자 의도 (info, theater, booking, search)
        current_input: 사용자 입력 텍스트

    Returns:
        도구 실행 결과 dict

    Raises:
        NotImplementedError: Phase 6에서 구현 예정
    """
    raise NotImplementedError(
        "Tool executor is planned for Phase 6. "
        f"Received intent='{intent}', input='{current_input}'"
    )
