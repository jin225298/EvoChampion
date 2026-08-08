from typing import Any

from src.models.messages import (
    AgentName,
    MessageHeader,
    MessageType,
    RoutedMessage,
    SearchRequestPayload,
)
from src.models.state import EvoState
from src.tools.agent_prompts import SEARCH_EXPERT_PROMPT
from src.tools.llm_decision import decide_json, prompt_for_agent
from src.tools.search_query import normalize_content_search_query

_ALLOWED_SOURCES = {"huggingface", "web", "github"}


def searcher_node(state: EvoState) -> dict:
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    prompt_state: dict[str, Any] = dict(state)
    request = SearchRequestPayload.model_validate(pending_message.payload)
    teacher_query = request.search_query
    previous_feedback = state.get("last_search_feedback") or {}
    decision = decide_json(
        agent_name="search_expert",
        prompt=prompt_for_agent(prompt_state, "search_expert", SEARCH_EXPERT_PROMPT),
        context={
            **request.model_dump(mode="json"),
            "previous_search_feedback": previous_feedback,
        },
        fallback={
            "search_query": request.search_query,
            "search_sources": request.search_sources,
            "reason": "pass through teacher request",
        },
        trace_id=trace_id,
        round_id=round_id,
        disable_thinking=True,
    )
    sources = [
        str(item)
        for item in decision.get("search_sources", request.search_sources)
        if str(item) in _ALLOWED_SOURCES
    ] or request.search_sources
    normalized_query = normalize_content_search_query(
        decision.get("search_query") or request.search_query,
        goal=request.goal,
        fallback=request.search_query,
    )
    request = SearchRequestPayload.model_validate({
        **request.model_dump(mode="json"),
        "search_query": normalized_query,
        "search_sources": sources,
    })
    print(
        f"[searcher] Round {round_id}: "
        f"teacher_query='{teacher_query}' -> "
        f"search_query='{request.search_query}' "
        f"(prev_new={previous_feedback.get('new_result_count', '?')} "
        f"cache_hit={previous_feedback.get('cache_hit_count', '?')})"
    )

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.SEARCHER,
            receiver=AgentName.HF_SEARCH_TOOL,
            message_type=MessageType.SEARCH_REQUEST,
        ),
        payload=request,
    )

    return {"pending_message": msg}
