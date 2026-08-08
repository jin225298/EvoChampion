import importlib

from config.settings import ROLLOUT_MAX_CONCURRENT

from src.models.messages import MessageType
from src.models.state import EvoState


def route_after_router(state: EvoState) -> str:
    receiver = state.get("route_decision", "system")
    if receiver == "filter":
        pending_message = state.get("pending_message")
        message_type = pending_message.header.message_type if pending_message is not None else MessageType.FORMAT_ERROR
        # P0 route guard: strict filter message routing
        if message_type == MessageType.ROLLOUT_RESULT:
            return "filter_post_rollout"
        if message_type == MessageType.MATERIALIZED_DATASET:
            return "filter_pre_rollout"
        if message_type == MessageType.ROLLOUT_REQUEST:
            return "rollout_dispatcher"
        # Unknown filter-bound message → format_error
        return "format_error"
    return receiver


def fan_out_to_workers(state: EvoState) -> list[object]:
    Send = importlib.import_module("langgraph.types").Send

    questions = list(state.get("candidate_questions", []) or [])
    if not questions:
        return []
    max_concurrent = max(1, int(ROLLOUT_MAX_CONCURRENT or 1))
    worker_count = min(len(questions), max_concurrent)
    base_count = len(questions) // worker_count
    remainder = len(questions) % worker_count

    sends = []
    question_offset = 0
    for worker_idx in range(worker_count):
        shard_count = base_count + (1 if worker_idx < remainder else 0)
        shard = questions[question_offset:question_offset + shard_count]
        sends.append(Send("rollout_worker", {
            "candidate_questions": shard,
            "champion_model_path": state.get("champion_model_path", ""),
            "trace_id": state.get("trace_id", ""),
            "round_id": state.get("round_id", 0),
            "rollout_repeat_count": 1,
            "rollout_worker_idx": worker_idx,
            "rollout_start_idx": question_offset,
            "rollout_config_hash": state.get("rollout_config_hash", ""),
            "rollout_judge_version": state.get("rollout_judge_version", ""),
        }))
        question_offset += shard_count

    return sends

def route_after_aggregator(state: EvoState) -> str:
    _ = state
    return "router"


def route_after_filter_post_rollout(state: EvoState) -> str:
    if state.get("data_replenishment_needed"):
        if state.get("dataset_review_active") and state.get("dataset_review_pending_refs"):
            return "dataset_reviewer"
        return "screening_entry"
    if state.get("data_replenishment_exhausted"):
        return "router"
    return "router"


def build_graph():
    graph_module = importlib.import_module("langgraph.graph")
    END = graph_module.END
    START = graph_module.START
    StateGraph = graph_module.StateGraph
    from src.nodes import (
        bootstrap_node,
        classifier_node,
        data_builder_node,
        dataset_schema_agent_node,
        dataset_reviewer_node,
        evaluator_node,
        filter_post_rollout_node,
        filter_pre_rollout_node,
        format_error_node,
        hf_search_tool_node,
        parameter_master_node,
        prompt_designer_node,
        rollout_aggregator_node,
        rollout_dispatcher_node,
        rollout_worker_node,
        router_node,
        searcher_node,
        screening_entry_node,
        strategy_inspector_node,
        teacher_node,
        trainer_node,
    )

    builder = StateGraph(EvoState)

    builder.add_node("bootstrap", bootstrap_node)
    builder.add_node("prompt_designer", prompt_designer_node)
    builder.add_node("router", router_node)
    builder.add_node("teacher", teacher_node)
    builder.add_node("searcher", searcher_node)
    builder.add_node("hf_search_tool", hf_search_tool_node)
    builder.add_node("dataset_schema_agent", dataset_schema_agent_node)
    builder.add_node("dataset_reviewer", dataset_reviewer_node)
    builder.add_node("screening_entry", screening_entry_node)
    builder.add_node("filter_pre_rollout", filter_pre_rollout_node)
    builder.add_node("filter_post_rollout", filter_post_rollout_node)
    builder.add_node("rollout_dispatcher", rollout_dispatcher_node)
    builder.add_node("rollout_worker", rollout_worker_node)
    builder.add_node("rollout_aggregator", rollout_aggregator_node)
    builder.add_node("classifier", classifier_node)
    builder.add_node("data_builder", data_builder_node)
    builder.add_node("parameter_master", parameter_master_node)
    builder.add_node("trainer", trainer_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("strategy_inspector", strategy_inspector_node)
    builder.add_node("format_error", format_error_node)

    builder.add_edge(START, "bootstrap")
    builder.add_edge("bootstrap", "prompt_designer")
    builder.add_edge("prompt_designer", "router")

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "teacher": "teacher",
            "searcher": "searcher",
            "hf_search_tool": "hf_search_tool",
            "dataset_schema_agent": "dataset_schema_agent",
            "dataset_reviewer": "dataset_reviewer",
            "screening_entry": "screening_entry",
            "rollout_dispatcher": "rollout_dispatcher",
            "filter_pre_rollout": "filter_pre_rollout",
            "filter_post_rollout": "filter_post_rollout",
            "classifier": "classifier",
            "data_builder": "data_builder",
            "parameter_master": "parameter_master",
            "trainer": "trainer",
            "evaluator": "evaluator",
            "strategy_inspector": "strategy_inspector",
            "format_error": "format_error",
            "system": END,
        },
    )

    builder.add_edge("teacher", "router")
    builder.add_edge("searcher", "router")
    builder.add_edge("hf_search_tool", "dataset_reviewer")
    builder.add_edge("dataset_schema_agent", "router")
    builder.add_edge("dataset_reviewer", "router")
    builder.add_edge("screening_entry", "router")
    builder.add_edge("filter_pre_rollout", "router")
    builder.add_conditional_edges(
        "filter_post_rollout",
        route_after_filter_post_rollout,
        {
            "dataset_reviewer": "dataset_reviewer",
            "screening_entry": "screening_entry",
            "router": "router",
        },
    )
    builder.add_edge("classifier", "router")
    builder.add_edge("data_builder", "router")
    builder.add_edge("parameter_master", "router")
    builder.add_edge("trainer", "router")
    builder.add_edge("evaluator", "router")
    builder.add_edge("strategy_inspector", "router")

    builder.add_conditional_edges(#并行来做前置推理确保题目的质量，推理结果由 rollout_aggregator 来汇总后发回 router 决定后续走 filter_post_rollout 还是 filter_pre_rollout
        "rollout_dispatcher",
        fan_out_to_workers,
        ["rollout_worker"],
    )
    builder.add_edge("rollout_worker", "rollout_aggregator")
    builder.add_conditional_edges(
        "rollout_aggregator",
        route_after_aggregator,
        {"router": "router"},
    )

    builder.add_edge("format_error", END)

    return builder


def compile_graph():
    builder = build_graph()
    return builder.compile()
