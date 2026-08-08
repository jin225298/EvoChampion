"""Lazy exports for LangGraph node functions.

Importing every node eagerly pulls optional runtime dependencies (LangGraph,
loguru, torch/vLLM) into lightweight unit tests. Keep package import cheap and
load node modules only when their symbols are requested.
"""

from importlib import import_module

_EXPORTS = {
    "bootstrap_node": ("src.nodes.bootstrap", "bootstrap_node"),
    "prompt_designer_node": ("src.nodes.prompt_designer", "prompt_designer_node"),
    "router_node": ("src.nodes.router", "router_node"),
    "format_error_node": ("src.nodes.format_error", "format_error_node"),
    "teacher_node": ("src.nodes.teacher", "teacher_node"),
    "searcher_node": ("src.nodes.searcher", "searcher_node"),
    "filter_pre_rollout_node": ("src.nodes.filter", "filter_pre_rollout_node"),
    "filter_post_rollout_node": ("src.nodes.filter", "filter_post_rollout_node"),
    "classifier_node": ("src.nodes.classifier", "classifier_node"),
    "hf_search_tool_node": ("src.nodes.hf_search_tool", "hf_search_tool_node"),
    "screening_entry_node": ("src.nodes.screening_entry", "screening_entry_node"),
    "dataset_schema_agent_node": ("src.nodes.dataset_schema_agent", "dataset_schema_agent_node"),
    "dataset_reviewer_node": ("src.nodes.dataset_reviewer", "dataset_reviewer_node"),
    "rollout_dispatcher_node": ("src.nodes.rollout_dispatcher", "rollout_dispatcher_node"),
    "rollout_worker_node": ("src.nodes.rollout_worker", "rollout_worker_node"),
    "rollout_aggregator_node": ("src.nodes.rollout_aggregator", "rollout_aggregator_node"),
    "data_builder_node": ("src.nodes.data_builder", "data_builder_node"),
    "parameter_master_node": ("src.nodes.parameter_master", "parameter_master_node"),
    "trainer_node": ("src.nodes.trainer", "trainer_node"),
    "evaluator_node": ("src.nodes.evaluator", "evaluator_node"),
    "strategy_inspector_node": ("src.nodes.strategy_inspector", "strategy_inspector_node"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    return getattr(import_module(module_name), attr_name)
