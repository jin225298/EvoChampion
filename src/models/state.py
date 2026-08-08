"""
LangGraph state definition for the EvoChampion system.
"""

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from src.models.messages import RoutedMessage


def _rollout_runs_reducer(old: list, new: list | None) -> list:
    if new is None:
        return []
    return old + new


class EvoState(TypedDict, total=False):
    user_goal: str
    trace_id: str
    round_id: int

    pending_message: RoutedMessage

    message_log: Annotated[list[str], operator.add]

    route_decision: str

    search_results: list[dict[str, Any]]
    candidate_dataset_refs: list[dict[str, Any]]
    materialized_dataset_questions: list[dict[str, Any]]
    candidate_questions: list[dict[str, Any]]
    filtered_questions: list[dict[str, Any]]
    train_questions: list[dict[str, Any]]
    lf_val_questions: list[dict[str, Any]]
    cotest_questions: list[dict[str, Any]]
    test_questions: list[dict[str, Any]]
    probe_questions: list[dict[str, Any]]
    mastered_questions: list[dict[str, Any]]
    classified_questions: list[dict[str, Any]]

    rollout_runs: Annotated[list, _rollout_runs_reducer]

    train_path: str
    lf_val_path: str
    cotest_path: str
    test_path: str
    probe_path: str
    dataset_info_path: str
    dataset_dir: str
    train_dataset_name: str
    lf_val_dataset_name: str

    champion_model_path: str
    candidate_model_path: str

    current_search_node_id: str
    search_dag_nodes: list[dict[str, Any]]
    search_dag_edges: list[dict[str, Any]]
    replay_buffer_entries: list[dict[str, Any]]
    replay_buffer_used_count: int
    replay_sample_ratio_override: float
    rollout_difficulty_distribution: dict[str, int]
    rollout_hard_ratio: float
    rollout_pass_count_histogram: dict[str, int]
    difficulty_threshold_policy: dict[str, Any]
    round_data_stats: dict[str, Any]

    global_probe_set_path: str
    probe_frozen_set_path: str
    frozen_probe_eval_method: str
    external_probe_path: str
    holdout_eval_path: str
    holdout_eval_questions: list[dict[str, Any]]
    mastered_memory_set_path: str
    heldout_registry_path: str
    round_heldout_questions: list[dict[str, Any]]
    round_retired_holdout_questions: list[dict[str, Any]]
    round_probe_pool_questions: list[dict[str, Any]]
    pending_test_buffer_questions: list[dict[str, Any]]
    pending_probe_pool_questions: list[dict[str, Any]]
    probe_pool_intake_questions: list[dict[str, Any]]
    reserved_dataset_question_ids: dict[str, list[str]]
    last_dataset_bundle: dict[str, Any]
    last_dataset_bundle_state: dict[str, Any]
    last_dataset_bundle_round_id: int
    last_attempt_question_ids: list[str]
    last_attempt_reserved_dataset_question_ids: dict[str, list[str]]

    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    old_skill_eval_count: int
    champion_holdout_baseline: float
    champion_frozen_probe_error: float
    base_frozen_error_rate: float
    should_stop: bool
    should_promote_candidate: bool
    budget_exhausted: bool
    termination_reason: str
    checkpoint_next_round_id: int
    last_inspection_decision: str
    kept_branch_decision: str
    kept_branch_node_id: str
    rollback_streak: int
    dataset_reviewed_ids: list[str]
    dataset_review_verdicts: list[dict[str, Any]]
    dataset_review_pending_refs: list[dict[str, Any]]
    dataset_review_job_id: str
    dataset_review_active: bool
    dataset_review_completed: bool
    dataset_review_drained_count: int
    candidate_action_plan: dict[str, Any]
    current_training_hyperparams: dict[str, Any]
    current_action_metadata: dict[str, Any]
    sampling_plan: dict[str, Any]
    difficulty_teacher_feedback: dict[str, Any]
    difficulty_retry_count: int
    diagnostic_mode: str
    probe_diagnostic_path: str
    probe_diagnostic_focus_path: str
    probe_focus_used_count: int
    agent_prompt_pack: dict[str, Any]
    agent_prompt_pack_path: str

    # Dynamic difficulty target from teacher, consumed by filter for weak-area oversampling.
    target_bucket: str

    # Pool + cursor: accumulate search results across rounds, consume gradually
    dataset_pool: list[dict[str, Any]]
    pool_cursor: int
    consumed_dataset_ids: list[str]
    search_generation: int
    last_search_feedback: dict[str, Any]
    dataset_schema_info: dict[str, Any]
    previous_dataset_refs: list[dict[str, Any]]
    dataset_states_path: str
    active_dataset_id: str
    cross_dataset_pool: dict[str, list[dict[str, Any]]]
    current_window_offset: int
    current_window_size: int
    data_window_offset: int
    data_window_size: int
    windows_loaded_this_round: int
    profile_items_loaded_this_round: int
    max_windows_per_round: int
    max_profile_items_per_round: int
    quota_met: bool
    quota_shortfall: dict[str, int]
    quota_accumulated_questions: list[dict[str, Any]]
    cached_rollout_scored_questions: list[dict[str, Any]]
    cached_replenishment_windows_seen: dict[str, list[int]]
    cached_replenishment_windows_seen_round_id: int
    replenishment_loaded_windows: dict[str, list[int]]
    replenishment_loaded_windows_round_id: int
    replenishment_attempted_windows: dict[str, list[int]]
    replenishment_attempted_windows_round_id: int
    data_replenishment_needed: bool
    data_replenishment_exhausted: bool
    replenishment_cycle_active: bool
    next_dataset_ref: dict[str, Any]
    screening_load_failures: list[dict[str, Any]]
    resume_training: bool
