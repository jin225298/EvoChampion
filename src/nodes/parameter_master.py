from typing import Any
from pathlib import Path

from src.models.messages import (
    AgentName,
    DatasetBundlePayload,
    DiagnosticRequestPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
    SearchRequestPayload,
)
from src.models.state import EvoState
from config.settings import (
    DATASET_SHARD_SIZE,
    MCTS_MUTATION_SCALE_MAX,
    MCTS_MUTATION_SCALE_MIN,
    MCTS_TUNER_COLD_START_EDGES,
    get_session_dir,
)
from src.tools.agent_prompts import (
    PARAMETER_MASTER_LEAN_PROMPT,
    PARAMETER_MASTER_MUTATION_PROMPT,
    PARAMETER_MASTER_RETRY_DATA_PROMPT,
    REPLAY_TEACHER_RATIO_PROMPT,
    TRAINING_HYPERPARAMS_PROMPT,
)
from src.tools.llm_decision import decide_json, decide_json_leaf, prompt_for_agent
from src.tools.strategy_policy import (
    build_parameter_edge_summary,
    build_parameter_master_card,
    build_curriculum_sampling_plan,
    decide_parameter_master_action,
    get_parameter_action_space,
)

_TRAINING_HYPERPARAM_KEYS = {
    "per_device_train_batch_size",
    "learning_rate",
    "num_train_epochs",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "warmup_steps",
    "warmup_mode",
    "warmup_value",
    "lr_scheduler_type",
    "finetuning_type",
}

_ALLOWED_MUTATION_SCALES = {0.9, 1.0, 1.1}


def _clean_reserved_map(raw: object) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[str]] = {}
    for dataset_id, question_ids in raw.items():
        if not isinstance(question_ids, list):
            continue
        clean_ids = [str(qid) for qid in question_ids if str(qid)]
        if clean_ids:
            result[str(dataset_id)] = clean_ids
    return result


def _defeat_last_attempt_reserved_questions(state: EvoState) -> dict[str, int]:
    cache_path = state.get("dataset_states_path", "")
    if not cache_path:
        return {}
    reserved = _clean_reserved_map(state.get("last_attempt_reserved_dataset_question_ids"))
    if not reserved:
        reserved = _clean_reserved_map(state.get("reserved_dataset_question_ids"))
    if not reserved:
        return {}

    from src.tools.dataset_state import DatasetStateManager

    mgr = DatasetStateManager(Path(str(cache_path)))
    counts: dict[str, int] = {}
    round_id = int(state.get("round_id", -1) or -1)
    metrics_after = state.get("metrics_after", {}) if isinstance(state.get("metrics_after"), dict) else {}
    probe_acc = float(
        metrics_after.get("probe_acc")
        or metrics_after.get("probe_acc_frozen")
        or metrics_after.get("probe_acc_after")
        or state.get("champion_frozen_probe_error", 0.0)
        or 0.0
    )
    for dataset_id, question_ids in reserved.items():
        mgr.mark_reserved_defeated(
            dataset_id,
            question_ids,
            round_id=round_id,
            probe_acc=probe_acc,
        )
        counts[dataset_id] = len(question_ids)
    return counts


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _valid_bundle_file(raw_path: str | None, dataset_dir: Path, *, required: bool) -> bool:
    if not raw_path:
        return not required
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        return False
    resolved = path.resolve()
    return _is_relative_to(resolved, dataset_dir)


def _valid_last_dataset_bundle(raw: object, trace_id: str) -> DatasetBundlePayload | None:
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        payload = DatasetBundlePayload.model_validate(raw)
    except Exception:
        return None
    try:
        session_dir = get_session_dir(trace_id).resolve()
        dataset_dir = Path(payload.dataset_dir)
        if dataset_dir.is_symlink() or not dataset_dir.is_dir():
            return None
        resolved_dataset_dir = dataset_dir.resolve()
    except OSError:
        return None
    if not _is_relative_to(resolved_dataset_dir, session_dir):
        return None
    required_paths = (
        payload.train_path,
        payload.cotest_path,
        payload.test_path,
        payload.dataset_info_path,
    )
    if not all(_valid_bundle_file(path, resolved_dataset_dir, required=True) for path in required_paths):
        return None
    optional_paths = (payload.probe_path, payload.lf_val_path)
    if not all(_valid_bundle_file(path, resolved_dataset_dir, required=False) for path in optional_paths):
        return None
    return payload


def _clean_scale(raw: object, fallback: float = 1.0) -> float:
    try:
        if isinstance(raw, int | float | str):
            value = float(raw)
        else:
            return fallback
    except (TypeError, ValueError):
        return fallback
    closest = min(_ALLOWED_MUTATION_SCALES, key=lambda item: abs(item - value))
    return closest if abs(closest - value) <= 0.051 else fallback


def _clamp_scaled(default_value: float, scale: float) -> float:
    low = default_value * MCTS_MUTATION_SCALE_MIN
    high = default_value * MCTS_MUTATION_SCALE_MAX
    return round(min(high, max(low, default_value * scale)), 12)


def _normalize_warmup_fields(hyperparams: dict) -> dict:
    normalized = dict(hyperparams)
    warmup_mode = str(normalized.pop("warmup_mode", "") or "").lower()
    warmup_value = normalized.pop("warmup_value", None)
    if warmup_mode == "ratio":
        try:
            ratio = float(warmup_value)
        except (TypeError, ValueError):
            ratio = None
        if ratio is not None:
            normalized["warmup_ratio"] = min(1.0, max(0.0, ratio))
            normalized.pop("warmup_steps", None)
    elif warmup_mode == "steps":
        try:
            steps = int(float(warmup_value))
        except (TypeError, ValueError):
            steps = None
        if steps is not None:
            normalized["warmup_steps"] = max(0, steps)
            normalized.pop("warmup_ratio", None)
    elif normalized.get("warmup_steps") is not None:
        try:
            normalized["warmup_steps"] = max(0, int(float(normalized["warmup_steps"])))
            normalized.pop("warmup_ratio", None)
        except (TypeError, ValueError):
            normalized.pop("warmup_steps", None)
    elif normalized.get("warmup_ratio") is not None:
        try:
            normalized["warmup_ratio"] = min(1.0, max(0.0, float(normalized["warmup_ratio"])))
        except (TypeError, ValueError):
            normalized.pop("warmup_ratio", None)
    return normalized


def _normalize_training_safety(hyperparams: dict) -> dict:
    normalized = dict(hyperparams)
    finetuning_type = str(normalized.get("finetuning_type", "full") or "full").lower()
    if finetuning_type == "full":
        try:
            batch_size = int(float(normalized.get("per_device_train_batch_size")))
        except (TypeError, ValueError):
            batch_size = 1
        normalized["per_device_train_batch_size"] = min(2, max(1, batch_size))
    elif finetuning_type == "lora":
        # LoRA safety: full-finetune learning rates (1e-6 to 1e-5) are too low
        # for LoRA to converge. This is a domain-knowledge floor, not an
        # override — the LLM's decision is preserved when it is in a valid
        # LoRA range (1e-5 to 1e-3).
        try:
            lr = float(normalized.get("learning_rate", 0))
        except (TypeError, ValueError):
            lr = 0
        if 0 < lr < 1e-5:
            normalized["learning_rate"] = 2e-4
    return normalized


def _apply_parameter_mutation(
    training_hyperparams: dict,
    replay_sample_ratio: float,
    template_defaults: dict,
    edge_summary: dict,
    state: EvoState,
) -> tuple[dict, float, dict]:
    edge_count = len(state.get("search_dag_edges", []) or [])
    if edge_count < MCTS_TUNER_COLD_START_EDGES:
        return training_hyperparams, replay_sample_ratio, {
            "enabled": False,
            "reason": f"cold_start_edges<{MCTS_TUNER_COLD_START_EDGES}",
        }

    fallback = {
        "lr_scale": 1.0,
        "replay_scale": 1.0,
        "reason": "deterministic no-op mutation",
    }
    prompt_state: dict[str, Any] = dict(state)
    decision = decide_json(
        agent_name="parameter_master.mutation",
        prompt=prompt_for_agent(prompt_state, "parameter_master", PARAMETER_MASTER_MUTATION_PROMPT),
        context={
            "round_id": state.get("round_id", 0),
            "edge_summary": edge_summary,
        },
        fallback=fallback,
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
    )
    lr_scale = _clean_scale(decision.get("lr_scale"), 1.0)
    replay_scale = _clean_scale(decision.get("replay_scale"), 1.0)

    mutated_hyperparams = dict(training_hyperparams)
    default_lr = template_defaults.get("learning_rate")
    if default_lr is not None and "learning_rate" in mutated_hyperparams:
        mutated_hyperparams["learning_rate"] = _clamp_scaled(float(default_lr), lr_scale)

    default_replay = float(template_defaults.get("replay_sample_ratio", replay_sample_ratio) or 0.0)
    mutated_replay = _clamp_scaled(default_replay, replay_scale) if default_replay > 0 else replay_sample_ratio
    mutated_replay = min(1.0, max(0.0, mutated_replay))

    return mutated_hyperparams, mutated_replay, {
        "enabled": True,
        "lr_scale": lr_scale,
        "replay_scale": replay_scale,
        "reason": str(decision.get("reason", "")),
        "raw_decision": {
            "lr_scale": decision.get("lr_scale"),
            "replay_scale": decision.get("replay_scale"),
        },
    }


def _forgetting_signal(metrics_after: dict, state: EvoState) -> dict:
    frozen_after = metrics_after.get("probe_acc_frozen", metrics_after.get("probe_acc_after"))
    frozen_champion_error = state.get("champion_frozen_probe_error")
    frozen_degrade = None
    if frozen_after is not None and frozen_champion_error is not None:
        try:
            frozen_degrade = (1.0 - float(frozen_after)) - float(frozen_champion_error)
        except (TypeError, ValueError):
            frozen_degrade = None
    return {
        "old_error_rate": metrics_after.get("old_error_rate", 0.0),
        "probe_acc_frozen": frozen_after,
        "champion_frozen_probe_error": frozen_champion_error,
        "frozen_degrade": frozen_degrade,
        "rollback_streak": state.get("rollback_streak", 0),
        "last_inspection_decision": state.get("last_inspection_decision", ""),
    }


def _decide_training_hyperparams(
    training_hyperparams: dict,
    metrics_after: dict,
    state: EvoState,
    protected_keys: set[str] | None = None,
) -> dict:
    protected_finetuning_type = str(training_hyperparams.get("finetuning_type", "full") or "full").lower()
    prompt_state: dict[str, Any] = dict(state)
    decision = decide_json(
        agent_name="parameter_master.training_hyperparams",
        prompt=prompt_for_agent(prompt_state, "training_hyperparams", TRAINING_HYPERPARAMS_PROMPT),
        context={
            "round_id": state.get("round_id", 0),
            "current_hyperparams": training_hyperparams,
            "probe_acc": metrics_after.get("probe_acc_frozen"),
            "new_skill_acc": metrics_after.get("new_skill_acc", 0.0),
            "forgetting_delta": metrics_after.get("forgetting_delta", 0.0),
            "training_summary": metrics_after.get("training_summary", {}),
            "round_data_stats": state.get("round_data_stats") or {},
            "finetuning_type": protected_finetuning_type,
        },
        fallback=training_hyperparams,
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
    )
    coerced = {}
    for key in _TRAINING_HYPERPARAM_KEYS:
        if key in decision:
            coerced[key] = decision[key]
    if not coerced:
        return _normalize_training_safety(training_hyperparams)
    if "finetuning_type" in coerced:
        coerced["finetuning_type"] = protected_finetuning_type
    merged = {**training_hyperparams, **coerced}
    for key in protected_keys or set():
        if key in training_hyperparams:
            merged[key] = training_hyperparams[key]
    merged["finetuning_type"] = protected_finetuning_type
    return _normalize_training_safety(_normalize_warmup_fields(merged))


def _replay_ratio_fallback(current_ratio: float, signal: dict) -> float:
    try:
        old_error = float(signal.get("old_error_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        old_error = 0.0
    try:
        degrade = float(signal.get("frozen_degrade", 0.0) or 0.0)
    except (TypeError, ValueError):
        degrade = 0.0
    rollback_streak = int(signal.get("rollback_streak", 0) or 0)

    ratio = float(current_ratio)
    if old_error > 0.5 or degrade > 0.03 or rollback_streak >= 2:
        ratio = max(ratio, 0.55)
    elif old_error > 0.25 or degrade > 0.0 or rollback_streak == 1:
        ratio = max(ratio, 0.40)
    return min(1.0, max(0.0, ratio))


def _decide_replay_ratio(
    replay_sample_ratio: float,
    edge_summary: dict,
    metrics_after: dict,
    state: EvoState,
) -> tuple[float, dict]:
    signal = _forgetting_signal(metrics_after, state)
    fallback_ratio = _replay_ratio_fallback(replay_sample_ratio, signal)
    prompt_state: dict[str, Any] = dict(state)
    raw_ratio, ok = decide_json_leaf(
        agent_name="replay_teacher.replay_sample_ratio",
        prompt=prompt_for_agent(prompt_state, "replay_teacher", REPLAY_TEACHER_RATIO_PROMPT),
        context={
            "round_id": state.get("round_id", 0),
            "current_replay_sample_ratio": replay_sample_ratio,
            "forgetting_signal": signal,
            "edge_summary": edge_summary,
        },
        field_name="replay_sample_ratio",
        fallback_value=fallback_ratio,
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
    )
    try:
        ratio = float(raw_ratio)
    except (TypeError, ValueError):
        ratio = fallback_ratio
    ratio = min(1.0, max(0.0, ratio))
    return ratio, {
        "agent": "replay_teacher",
        "accepted_llm": ok,
        "forgetting_signal": signal,
        "fallback_ratio": fallback_ratio,
        "reason": "forgetting-aware replay_sample_ratio",
    }


def _deterministic_retry_same_data_fallback(state: EvoState, action_metadata: dict) -> bool:
    if state.get("last_inspection_decision") not in {"rollback", "prune"}:
        return False
    return bool(action_metadata.get("method_retry_same_data"))


def _decide_retry_same_data(state: EvoState, action_metadata: dict) -> tuple[bool, dict]:
    fallback_retry = _deterministic_retry_same_data_fallback(state, action_metadata)
    repeated_failure = (
        state.get("last_inspection_decision") in {"rollback", "prune"}
        and int(state.get("rollback_streak", 0) or 0) >= 2
    )
    if repeated_failure:
        fallback_retry = False
    data_pressure = action_metadata.get("data_pressure") or state.get("round_data_stats") or {}
    raw_retry, ok = decide_json_leaf(
        agent_name="parameter_master.retry_same_data",
        prompt=prompt_for_agent(dict(state), "parameter_master.retry_same_data", PARAMETER_MASTER_RETRY_DATA_PROMPT),
        context={
            "round_id": state.get("round_id", 0),
            "last_inspection_decision": state.get("last_inspection_decision", ""),
            "rollback_streak": state.get("rollback_streak", 0),
            "data_pressure": data_pressure,
            "has_last_dataset_bundle": bool(state.get("last_dataset_bundle")),
            "has_last_bundle_state": bool(state.get("last_dataset_bundle_state")),
            "has_reserved_questions": bool(state.get("last_attempt_reserved_dataset_question_ids")),
            "deterministic_retry_same_data": fallback_retry,
            "current_action_key": action_metadata.get("action_key", ""),
            "dataset_selection_mode": action_metadata.get("dataset_selection_mode", "single_shard"),
        },
        field_name="retry_same_data",
        fallback_value=fallback_retry,
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
    )
    retry = raw_retry if isinstance(raw_retry, bool) else str(raw_retry).strip().lower() in {"true", "1", "yes"}
    if state.get("last_inspection_decision") not in {"rollback", "prune"}:
        retry = False
    if repeated_failure:
        retry = False
    return bool(retry), {
        "agent": "parameter_master.retry_same_data",
        "accepted_llm": ok,
        "fallback_retry_same_data": fallback_retry,
        "retry_same_data": bool(retry),
        "forced_fresh_dataset": repeated_failure,
        "reason": "rollback data lifecycle leaf decision",
    }


def _coerce_action_parameters(action_key: str, fallback_hyperparams: dict) -> tuple[dict, float]:
    action_space = get_parameter_action_space()
    selected = dict(action_space.get(action_key, {}))
    if not selected:
        return dict(fallback_hyperparams), 0.0
    replay_sample_ratio = float(selected.pop("replay_sample_ratio", 0.0) or 0.0)
    training_hyperparams = {
        key: value
        for key, value in selected.items()
        if key in _TRAINING_HYPERPARAM_KEYS
    }
    return training_hyperparams, min(1.0, max(0.0, replay_sample_ratio))


def _ensure_lora_defaults(hyperparams: dict) -> None:
    from config.settings import LORA_ALPHA, LORA_RANK
    hyperparams.setdefault("lora_rank", LORA_RANK)
    hyperparams.setdefault("lora_alpha", LORA_ALPHA)
    # LoRA needs a higher learning rate than full finetune and enough epochs
    # to converge on small datasets. These are starting defaults — the
    # parameter_master LLM can override them based on eval feedback.
    hyperparams.setdefault("learning_rate", 2e-4)
    hyperparams.setdefault("num_train_epochs", 5)
    hyperparams.setdefault("gradient_accumulation_steps", 1)


def _state_int(state: EvoState, key: str, fallback: int) -> int:
    try:
        return int(state.get(key, fallback) or fallback)
    except (TypeError, ValueError):
        return fallback


def _merge_teacher_sampling_plan(curriculum_plan: dict, teacher_plan: dict | None) -> dict:
    if not isinstance(teacher_plan, dict) or not teacher_plan:
        return curriculum_plan

    merged = dict(curriculum_plan)
    teacher_source = str(teacher_plan.get("source", "") or "")
    teacher_data_pressure = (
        teacher_source == "teacher_data_pressure"
        or bool(teacher_plan.get("data_pressure"))
        or bool(teacher_plan.get("dataset_policy_hint"))
    )

    if teacher_data_pressure and isinstance(teacher_plan.get("difficulty_weights"), dict):
        merged["difficulty_weights"] = dict(teacher_plan["difficulty_weights"])
        merged["source"] = "parameter_master_curriculum+teacher_data_pressure"
    if isinstance(teacher_plan.get("module_weights"), dict) and teacher_plan["module_weights"]:
        merged["module_weights"] = dict(teacher_plan["module_weights"])
    if teacher_plan.get("target_bucket"):
        merged["target_bucket"] = teacher_plan["target_bucket"]
    for key in ("dataset_policy_hint", "data_pressure"):
        if key in teacher_plan:
            merged[key] = teacher_plan[key]
    return merged


def parameter_master_node(state: EvoState) -> dict:
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    prompt_state: dict[str, Any] = dict(state)
    search_request = SearchRequestPayload.model_validate(pending_message.payload)
    metrics_after = state.get("metrics_after") or {}

    action_space = get_parameter_action_space()
    default_template = action_space.get("balanced_default", {})
    preliminary_edge_summary = build_parameter_edge_summary(
        nodes=list(state.get("search_dag_nodes", [])),
        edges=list(state.get("search_dag_edges", [])),
        current_template="balanced_default",
        template_defaults=default_template,
        rollback_streak=int(state.get("rollback_streak", 0) or 0),
        metrics_after=metrics_after,
        previous_decision=str(state.get("last_inspection_decision", "") or ""),
        round_data_stats=state.get("round_data_stats") or {},
    )

    decision = decide_parameter_master_action(
        round_id=int(state.get("round_id", 0) or 0),
        current_search_node_id=str(state.get("current_search_node_id", "") or ""),
        search_dag_nodes=list(state.get("search_dag_nodes", [])),
        search_dag_edges=list(state.get("search_dag_edges", [])),
        previous_decision=str(state.get("last_inspection_decision", "") or ""),
        rollback_streak=int(state.get("rollback_streak", 0) or 0),
        round_data_stats=state.get("round_data_stats") or {},
        edge_summary=preliminary_edge_summary,
        data_window_offset=_state_int(
            state,
            "current_window_offset" if state.get("current_window_offset") is not None else "data_window_offset",
            0,
        ),
        data_window_size=_state_int(
            state,
            "current_window_size" if state.get("current_window_size") is not None else "data_window_size",
            DATASET_SHARD_SIZE,
        ),
        metrics_after=metrics_after,
    )

    print(f"[parameter_master] Round {state.get('round_id', 0)} {decision.decision_summary}")
    decision_metadata = decision.action_metadata if isinstance(decision.action_metadata, dict) else {}
    mcts_metadata = decision_metadata.get("mcts") if isinstance(decision_metadata.get("mcts"), dict) else {}
    continuous_selector = (
        decision_metadata.get("search_space") == "continuous"
        or mcts_metadata.get("action_selector") == "continuous_surrogate_ucb"
    )
    parameter_master_card = build_parameter_master_card(
        round_id=int(state.get("round_id", 0) or 0),
        previous_decision=str(state.get("last_inspection_decision", "") or ""),
        rollback_streak=int(state.get("rollback_streak", 0) or 0),
        metrics_after=metrics_after,
        round_data_stats=state.get("round_data_stats") or {},
        search_dag_nodes=list(state.get("search_dag_nodes", [])),
        search_dag_edges=list(state.get("search_dag_edges", [])),
        decision=decision,
    )
    llm_decision = decide_json(
        agent_name="parameter_master",
        prompt=prompt_for_agent(prompt_state, "parameter_master", PARAMETER_MASTER_LEAN_PROMPT),
        context={
            "parameter_master_card": parameter_master_card,
            "candidate_action_keys": list(action_space.keys()),
        },
        fallback={
            "action_key": decision.action_key,
            "replay_sample_ratio": decision.replay_sample_ratio,
            "dataset_selection_mode": decision.action_metadata.get("dataset_selection_mode", "single_shard"),
            "reason": decision.decision_summary,
        },
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
    )
    requested_action_key = str(llm_decision.get("action_key") or "")
    selected_action_key = str(decision.action_key)
    llm_action_accepted = False
    if continuous_selector:
        llm_action_accepted = requested_action_key in {"", selected_action_key}
        dataset_mode = llm_decision.get(
            "dataset_selection_mode",
            decision.action_metadata.get("dataset_selection_mode", "single_shard"),
        )
        if not llm_action_accepted:
            dataset_mode = decision.action_metadata.get("dataset_selection_mode", "single_shard")
        llm_decision = {
            **llm_decision,
            "action_key": selected_action_key,
            "replay_sample_ratio": decision.replay_sample_ratio,
            "dataset_selection_mode": dataset_mode,
            "reason": str(llm_decision.get("reason", "")),
        }
    else:
        selected_action_key = str(llm_decision.get("action_key") or decision.action_key)
        llm_action_accepted = selected_action_key in action_space
    if not continuous_selector and not llm_action_accepted:
        selected_action_key = decision.action_key
        llm_decision = {
            "action_key": decision.action_key,
            "replay_sample_ratio": decision.replay_sample_ratio,
            "dataset_selection_mode": decision.action_metadata.get("dataset_selection_mode", "single_shard"),
            "finetuning_type": "full",
            "reason": f"invalid LLM action_key rejected; {decision.decision_summary}",
        }

    if continuous_selector:
        training_hyperparams = dict(decision.training_hyperparams)
        action_default_replay_ratio = float(decision.replay_sample_ratio)
    else:
        training_hyperparams, action_default_replay_ratio = _coerce_action_parameters(
            selected_action_key,
            decision.training_hyperparams,
        )
    template_defaults = dict(action_space.get(selected_action_key, {}))
    if not action_default_replay_ratio:
        action_default_replay_ratio = decision.replay_sample_ratio

    template_finetuning = str(
        template_defaults.get("finetuning_type")
        or training_hyperparams.get("finetuning_type")
        or "full"
    ).lower()
    if template_finetuning == "lora":
        training_hyperparams["finetuning_type"] = "lora"
        template_defaults["finetuning_type"] = "lora"
        _ensure_lora_defaults(training_hyperparams)
    else:
        training_hyperparams["finetuning_type"] = "full"
        template_defaults["finetuning_type"] = "full"
        training_hyperparams.pop("lora_rank", None)
        training_hyperparams.pop("lora_alpha", None)
    training_hyperparams = _normalize_training_safety(training_hyperparams)

    replay_sample_ratio = action_default_replay_ratio
    if not continuous_selector:
        try:
            replay_sample_ratio = min(1.0, max(0.0, float(llm_decision.get("replay_sample_ratio", replay_sample_ratio))))
        except (TypeError, ValueError):
            pass
        plan_replay_hint = search_request.sampling_plan.get("replay_sample_ratio_hint")
        if plan_replay_hint is not None:
            try:
                replay_sample_ratio = min(1.0, max(0.0, float(plan_replay_hint)))
            except (TypeError, ValueError):
                pass

    edge_summary = build_parameter_edge_summary(
        nodes=list(state.get("search_dag_nodes", [])),
        edges=list(state.get("search_dag_edges", [])),
        current_template=selected_action_key,
        template_defaults=template_defaults,
        rollback_streak=int(state.get("rollback_streak", 0) or 0),
        metrics_after=metrics_after,
        previous_decision=str(state.get("last_inspection_decision", "") or ""),
        round_data_stats=state.get("round_data_stats") or {},
    )
    if continuous_selector:
        mutation_metadata = {
            "enabled": False,
            "reason": "continuous_selector_owns_params",
        }
    else:
        training_hyperparams, replay_sample_ratio, mutation_metadata = _apply_parameter_mutation(
            training_hyperparams=training_hyperparams,
            replay_sample_ratio=replay_sample_ratio,
            template_defaults=template_defaults,
            edge_summary=edge_summary,
            state=state,
        )
    training_hyperparams = _normalize_training_safety(_normalize_warmup_fields(training_hyperparams))
    lr_before = training_hyperparams.get("learning_rate")
    protected_training_keys = {
        "learning_rate",
        "num_train_epochs",
        "finetuning_type",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "lora_rank",
        "lora_alpha",
    } if continuous_selector else None
    training_hyperparams = _decide_training_hyperparams(
        training_hyperparams=training_hyperparams,
        metrics_after=metrics_after,
        state=state,
        protected_keys=protected_training_keys,
    )
    lr_after = training_hyperparams.get("learning_rate")
    if lr_before is not None and lr_after is not None and abs(float(lr_before) - float(lr_after)) > 1e-8:
        print(f"[parameter_master] LLM adjusted learning_rate: {lr_before} -> {lr_after}")
    replay_sample_ratio, replay_teacher_metadata = _decide_replay_ratio(
        replay_sample_ratio=replay_sample_ratio,
        edge_summary=edge_summary,
        metrics_after=metrics_after,
        state=state,
    )

    curriculum_plan = build_curriculum_sampling_plan(
        probe_easy_acc=metrics_after.get("probe_easy_acc"),
        probe_medium_acc=metrics_after.get("probe_medium_acc"),
        probe_hard_acc=metrics_after.get("probe_hard_acc"),
    )
    curriculum_plan = _merge_teacher_sampling_plan(
        curriculum_plan,
        search_request.sampling_plan,
    )
    round_data_stats = state.get("round_data_stats") or {}
    action_metadata = dict(decision.action_metadata)
    action_metadata["action_key"] = selected_action_key
    action_metadata["training_hyperparams"] = training_hyperparams
    action_metadata["replay_sample_ratio"] = replay_sample_ratio
    if continuous_selector and isinstance(action_metadata.get("continuous_params"), dict):
        action_metadata["continuous_params"] = {
            **action_metadata["continuous_params"],
            "learning_rate": training_hyperparams.get("learning_rate"),
            "replay_sample_ratio": replay_sample_ratio,
            "num_train_epochs": training_hyperparams.get("num_train_epochs"),
            "finetuning_type": training_hyperparams.get("finetuning_type", action_metadata["continuous_params"].get("finetuning_type")),
        }
    action_metadata["edge_summary"] = edge_summary
    action_metadata["parameter_mutation"] = mutation_metadata
    action_metadata["replay_teacher"] = replay_teacher_metadata
    if str(llm_decision.get("dataset_selection_mode", "")) in {"single_shard", "merge_shards"}:
        action_metadata["dataset_selection_mode"] = str(llm_decision["dataset_selection_mode"])
    if action_metadata.get("fresh_dataset_forced"):
        action_metadata["dataset_selection_mode"] = "merge_shards"
    should_reuse_exact_batch, retry_data_metadata = _decide_retry_same_data(state, action_metadata)
    if retry_data_metadata.get("forced_fresh_dataset"):
        action_metadata["fresh_dataset_forced"] = True
        action_metadata["dataset_selection_mode"] = "merge_shards"
        action_metadata["exact_batch_retry"] = False
        action_metadata["exact_batch_retry_unavailable"] = True
        should_reuse_exact_batch = False
    action_metadata["method_retry_same_data"] = should_reuse_exact_batch
    action_metadata["retry_data_teacher"] = retry_data_metadata
    if should_reuse_exact_batch:
        action_metadata["dataset_selection_mode"] = "single_shard"
    action_metadata["llm_agent"] = {
        "agent": "parameter_master",
        "action_key": selected_action_key,
        "action_accepted": llm_action_accepted,
        "mcts_action_key": decision.action_key,
        "finetuning_type": training_hyperparams.get("finetuning_type", "full"),
        "scope": "training_hyperparams_and_replay_ratio",
        "reason": str(llm_decision.get("reason", "")),
    }
    action_metadata["sampling_plan"] = curriculum_plan
    action_metadata["teacher_sampling_plan"] = search_request.sampling_plan
    action_metadata["curriculum_probe_acc"] = {
        "easy": metrics_after.get("probe_easy_acc"),
        "medium": metrics_after.get("probe_medium_acc"),
        "hard": metrics_after.get("probe_hard_acc"),
    }
    print(
        "[parameter_master] curriculum sampling_plan="
        f"{curriculum_plan.get('difficulty_weights', {})}, "
        f"module_weights={curriculum_plan.get('module_weights', {})}, "
        f"probe_difficulty_acc={action_metadata['curriculum_probe_acc']}"
    )

    reusable_bundle = _valid_last_dataset_bundle(state.get("last_dataset_bundle"), trace_id) if should_reuse_exact_batch else None
    if should_reuse_exact_batch and reusable_bundle is not None:
        action_metadata["exact_batch_retry"] = True
        action_metadata["exact_batch_retry_source_round_id"] = state.get("last_dataset_bundle_round_id")
        msg = RoutedMessage(
            header=MessageHeader(
                trace_id=trace_id,
                round_id=round_id,
                sender=AgentName.PARAMETER_MASTER,
                receiver=AgentName.TRAINER,
                message_type=MessageType.DATASET_BUNDLE,
            ),
            payload=reusable_bundle,
        )
        restored_bundle_state = dict(state.get("last_dataset_bundle_state") or {})
        restored_bundle_state.pop("pending_message", None)
        restored_bundle_state.pop("current_training_hyperparams", None)
        restored_bundle_state.pop("current_action_metadata", None)
        return {
            **restored_bundle_state,
            "pending_message": msg,
            "current_training_hyperparams": training_hyperparams,
            "current_action_metadata": action_metadata,
            "replay_sample_ratio_override": replay_sample_ratio,
            "current_search_node_id": decision.branch_parent_node_id,
            "sampling_plan": curriculum_plan,
            "target_bucket": decision.action_metadata.get(
                "target_bucket_override",
                state.get("target_bucket", ""),
            ) or state.get("target_bucket", ""),
            "diagnostic_mode": "train",
            "data_replenishment_needed": False,
            "data_replenishment_exhausted": False,
            "replenishment_cycle_active": False,
            "quota_accumulated_questions": [],
            "quota_shortfall": {},
            "rollout_runs": None,
        }
    if should_reuse_exact_batch and reusable_bundle is None:
        action_metadata["method_retry_same_data"] = False
        action_metadata["exact_batch_retry"] = False
        action_metadata["exact_batch_retry_unavailable"] = True
        action_metadata["dataset_selection_mode"] = "merge_shards"

    defeated_reserved: dict[str, int] = {}
    if state.get("last_inspection_decision") == "rollback" and state.get("last_attempt_reserved_dataset_question_ids"):
        defeated_reserved = _defeat_last_attempt_reserved_questions(state)
        if defeated_reserved:
            action_metadata["defeated_previous_attempt_by_dataset"] = defeated_reserved

    if action_metadata.get("diagnostic_mode") == "probe_diagnostic":
        msg = RoutedMessage(
            header=MessageHeader(
                trace_id=trace_id,
                round_id=round_id,
                sender=AgentName.PARAMETER_MASTER,
                receiver=AgentName.TRAINER,
                message_type=MessageType.DIAGNOSTIC_REQUEST,
            ),
            payload=DiagnosticRequestPayload(
                reason="rollback_streak>=3; run frozen-probe diagnostic without training",
                probe_set_path=state.get("probe_frozen_set_path", state.get("global_probe_set_path", "")),
                champion_model_path=state.get("champion_model_path", ""),
            ),
        )
        return {
            "pending_message": msg,
            "current_training_hyperparams": training_hyperparams,
            "current_action_metadata": action_metadata,
            "replay_sample_ratio_override": replay_sample_ratio,
            "current_search_node_id": decision.branch_parent_node_id,
            "diagnostic_mode": "probe_diagnostic",
            "sampling_plan": curriculum_plan,
            "reserved_dataset_question_ids": {},
            "last_attempt_reserved_dataset_question_ids": {},
        }

    merged_payload = search_request.model_dump()
    merged_payload["sampling_plan"] = curriculum_plan
    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.PARAMETER_MASTER,
            receiver=AgentName.SEARCHER,
            message_type=MessageType.SEARCH_REQUEST,
        ),
        payload=SearchRequestPayload.model_validate(merged_payload),
    )

    result = {
        "pending_message": msg,
        "current_training_hyperparams": training_hyperparams,
        "current_action_metadata": action_metadata,
        "replay_sample_ratio_override": replay_sample_ratio,
        "current_search_node_id": decision.branch_parent_node_id,
        "sampling_plan": curriculum_plan,
        "target_bucket": decision.action_metadata.get(
            "target_bucket_override",
            state.get("target_bucket", ""),
        ) or state.get("target_bucket", ""),
        "diagnostic_mode": action_metadata.get("diagnostic_mode", "train"),
    }
    if defeated_reserved:
        result["reserved_dataset_question_ids"] = {}
        result["last_attempt_reserved_dataset_question_ids"] = {}
        result["round_data_stats"] = {
            **round_data_stats,
            "reserved_finalized_by_dataset": defeated_reserved,
            "reserved_finalized_action": "defeated",
        }
    return result
