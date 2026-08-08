import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.settings import (
    USE_AGENT_INSPECTION,
    get_session_dir,
    PROBE_POOL_MAX_SIZE,
    TEST_BUFFER_MAX_SIZE,
    TEST_BUFFER_OVERFLOW_TO_TRAIN,
)
from src.models.messages import (
    AgentName,
    DiagnosticResultPayload,
    EvalResultPayload,
    InspectionResultPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
)
from src.tools.replay_buffer import save_replay_buffer
from src.tools.search_dag import save_search_dag
from src.tools.question_registry import (
    mark_questions_train_seen,
    mark_questions_active_holdout,
    mark_questions_probe_holdout,
    retire_holdout_questions,
)
from src.tools.agent_prompts import INSPECTION_DECISION_PROMPT, INSPECTION_CONFIDENCE_PROMPT, INSPECTION_REPLAY_PROMPT
from src.tools.llm_decision import decide_json, decide_json_leaf, prompt_for_agent
from src.tools.strategy_policy import (
    INSPECTION_DECISIONS,
    build_replay_entries_policy,
    decide_inspection,
    inspection_gate_blocker_reason,
    update_search_dag_policy,
)
from src.tools.candidate_cleanup import cleanup_candidate_models


def _persist_search_dag(state: Mapping[str, Any]) -> None:
    session_dir = get_session_dir(str(state["trace_id"]))
    save_search_dag(
        nodes=state.get("search_dag_nodes", []),
        edges=state.get("search_dag_edges", []),
        current_node_id=state.get("current_search_node_id", ""),
        session_dir=session_dir,
    )


def _persist_replay_buffer(state: Mapping[str, Any]) -> None:
    session_dir = get_session_dir(str(state["trace_id"]))
    save_replay_buffer(
        entries=state.get("replay_buffer_entries", []),
        session_dir=session_dir,
    )


def _persist_evolution_checkpoint(state: Mapping[str, Any]) -> None:
    session_dir = get_session_dir(str(state["trace_id"]))
    checkpoint_path = session_dir / "evolution_checkpoint.json"
    checkpoint = {
        "trace_id": state.get("trace_id", ""),
        "next_round_id": state.get(
            "checkpoint_next_round_id",
            int(state.get("round_id", 0) or 0) + 1,
        ),
        "champion_model_path": state.get("champion_model_path", ""),
        "current_search_node_id": state.get("current_search_node_id", ""),
        "replay_sample_ratio_override": state.get("replay_sample_ratio_override", 0.30),
        "round_data_stats": state.get("round_data_stats", {}),
        "rollout_difficulty_distribution": state.get("rollout_difficulty_distribution", {}),
        "rollout_hard_ratio": state.get("rollout_hard_ratio", 0.0),
        "last_inspection_decision": state.get("last_inspection_decision", ""),
        "kept_branch_decision": state.get("kept_branch_decision", ""),
        "kept_branch_node_id": state.get("kept_branch_node_id", ""),
        "rollback_streak": state.get("rollback_streak", 0),
        "mastered_memory_set_path": state.get("mastered_memory_set_path", ""),
        "probe_frozen_set_path": state.get("probe_frozen_set_path", ""),
        "frozen_probe_eval_method": state.get("frozen_probe_eval_method", ""),
        "external_probe_path": state.get("external_probe_path", ""),
        "holdout_eval_path": state.get("holdout_eval_path", ""),
        "champion_holdout_baseline": state.get("champion_holdout_baseline"),
        "champion_frozen_probe_error": state.get("champion_frozen_probe_error"),
        "probe_diagnostic_path": state.get("probe_diagnostic_path", ""),
        "probe_diagnostic_focus_path": state.get("probe_diagnostic_focus_path", ""),
        "target_bucket": state.get("target_bucket", ""),
        "last_dataset_bundle": state.get("last_dataset_bundle", {}),
        "last_dataset_bundle_state": state.get("last_dataset_bundle_state", {}),
        "last_dataset_bundle_round_id": state.get("last_dataset_bundle_round_id"),
        "last_attempt_question_ids": state.get("last_attempt_question_ids", []),
        "last_attempt_reserved_dataset_question_ids": state.get("last_attempt_reserved_dataset_question_ids", {}),
        "should_stop": state.get("should_stop", False),
        "budget_exhausted": state.get("budget_exhausted", False),
        "termination_reason": state.get("termination_reason", ""),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def _write_mastered_memory(state: Mapping[str, Any], questions: list[dict]) -> str:
    session_dir = get_session_dir(str(state["trace_id"]))
    path = session_dir / "mastered_memory.json"
    existing: list[dict] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            existing = [q for q in loaded if isinstance(q, dict)]

    seen = {q.get("question_id", "") for q in existing}
    merged = list(existing)
    for q in questions:
        qid = q.get("question_id", "") if isinstance(q, dict) else ""
        if not qid or qid in seen:
            continue
        merged.append(q)
        seen.add(qid)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return str(path)


def _load_json_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _question_key(q: dict) -> tuple[str, object]:
    if not isinstance(q, dict):
        return ("missing", "")
    dataset_id = str(q.get("source_dataset_id") or "")
    row_id = str(q.get("source_dataset_row_id") or "")
    if dataset_id and row_id:
        return (
            "source_row",
            (
                dataset_id,
                str(q.get("source_dataset_subset") or ""),
                str(q.get("source_dataset_split") or ""),
                row_id,
            ),
        )
    question_id = str(q.get("question_id") or "")
    if question_id:
        return ("question_id", question_id)
    return ("text", str(q.get("question_text", "")))


def _persist_pending_buffers(state: Mapping[str, Any]) -> None:
    session_dir = get_session_dir(str(state["trace_id"]))
    pending_test = [
        q for q in state.get("pending_test_buffer_questions", [])
        if isinstance(q, dict)
    ]
    pending_probe = [
        q for q in state.get("pending_probe_pool_questions", [])
        if isinstance(q, dict)
    ]

    if pending_test:
        tb_path = session_dir / "test_buffer.json"
        existing = _load_json_if_exists(tb_path)
        seen = {_question_key(q): True for q in existing}
        for q in pending_test:
            if _question_key(q) not in seen:
                existing.append(q)
                seen[_question_key(q)] = True
        overflow: list[dict] = []
        if len(existing) > TEST_BUFFER_MAX_SIZE:
            overflow = existing[TEST_BUFFER_MAX_SIZE:]
            existing = existing[:TEST_BUFFER_MAX_SIZE]
        with open(tb_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        if overflow and TEST_BUFFER_OVERFLOW_TO_TRAIN:
            print(f"[strategy_inspector] test_buffer overflow: {len(overflow)} questions evicted to training eligibility")

    if pending_probe:
        pp_path = session_dir / "probe_pool.json"
        existing = _load_json_if_exists(pp_path)
        seen = {_question_key(q): True for q in existing}
        for q in pending_probe:
            if _question_key(q) not in seen:
                existing.append(q)
                seen[_question_key(q)] = True
        existing.sort(key=lambda q: float(q.get("pass_rate", 0.0) or 0.0))
        overflow: list[dict] = []
        if len(existing) > PROBE_POOL_MAX_SIZE:
            overflow = existing[PROBE_POOL_MAX_SIZE:]
            existing = existing[:PROBE_POOL_MAX_SIZE]
        with open(pp_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        if overflow and TEST_BUFFER_OVERFLOW_TO_TRAIN:
            print(f"[strategy_inspector] probe_pool overflow: {len(overflow)} questions evicted to training eligibility")


def _persist_committed_round_heldout(state: Mapping[str, Any]) -> None:
    dataset_dir = state.get("dataset_dir", "")
    if not dataset_dir:
        return
    round_heldout = [
        q for q in state.get("round_heldout_questions", [])
        if isinstance(q, dict)
    ]
    if not round_heldout:
        return
    path = Path(dataset_dir) / "round_heldout_questions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(round_heldout, f, ensure_ascii=False, indent=2)


def _reserved_question_ids_by_dataset(state: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = state.get("reserved_dataset_question_ids") or {}
    if isinstance(raw, dict):
        result: dict[str, list[str]] = {}
        for dataset_id, question_ids in raw.items():
            if not isinstance(question_ids, list):
                continue
            clean_ids = [str(qid) for qid in question_ids if str(qid)]
            if clean_ids:
                result[str(dataset_id)] = clean_ids
        if result:
            return result

    by_dataset: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    fallback_fields = (
        "train_questions",
        "lf_val_questions",
        "cotest_questions",
        "new_test_questions",
        "probe_pool_intake_questions",
    )
    for field in fallback_fields:
        for question in state.get(field, []) or []:
            if not isinstance(question, dict) or question.get("source_role") == "replay":
                continue
            dataset_id = str(question.get("source_dataset_id") or "")
            question_id = str(question.get("question_id") or "")
            if not dataset_id or not question_id:
                continue
            dataset_seen = seen.setdefault(dataset_id, set())
            if question_id in dataset_seen:
                continue
            dataset_seen.add(question_id)
            by_dataset.setdefault(dataset_id, []).append(question_id)
    return by_dataset


def _method_retry_same_data(state: Mapping[str, Any]) -> bool:
    action_metadata = state.get("current_action_metadata") or {}
    return bool(
        isinstance(action_metadata, dict)
        and action_metadata.get("method_retry_same_data")
    )


def _has_reusable_last_attempt(state: Mapping[str, Any]) -> bool:
    bundle = state.get("last_dataset_bundle")
    reserved = state.get("last_attempt_reserved_dataset_question_ids")
    return bool(isinstance(bundle, dict) and bundle and isinstance(reserved, dict) and reserved)


def _finalize_reserved_questions(state: Mapping[str, Any], decision: str) -> dict[str, int]:
    cache_path = state.get("dataset_states_path", "")
    if not cache_path:
        return {}
    reserved = _reserved_question_ids_by_dataset(state)
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
        if decision in {"promote", "provisional_promote"}:
            mgr.mark_reserved_used(dataset_id, question_ids)
            counts[dataset_id] = len(question_ids)
        elif decision == "rollback" and _has_reusable_last_attempt(state):
            counts[dataset_id] = len(question_ids)
        else:
            mgr.mark_reserved_defeated(
                dataset_id,
                question_ids,
                round_id=round_id,
                probe_acc=probe_acc,
            )
            counts[dataset_id] = len(question_ids)
    return counts


def _llm_inspection_override(
    state: dict[str, Any],
    metrics: EvalResultPayload,
    policy,
) -> tuple[str, str, float, bool]:
    base_ctx = {
        "round_id": state.get("round_id", 0),
        "metrics": metrics.model_dump(mode="json"),
        "rollback_streak": state.get("rollback_streak", 0),
        "search_dag_edges_tail": list(state.get("search_dag_edges", []))[-5:],
        "deterministic_fallback": {
            "decision": policy.decision,
            "reason": policy.reason,
        },
    }
    trace_id = state.get("trace_id", "")
    round_id = state.get("round_id", 0)

    # Leaf 1: 先决定 promote/rollback/prune
    if USE_AGENT_INSPECTION:
        decision_raw = decide_json(
            agent_name="inspection_agent.decision",
            prompt=prompt_for_agent(state, "inspection_agent.decision", INSPECTION_DECISION_PROMPT),
            context=base_ctx,
            fallback={"decision": policy.decision, "reason": policy.reason},
            trace_id=trace_id,
            round_id=round_id,
        )
        chosen = str(decision_raw.get("decision", policy.decision))
        reason_str = str(decision_raw.get("reason", policy.reason))
    else:
        chosen = policy.decision
        reason_str = policy.reason
    if chosen not in INSPECTION_DECISIONS:
        chosen = policy.decision
    # Guard: block promote when gates fail — training can't override physics
    if chosen in {"promote", "provisional_promote", "keep_branch"} and (
        not metrics.pass_frozen_gate
        or not metrics.pass_old_skill_gate
    ):
        chosen = "rollback"
        reason_str = inspection_gate_blocker_reason(metrics)
    if chosen == "promote" and not metrics.pass_new_skill_gate:
        chosen = "provisional_promote"
        reason_str = inspection_gate_blocker_reason(metrics)
    # Guard: training failure always blocks promotion (candidate is unloadable)
    training_failed = bool(state.get("metrics_after", {}).get("training_failed"))
    if training_failed:
        chosen = "rollback"
        reason_str = "candidate training failed"
    if chosen == "rollback" and not training_failed:
        reason_str = inspection_gate_blocker_reason(metrics)

    # Leaf 2: 基于决策给出置信度
    confidence_ctx = {**base_ctx, "decision": chosen}
    confidence_val, _ok = decide_json_leaf(
        agent_name="inspection_agent.confidence",
        prompt=prompt_for_agent(state, "inspection_agent.confidence", INSPECTION_CONFIDENCE_PROMPT),
        context=confidence_ctx,
        field_name="confidence",
        fallback_value=policy.confidence,
        trace_id=trace_id,
        round_id=round_id,
    )
    try:
        confidence = float(confidence_val if confidence_val is not None else policy.confidence)
    except (TypeError, ValueError):
        confidence = policy.confidence
    confidence = max(0.0, min(1.0, confidence))

    # Leaf 3: 基于决策决定是否存入 replay buffer
    replay_val, _ok = decide_json_leaf(
        agent_name="inspection_agent.replay",
        prompt=prompt_for_agent(state, "inspection_agent.replay", INSPECTION_REPLAY_PROMPT),
        context=confidence_ctx,
        field_name="should_store_to_replay_buffer",
        fallback_value=policy.should_store_to_replay_buffer,
        trace_id=trace_id,
        round_id=round_id,
    )
    should_store = bool(replay_val if replay_val is not None else policy.should_store_to_replay_buffer)

    return chosen, f"decision={chosen} confidence={confidence:.2f} agent_reason='{reason_str}'", confidence, should_store


def strategy_inspector_node(state: dict[str, Any]) -> dict:
    pending_message = state["pending_message"]
    if pending_message.header.message_type == MessageType.DIAGNOSTIC_RESULT:
        return _handle_diagnostic_result(state)

    metrics = EvalResultPayload.model_validate(pending_message.payload)
    round_id = int(state.get("round_id", 0) or 0)
    trace_id = str(state.get("trace_id") or "")
    rollback_streak = int(state.get("rollback_streak", 0) or 0)

    policy = decide_inspection(metrics, round_id, rollback_streak, list(state.get("search_dag_edges", []))[-5:])
    final_decision, final_reason, final_confidence, should_store_to_replay = _llm_inspection_override(
        state,
        metrics,
        policy,
    )

    inspection = InspectionResultPayload(
        decision=final_decision,
        reason=final_reason,
        confidence=final_confidence,
        should_store_to_replay_buffer=should_store_to_replay,
        should_update_checkpoint=policy.should_update_checkpoint,
        should_update_search_dag=True,
    )

    next_receiver = AgentName.SYSTEM if policy.should_terminate else AgentName.TEACHER

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.STRATEGY_INSPECTOR,
            receiver=next_receiver,
            message_type=MessageType.INSPECTION_RESULT,
        ),
        payload=inspection,
    )

    updates: dict = {
        "pending_message": msg,
        "should_stop": policy.achieved_target,
        "should_promote_candidate": final_decision == "promote",
        "budget_exhausted": policy.budget_exhausted,
        "termination_reason": policy.termination_reason,
        "checkpoint_next_round_id": round_id + 1,
        "last_inspection_decision": final_decision,
        "probe_diagnostic_focus_path": "",
        "rollback_streak": (int(state.get("rollback_streak", 0) or 0) + 1)
        if final_decision == "rollback"
        else 0,
        "kept_branch_decision": final_decision if final_decision in {"provisional_promote", "keep_branch"} else "",
    }

    finalized_reserved = _finalize_reserved_questions(state, final_decision)
    if finalized_reserved:
        if final_decision in {"promote", "provisional_promote"}:
            lifecycle_action = "used"
        elif final_decision == "rollback" and _has_reusable_last_attempt(state):
            lifecycle_action = "reserved_for_retry"
        else:
            lifecycle_action = "defeated"
        if lifecycle_action != "reserved_for_retry":
            updates["reserved_dataset_question_ids"] = {}
        updates["round_data_stats"] = {
            **(state.get("round_data_stats") or {}),
            "reserved_finalized_by_dataset": finalized_reserved,
            "reserved_finalized_action": lifecycle_action,
        }
        print(
            "[strategy_inspector] Finalized reserved questions: "
            f"action={lifecycle_action} counts={finalized_reserved}"
        )

    # Rollback restores the previous global heldout checkpoint by not committing
    # this round's test/probe registry mutations.
    if final_decision != "rollback":
        _persist_pending_buffers(state)
        _persist_committed_round_heldout(state)
        round_heldout = [
            q for q in state.get("round_heldout_questions", [])
            if isinstance(q, dict)
        ]
        if round_heldout:
            mark_questions_active_holdout(
                state.get("heldout_registry_path") or str(get_session_dir(trace_id) / "heldout_registry.json"),
                round_heldout,
                metadata={"round_id": round_id, "decision": final_decision},
            )
        round_probe_pool = [
            q for q in state.get("round_probe_pool_questions", [])
            if isinstance(q, dict)
        ]
        if round_probe_pool:
            mark_questions_probe_holdout(
                state.get("heldout_registry_path") or str(get_session_dir(trace_id) / "heldout_registry.json"),
                round_probe_pool,
                metadata={"round_id": round_id, "decision": final_decision},
            )
        retired_holdout = [
            q for q in state.get("round_retired_holdout_questions", [])
            if isinstance(q, dict)
        ]
        if retired_holdout:
            retire_holdout_questions(
                state.get("heldout_registry_path") or str(get_session_dir(trace_id) / "heldout_registry.json"),
                retired_holdout,
                metadata={"round_id": round_id, "decision": final_decision},
            )
            mark_questions_train_seen(
                state.get("heldout_registry_path") or str(get_session_dir(trace_id) / "heldout_registry.json"),
                retired_holdout,
                metadata={"round_id": round_id, "decision": final_decision, "reason": "test_buffer_overflow_to_train"},
            )

        train_seen = [
            q for q in state.get("train_questions", [])
            if isinstance(q, dict)
        ]
        train_seen.extend(
            q for q in state.get("lf_val_questions", [])
            if isinstance(q, dict)
        )
        if train_seen:
            mark_questions_train_seen(
                state.get("heldout_registry_path") or str(get_session_dir(trace_id) / "heldout_registry.json"),
                train_seen,
                metadata={"round_id": round_id, "decision": final_decision},
            )

    # Promote: champion model + mastered memory
    previous_champion_model_path = str(state.get("champion_model_path") or "")
    if final_decision == "promote":
        updates["champion_model_path"] = state["candidate_model_path"]
        action_metadata = state.get("current_action_metadata", {})
        if isinstance(action_metadata, dict) and action_metadata.get("data_window_offset") is not None:
            committed_offset = int(action_metadata.get("data_window_offset", 0) or 0)
        else:
            committed_offset = int(state.get("data_window_offset", 0) or 0)
        updates["data_window_offset"] = committed_offset + int(
            state.get("data_window_size", 0) or 0
        )

        scored_questions_raw = state.get("mastered_questions", [])
        mastered_questions = []
        for q in scored_questions_raw:
            if isinstance(q, dict):
                mastered_questions.append(q)
        if mastered_questions:
            master_path = _write_mastered_memory(state, mastered_questions)
            updates["mastered_memory_set_path"] = str(master_path)

        _revive_deferred_if_improved(state, metrics)

    # Search DAG: append new node + edge for this round
    parent_node_id = state.get("current_search_node_id", "")
    current_nodes = list(state.get("search_dag_nodes", []))
    current_edges = list(state.get("search_dag_edges", []))
    replay_used = int(state.get("replay_buffer_used_count", 0) or 0) > 0
    current_action_metadata = dict(state.get("current_action_metadata", {}) or {})
    metrics_after = state.get("metrics_after", {}) if isinstance(state.get("metrics_after"), dict) else {}
    training_summary = metrics_after.get("training_summary")
    if isinstance(training_summary, dict):
        current_action_metadata["training_summary"] = training_summary

    dag_result = update_search_dag_policy(
        nodes=current_nodes,
        edges=current_edges,
        parent_node_id=parent_node_id,
        decision=final_decision,
        metrics=metrics,
        round_id=round_id,
        replay_used=replay_used,
        action_metadata=current_action_metadata,
    )

    updates["search_dag_nodes"] = dag_result.nodes
    updates["search_dag_edges"] = dag_result.edges
    updates["current_search_node_id"] = dag_result.next_parent_node_id
    if final_decision in {"provisional_promote", "keep_branch"}:
        updates["kept_branch_node_id"] = getattr(
            dag_result,
            "new_node_id",
            dag_result.next_parent_node_id,
        )

    # Replay buffer tracks only promoted champion training data.
    replay_questions = list(state.get("train_questions", []))
    if final_decision == "promote":
        updates["champion_holdout_baseline"] = state.get("champion_holdout_baseline")
        updates["champion_frozen_probe_error"] = state.get("champion_frozen_probe_error")
    if should_store_to_replay and final_decision == "promote":
        updates["replay_buffer_entries"] = build_replay_entries_policy(
            filtered_questions=replay_questions,
            metrics=metrics,
            round_id=round_id,
            decision=final_decision,
            existing_entries=list(state.get("replay_buffer_entries", [])),
        )

    # Persist mechanism state for cross-process resume.
    persisted_state = {**state, **updates}
    persisted_ok = False
    try:
        _persist_search_dag(persisted_state)
        _persist_replay_buffer(persisted_state)
        _persist_evolution_checkpoint(persisted_state)
        persisted_ok = True
    except Exception as exc:
        print(f"[strategy_inspector] Persist warning: {exc}")
    if persisted_ok:
        try:
            cleanup_candidate_models(persisted_state)
        except Exception as exc:
            print(f"[strategy_inspector] Candidate cleanup warning: {exc}")
        try:
            from src.tools.model_runner import clear_non_champion_vllm_actors

            clear_non_champion_vllm_actors(
                str(persisted_state.get("champion_model_path") or ""),
                candidate_model_path=str(state.get("candidate_model_path") or ""),
                previous_champion_model_path=previous_champion_model_path,
            )
        except Exception as exc:
            print(f"[strategy_inspector] vLLM actor cleanup warning: {exc}")

    trusted_probe = metrics.probe_acc_frozen
    if trusted_probe is None:
        trusted_probe = metrics.probe_acc_after
    print(f"[strategy_inspector] Round {round_id} decision={final_decision} "
          f"reason='{final_reason}' probe_acc={metrics.probe_acc_after:.3f} "
          f"probe_acc_frozen={trusted_probe:.3f} "
          f"pass_frozen_gate={metrics.pass_frozen_gate} "
          f"achieved_target={policy.achieved_target} budget_exhausted={policy.budget_exhausted} "
          f"next={next_receiver.value}{' (terminate)' if policy.should_terminate else ''}")

    if next_receiver == AgentName.TEACHER:
        updates["round_id"] = round_id + 1

    return updates


def _handle_diagnostic_result(state: dict[str, Any]) -> dict:
    pending_message = state["pending_message"]
    diagnostic = DiagnosticResultPayload.model_validate(pending_message.payload)
    round_id = int(state.get("round_id", 0) or 0)
    trace_id = str(state.get("trace_id") or "")

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.STRATEGY_INSPECTOR,
            receiver=AgentName.TEACHER,
            message_type=MessageType.INSPECTION_RESULT,
        ),
        payload=InspectionResultPayload(
            decision="diagnostic",
            reason="probe diagnostic completed; skipped training and redirected next round",
            confidence=1.0,
            should_store_to_replay_buffer=False,
            should_update_checkpoint=False,
            should_update_search_dag=False,
        ),
    )

    updates = {
        "pending_message": msg,
        "checkpoint_next_round_id": round_id + 1,
        "last_inspection_decision": "diagnostic",
        "diagnostic_mode": "train",
        "probe_diagnostic_path": diagnostic.diagnostic_path,
        "probe_diagnostic_focus_path": diagnostic.worst_probe_path,
        "target_bucket": diagnostic.target_bucket,
        "round_id": round_id + 1,
    }

    try:
        persisted_state = {**state, **updates}
        _persist_evolution_checkpoint(persisted_state)
    except Exception as exc:
        print(f"[strategy_inspector] Diagnostic persist warning: {exc}")

    print(
        f"[strategy_inspector] Round {round_id} diagnostic_complete "
        f"target_bucket={diagnostic.target_bucket} worst_probe_count={diagnostic.worst_probe_count} "
        "next=teacher"
    )
    return updates


def _revive_deferred_if_improved(state: Mapping[str, Any], metrics) -> None:
    from pathlib import Path

    from config.settings import PROBE_IMPROVEMENT_REFRESH_THRESHOLD
    from src.tools.dataset_state import DatasetStateManager

    cache_path = state.get("dataset_states_path", "")
    if not cache_path:
        return
    probe_acc = metrics.probe_acc_frozen or metrics.probe_acc_after or 0.0
    mgr = DatasetStateManager(Path(cache_path))
    mgr.load_all_cached()
    revived_total = 0
    for dataset_id in list(mgr.datasets.keys()):
        if mgr.load_cached(dataset_id) is None:
            continue
        revived = mgr.revive_defeated_if_improved(
            dataset_id,
            probe_acc,
            PROBE_IMPROVEMENT_REFRESH_THRESHOLD,
        )
        if revived:
            revived_total += revived
            print(
                f"[strategy_inspector] Revived {revived} defeated items in {dataset_id} "
                f"after probe improvement to {probe_acc:.3f}"
            )
    if revived_total:
        print(f"[strategy_inspector] Total revived defeated items: {revived_total}")
