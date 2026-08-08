import json
from typing import Any

from config.settings import get_session_dir
from src.models.state import EvoState
from src.tools.strategy_policy import build_sampling_plan, decide_teacher_search
from src.tools.agent_prompts import (
    TEACHING_TEACHER_DATASET_POLICY_PROMPT,
    TEACHING_TEACHER_DIFFICULTY_PROMPT,
    TEACHING_TEACHER_DIFFICULTY_WEIGHTS_PROMPT,
    TEACHING_TEACHER_SEARCH_QUERY_PROMPT,
)
from src.tools.llm_decision import clamp_distribution, decide_json_leaf, prompt_for_agent
from src.tools.search_query import normalize_content_search_query

_DYNAMIC_DIFFICULTIES = {"easy", "medium", "hard", "unknown"}
_DATASET_POLICIES = {"single_shard", "merge_shards", "merge_or_replace"}


def _clean_search_query(raw: object, fallback: str, goal: str = "") -> str:
    return normalize_content_search_query(raw, goal=goal, fallback=fallback)[:160]


def _clean_enum(raw: object, allowed: set[str], fallback: str) -> str:
    """将 LLM 输出的枚举值清洗为合法值，支持模糊匹配。

    1. 精确匹配 → 直接返回
    2. 包含匹配 → 返回被包含的合法枚举值（"super_easy" 包含 "easy" → "easy"）
    3. 都不匹配 → 返回 fallback
    """
    value = str(raw or "").strip().lower()
    if value in allowed:
        return value
    # 模糊匹配：检查 value 中是否包含某个合法枚举值
    matches = [a for a in allowed if a in value]
    if matches:
        # 多个匹配时取最长（最具体），如 "hard" vs "harder" 场景
        return max(matches, key=len)
    return fallback


def _apply_leaf_teacher_decisions(
    state: EvoState,
    fallback_decision: dict,
    sampling_plan: dict,
    round_id: int,
) -> tuple[dict, list[str]]:
    trace_id = state.get("trace_id", "")
    metrics_after = state.get("metrics_after") or {}
    per_difficulty_acc = (
        metrics_after.get("per_difficulty_acc_frozen")
        or metrics_after.get("per_difficulty_acc")
        or {
            key: value
            for key, value in {
                "easy": metrics_after.get("probe_easy_acc"),
                "medium": metrics_after.get("probe_medium_acc"),
                "hard": metrics_after.get("probe_hard_acc"),
            }.items()
            if value is not None
        }
    )
    round_data_stats = state.get("round_data_stats") or {}
    difficulty_feedback = state.get("difficulty_teacher_feedback") or {}
    successes: list[str] = []
    prompt_state: dict[str, Any] = dict(state)

    raw_query, ok = decide_json_leaf(
        agent_name="teaching_teacher.search_query",
        prompt=prompt_for_agent(
            prompt_state,
            "teaching_teacher.search_query",
            TEACHING_TEACHER_SEARCH_QUERY_PROMPT,
        ),
        context={
            "goal": state.get("user_goal", ""),
            "round_id": round_id,
            "per_difficulty_acc": per_difficulty_acc,
            "data_pressure": {
                "hard_ratio": round_data_stats.get("hard_ratio", 0.0),
                "low_train_signal": round_data_stats.get("low_train_signal", False),
                "hard_dominated_signal": round_data_stats.get("hard_dominated_signal", False),
            },
            "difficulty_teacher_feedback": difficulty_feedback,
        },
        field_name="search_query",
        fallback_value=fallback_decision["search_query"],
        trace_id=trace_id,
        round_id=round_id,
        max_new_tokens=1024,
        disable_thinking=False,
    )
    search_query = _clean_search_query(
        raw_query,
        fallback_decision["search_query"],
        goal=str(state.get("user_goal", "")),
    )
    if ok and search_query != fallback_decision["search_query"]:
        successes.append("search_query")

    raw_difficulty, ok = decide_json_leaf(
        agent_name="teaching_teacher.target_difficulty",
        prompt=prompt_for_agent(
            prompt_state,
            "teaching_teacher.target_difficulty",
            TEACHING_TEACHER_DIFFICULTY_PROMPT,
        ),
        context={
            "round_id": round_id,
            "per_difficulty_acc": per_difficulty_acc,
            "rollout_difficulty_distribution": state.get("rollout_difficulty_distribution", {}),
            "rollout_hard_ratio": state.get("rollout_hard_ratio", 0.0),
            "data_pressure": round_data_stats,
            "difficulty_teacher_feedback": difficulty_feedback,
        },
        field_name="target_difficulty",
        fallback_value=fallback_decision["target_difficulty"],
        trace_id=trace_id,
        round_id=round_id,
        max_new_tokens=1024,
        disable_thinking=False,
    )
    target_difficulty = _clean_enum(
        raw_difficulty,
        _DYNAMIC_DIFFICULTIES,
        fallback_decision["target_difficulty"],
    )
    if ok and target_difficulty != fallback_decision["target_difficulty"]:
        successes.append("target_difficulty")

    raw_weights, ok = decide_json_leaf(
        agent_name="teaching_teacher.difficulty_weights",
        prompt=prompt_for_agent(
            prompt_state,
            "teaching_teacher.difficulty_weights",
            TEACHING_TEACHER_DIFFICULTY_WEIGHTS_PROMPT,
        ),
        context={
            "round_id": round_id,
            "target_difficulty": target_difficulty,
            "per_difficulty_acc": per_difficulty_acc,
            "rollout_difficulty_distribution": state.get("rollout_difficulty_distribution", {}),
            "rollout_hard_ratio": state.get("rollout_hard_ratio", 0.0),
            "difficulty_threshold_policy": state.get("difficulty_threshold_policy", {}),
            "data_pressure": round_data_stats,
            "difficulty_teacher_feedback": difficulty_feedback,
        },
        field_name="difficulty_weights",
        fallback_value=fallback_decision["difficulty_weights"],
        trace_id=trace_id,
        round_id=round_id,
        max_new_tokens=1024,
        disable_thinking=False,
    )
    difficulty_weights = clamp_distribution(
        raw_weights,
        _DYNAMIC_DIFFICULTIES - {"unknown"},
        sampling_plan.get("difficulty_weights") or {"easy": 0.15, "medium": 0.70, "hard": 0.15},
    )
    if ok and difficulty_weights != fallback_decision["difficulty_weights"]:
        successes.append("difficulty_weights")

    raw_policy, ok = decide_json_leaf(
        agent_name="teaching_teacher.dataset_policy_hint",
        prompt=prompt_for_agent(
            prompt_state,
            "teaching_teacher.dataset_policy_hint",
            TEACHING_TEACHER_DATASET_POLICY_PROMPT,
        ),
        context={
            "round_id": round_id,
            "data_pressure": round_data_stats,
            "rollout_hard_ratio": state.get("rollout_hard_ratio", 0.0),
            "rollback_streak": state.get("rollback_streak", 0),
            "difficulty_teacher_feedback": difficulty_feedback,
        },
        field_name="dataset_policy_hint",
        fallback_value=fallback_decision["dataset_policy_hint"],
        trace_id=trace_id,
        round_id=round_id,
        max_new_tokens=1024,
        disable_thinking=False,
    )
    dataset_policy_hint = _clean_enum(
        raw_policy,
        _DATASET_POLICIES,
        fallback_decision["dataset_policy_hint"],
    )
    if ok and dataset_policy_hint != fallback_decision["dataset_policy_hint"]:
        successes.append("dataset_policy_hint")

    return {
        "search_query": search_query,
        "target_difficulty": target_difficulty,
        "difficulty_weights": difficulty_weights,
        "module_weights": fallback_decision.get("module_weights", {}),
        "dataset_policy_hint": dataset_policy_hint,
        "reason": (
            "leaf LLM teacher decisions: " + ",".join(successes)
            if successes
            else fallback_decision["reason"]
        ),
    }, successes


def _update_teacher_decisions(state: EvoState, target_bucket: str, search_query: str) -> None:
    """记录教学教师本轮聚焦的 rollout 动态难度。"""
    trace_id = str(state.get("trace_id", ""))
    session_dir = get_session_dir(trace_id)
    decisions_path = session_dir / "teacher_decisions.jsonl"
    round_id = state.get("round_id", 0)
    metrics_after = state.get("metrics_after") or {}

    updated_lines = []
    found = False
    if decisions_path.exists():
        with open(decisions_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("round") == round_id and not found:
                    entry["weakest_bucket"] = target_bucket
                    entry["decision"] = f"focus on {target_bucket} difficulty, query='{search_query}'"
                    found = True
                updated_lines.append(entry)

    if not found:
        entry = {
            "round": round_id,
            "input_per_difficulty_acc": metrics_after.get("per_difficulty_acc_frozen", {}),
            "weakest_bucket": target_bucket,
            "decision": f"focus on {target_bucket} difficulty, query='{search_query}'",
        }
        updated_lines.append(entry)

    with open(decisions_path, "w", encoding="utf-8") as f:
        for entry in updated_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def teacher_node(state: EvoState) -> dict:
    from config.settings import get_classifier_labels
    from src.models.messages import (
        AgentName,
        GoalRequestPayload,
        MessageHeader,
        MessageType,
        RoutedMessage,
        SearchRequestPayload,
    )

    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    trace_id = str(state.get("trace_id", ""))
    message_type = pending_message.header.message_type
    round_id = state.get("round_id", 0)
    round_data_stats = state.get("round_data_stats") or {}
    sampling_plan = {}

    if message_type == MessageType.GOAL_REQUEST and round_id == 0:
        goal_payload = GoalRequestPayload.model_validate(pending_message.payload)
        policy = decide_teacher_search(
            goal=goal_payload.goal,
            round_id=round_id,
            metrics_after=state.get("metrics_after") or {},
            default_bucket="hard",
            round_data_stats=round_data_stats,
        )
        search_query = policy.search_query
        target_bucket = ""
        sampling_plan = policy.sampling_plan
    elif state.get("probe_diagnostic_focus_path"):
        target_bucket = state.get("target_bucket", "hard")
        base_goal = str(state.get("user_goal", "") or "reasoning dataset").strip()
        search_query = f"{base_goal} {target_bucket or 'hard'} probe failure"
        focus_modules = _load_focus_module_counts(state.get("probe_diagnostic_focus_path", ""))
        sampling_plan = build_sampling_plan(target_bucket, focus_modules=focus_modules)
        sampling_plan["source"] = "probe_diagnostic_focus"
        _update_teacher_decisions(state, target_bucket, search_query)
    elif round_id > 0:
        policy = decide_teacher_search(
            goal=state.get("user_goal", ""),
            round_id=round_id,
            metrics_after=state.get("metrics_after") or {},
            default_bucket="hard",
            round_data_stats=round_data_stats,
        )
        search_query = policy.search_query
        target_bucket = policy.target_bucket
        sampling_plan = policy.sampling_plan
        _update_teacher_decisions(state, target_bucket, search_query)
    else:
        policy = decide_teacher_search(
            goal=state.get("user_goal", ""),
            round_id=round_id,
            metrics_after=state.get("metrics_after") or {},
            default_bucket="hard",
            round_data_stats=round_data_stats,
        )
        search_query = policy.search_query
        target_bucket = ""
        sampling_plan = policy.sampling_plan

    fallback_decision = {
        "search_query": search_query,
        "target_difficulty": target_bucket,
        "difficulty_weights": sampling_plan.get("difficulty_weights", {}),
        "module_weights": sampling_plan.get("module_weights", {}),
        "dataset_policy_hint": sampling_plan.get("dataset_policy_hint", "single_shard"),
        "reason": "deterministic teacher fallback",
    }
    llm_decision, _leaf_successes = _apply_leaf_teacher_decisions(
        state=state,
        fallback_decision=fallback_decision,
        sampling_plan=sampling_plan,
        round_id=round_id,
    )
    search_query = normalize_content_search_query(
        llm_decision.get("search_query") or search_query,
        goal=state.get("user_goal", ""),
        fallback=search_query,
    )
    target_bucket = str(
        llm_decision.get("target_difficulty")
        or llm_decision.get("target_bucket")
        or target_bucket
    )
    if target_bucket not in _DYNAMIC_DIFFICULTIES:
        target_bucket = fallback_decision["target_difficulty"]
    sampling_plan = dict(sampling_plan)
    sampling_plan["difficulty_weights"] = clamp_distribution(
        llm_decision.get("difficulty_weights"),
        _DYNAMIC_DIFFICULTIES - {"unknown"},
        sampling_plan.get("difficulty_weights") or {"easy": 0.15, "medium": 0.70, "hard": 0.15},
    )
    if isinstance(llm_decision.get("module_weights"), dict):
        sampling_plan["module_weights"] = clamp_distribution(
            llm_decision.get("module_weights"),
            set(str(k) for k in llm_decision.get("module_weights", {}).keys()),
            sampling_plan.get("module_weights") or {},
        )
    sampling_plan["target_difficulty"] = target_bucket
    sampling_plan["target_bucket"] = target_bucket
    if llm_decision.get("dataset_policy_hint"):
        sampling_plan["dataset_policy_hint"] = str(llm_decision["dataset_policy_hint"])
    sampling_plan["llm_agent"] = {
        "agent": "teaching_teacher",
        "reason": str(llm_decision.get("reason", "")),
    }

    payload = SearchRequestPayload(
        search_sources=["huggingface", "web"],
        search_query=search_query,
        goal=state.get("user_goal", ""),
        retrieval_mode="full_dataset",
        dataset_role="train_dataset",
        sampling_owner="filter",
        target_labels=get_classifier_labels(),
        sampling_plan=sampling_plan,
    )

    print(f"[teacher] Round {round_id}: search_query='{search_query}' target_bucket='{target_bucket}'")
    print(f"[teacher] Strategy: retrieval_mode=full_dataset, sampling_owner=filter")
    print(f"[teacher] Sampling plan: difficulty={sampling_plan.get('difficulty_weights', {})}")

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.TEACHER,
            receiver=AgentName.PARAMETER_MASTER,
            message_type=MessageType.SEARCH_REQUEST,
        ),
        payload=payload,
    )

    return {
        "pending_message": msg,
        "target_bucket": target_bucket,
        "sampling_plan": sampling_plan,
    }


def _load_focus_module_counts(path: str) -> dict[str, int]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return {}
    counts: dict[str, int] = {}
    if not isinstance(loaded, list):
        return counts
    for item in loaded:
        if not isinstance(item, dict):
            continue
        module = str(item.get("module", "") or "")
        if not module or module == "unknown":
            continue
        counts[module] = counts.get(module, 0) + 1
    return counts
