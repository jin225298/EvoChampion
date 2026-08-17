# =============================================================================
# rollout_aggregator.py — Rollout 聚合器节点
# =============================================================================
# 本节点位于 N 路并行 rollout 之后，负责：
#   1. 合并所有 parallel worker 的推理结果
#   2. 统计每道题的 pass_count / pass_rate / 动态难度
#   3. 调用 Difficulty Teacher（4 个 leaf LLM）生成采样建议
#   4. 始终发给 filter_post_rollout（继续主流程），由窗口补齐机制处理配额不足
#
# 工作流程：
#   rollout_worker × N（并行推理）
#        ↓ 结果通过 operator.add reducer 聚合到 state.rollout_runs
#   rollout_aggregator_node()
#        ├─ 合并校正：按 question_id 合并 worker 的单次 cascade 结果
#        ├─ 难度标注：保留 worker 的 cascade stage 标签
#        ├─ 难度教师：4 个 leaf LLM 生成 advisory metadata
#        └─ 路由：filter_post_rollout
# =============================================================================

from config.settings import DATA_HARD_RATIO_MERGE_THRESHOLD, IS_CODE_DOMAIN
from src.models.messages import (
    AgentName,
    MessageHeader,
    MessageType,
    QuestionScore,
    RolloutResultPayload,
    RoutedMessage,
)
from src.models.state import EvoState
from src.tools.agent_prompts import (
    DIFFICULTY_TEACHER_ACCEPT_PROMPT,   # leaf 1: 是否接收这批题
    DIFFICULTY_TEACHER_POLICY_PROMPT,   # leaf 3: 数据策略提示
    DIFFICULTY_TEACHER_TARGET_PROMPT,   # leaf 2: 目标难度
    DIFFICULTY_TEACHER_WEIGHTS_PROMPT,  # leaf 4: 难度权重配比
    get_agent_prompt,
)
from src.tools.difficulty_tagger import (
    build_pass_count_histogram,
    derive_threshold_policy_from_histogram,
    pass_rate_to_dynamic_difficulty,  # re-exported for legacy tests/imports; not used for cascade labels
)
from src.tools.llm_decision import clamp_distribution, decide_json_leaf
from src.tools.message_artifacts import write_json_artifact
from src.tools.question_fields import add_processed_question_fields

# 合法的难度标签和数据策略枚举
_DYNAMIC_DIFFICULTIES = {"easy", "medium", "hard", "unknown"}
_DATASET_POLICIES = {"single_shard", "merge_shards", "merge_or_replace"}


# ── _clean_bool ──────────────────────────────────────────────────────────────
# 宽松布尔值清洗：支持 "true"/"yes"/"1"/"accept" → True
# ──────────────────────────────────────────────────────────────────────────────
def _clean_bool(raw: object, fallback: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"true", "yes", "1", "accept"}:
            return True
        if value in {"false", "no", "0", "reject"}:
            return False
    return fallback


# ── _clean_enum ──────────────────────────────────────────────────────────────
# 枚举值清洗：只接受白名单内的值，否则 fallback
# ──────────────────────────────────────────────────────────────────────────────
def _clean_enum(raw: object, allowed: set[str], fallback: str) -> str:
    value = str(raw or "").strip()
    return value if value in allowed else fallback


# ── _decide_difficulty_feedback ──────────────────────────────────────────────
# 难度教师（Difficulty Teacher）—— 生成这批 rollout 题的采样建议。
#
# 这是 rollout 之后的 advisory 层：
#   如果 hard 占比过高 → 记录信号并降低 hard 权重
#   无论如何继续主流程，由 filter 的 quota/window replenishment 处理配额不足
#
# 4 个 leaf LLM 调用顺序：
#   ① accept_batch       — 先判断是否接收（后续依赖它）
#   ② target_difficulty  — 基于 accept 结果决定目标难度
#   ③ dataset_policy     — 基于前两者决定数据策略
#   ④ difficulty_weights — 基于前三者决定难度配比
#
# 核心逻辑：
#   - hard_ratio >= DATA_HARD_RATIO_MERGE_THRESHOLD → advisory data_pressure
#   - 不再因为题目过难整批 reject / 回 teacher retry
# ──────────────────────────────────────────────────────────────────────────────
def _decide_difficulty_feedback(
    state: EvoState,
    difficulty_counts: dict,    # 各难度的题目数量统计
    hard_ratio: float,          # hard 题目占比
    candidate_count: int,       # 候选题目总数
) -> dict:
    hard_signal_threshold = max(0.0, min(1.0, float(DATA_HARD_RATIO_MERGE_THRESHOLD or 0.85)))
    hard_dominated = candidate_count > 0 and hard_ratio >= hard_signal_threshold
    fallback = {
        "accept_batch": True,
        "target_difficulty": "medium" if hard_dominated else state.get("target_bucket", ""),
        "dataset_policy_hint": "merge_or_replace" if hard_dominated else "single_shard",
        "difficulty_weights": {"easy": 0.25, "medium": 0.60, "hard": 0.15}
        if hard_dominated
        else {"easy": 0.15, "medium": 0.70, "hard": 0.15},
        "reason": "hard_ratio advisory signal" if hard_dominated else "fallback difficulty teacher",
    }

    if state.get("replenishment_cycle_active"):
        return {
            **fallback,
            "advisory_accept_batch": fallback["accept_batch"],
            "hard_dominated_signal": hard_dominated,
            "reason": f"{fallback['reason']}; skipped leaf LLM during replenishment",
        }

    trace_id = state.get("trace_id", "")
    round_id = state.get("round_id", 0)
    base_context = {
        "round_id": round_id,
        "difficulty_distribution": difficulty_counts,
        "hard_ratio": hard_ratio,
        "candidate_count": candidate_count,
        "target_difficulty": state.get("target_bucket", ""),
    }
    successes: list[str] = []

    # ── leaf ①: 是否接收这批题 ──
    raw_accept, ok = decide_json_leaf(
        agent_name="difficulty_teacher.accept_batch",
        prompt=get_agent_prompt("difficulty_teacher.accept_batch"),
        context=base_context,
        field_name="accept_batch",
        fallback_value=fallback["accept_batch"],
        trace_id=trace_id,
        round_id=round_id,
    )
    _advisory_accept = _clean_bool(raw_accept, fallback["accept_batch"])
    accept_batch = True
    if ok and _advisory_accept != fallback["accept_batch"]:
        successes.append("accept_batch")

    # ── leaf ②: 目标难度（依赖①的结果） ──
    raw_target, ok = decide_json_leaf(
        agent_name="difficulty_teacher.target_difficulty",
        prompt=get_agent_prompt("difficulty_teacher.target_difficulty"),
        context={**base_context, "accept_batch": accept_batch},
        field_name="target_difficulty",
        fallback_value=fallback["target_difficulty"],
        trace_id=trace_id,
        round_id=round_id,
    )
    target_difficulty = _clean_enum(raw_target, _DYNAMIC_DIFFICULTIES, fallback["target_difficulty"])
    if ok and target_difficulty != fallback["target_difficulty"]:
        successes.append("target_difficulty")

    # ── leaf ③: 数据策略提示（依赖①②） ──
    raw_policy, ok = decide_json_leaf(
        agent_name="difficulty_teacher.dataset_policy_hint",
        prompt=get_agent_prompt("difficulty_teacher.dataset_policy_hint"),
        context={**base_context, "accept_batch": accept_batch, "target_difficulty": target_difficulty},
        field_name="dataset_policy_hint",
        fallback_value=fallback["dataset_policy_hint"],
        trace_id=trace_id,
        round_id=round_id,
    )
    dataset_policy_hint = _clean_enum(raw_policy, _DATASET_POLICIES, fallback["dataset_policy_hint"])
    if ok and dataset_policy_hint != fallback["dataset_policy_hint"]:
        successes.append("dataset_policy_hint")

    # ── leaf ④: 难度权重配比（依赖①②③） ──
    raw_weights, ok = decide_json_leaf(
        agent_name="difficulty_teacher.difficulty_weights",
        prompt=get_agent_prompt("difficulty_teacher.difficulty_weights"),
        context={
            **base_context,
            "accept_batch": accept_batch,
            "target_difficulty": target_difficulty,
            "dataset_policy_hint": dataset_policy_hint,
        },
        field_name="difficulty_weights",
        fallback_value=fallback["difficulty_weights"],
        trace_id=trace_id,
        round_id=round_id,
    )
    difficulty_weights = clamp_distribution(
        raw_weights,
        {"easy", "medium", "hard"},
        fallback["difficulty_weights"],
    )
    if ok and difficulty_weights != fallback["difficulty_weights"]:
        successes.append("difficulty_weights")

    return {
        "accept_batch": accept_batch,
        "advisory_accept_batch": _advisory_accept,
        "target_difficulty": target_difficulty,
        "dataset_policy_hint": dataset_policy_hint,
        "difficulty_weights": difficulty_weights,
        "hard_dominated_signal": hard_dominated,
        "reason": (
            "leaf LLM difficulty decisions: " + ",".join(successes)
            if successes
            else fallback["reason"]
        ),
    }


# ── rollout_aggregator_node ──────────────────────────────────────────────────
# LangGraph 节点入口。N 路 parallel rollout worker 的结果通过 state.rollout_runs
# （operator.add reducer）汇聚到这里。
#
# 处理流程：
#   1. 合并：按 question_id 合并 worker 的单次 cascade 结果
#   2. 标难度：使用 worker 输出的 stage-based dynamic_difficulty
#   3. 难度教师生成 advisory sampling metadata
#   4. 路由：始终交给 filter_post_rollout（继续主流程）
# ──────────────────────────────────────────────────────────────────────────────
def rollout_aggregator_node(state: EvoState) -> dict:
    # ── 1. 合并所有 worker 的推理结果 ──
    all_runs = state.get("rollout_runs", [])

    merged = {}
    for run in all_runs:
        for row in run:
            qid = row.get("question_id", "")
            if qid not in merged:
                merged[qid] = add_processed_question_fields(
                    {
                        "question_id": qid,
                        "question_text": row.get("question_text", ""),
                        "gold_answer": row.get("gold_answer", ""),
                        "evaluation_method": row.get("evaluation_method", "gold"),
                        "needs_judge": bool(row.get("needs_judge", False)),
                        "source_dataset_id": row.get("source_dataset_id"),
                        "source_dataset_row_id": row.get("source_dataset_row_id"),
                        "source_dataset_split": row.get("source_dataset_split"),
                        "source_dataset_subset": row.get("source_dataset_subset"),
                        "source_dataset_requested_split": row.get("source_dataset_requested_split"),
                        "source_dataset_split_names": row.get("source_dataset_split_names") or [],
                        "source_dataset_columns": row.get("source_dataset_columns") or [],
                        "source_dataset_first_row": row.get("source_dataset_first_row") or {},
                        "source_dataset_schema": row.get("source_dataset_schema") or {},
                        "dataset_window_id": row.get("dataset_window_id"),
                        "dataset_window_offset": row.get("dataset_window_offset"),
                        "dataset_window_limit": row.get("dataset_window_limit"),
                        "source_role": row.get("source_role"),
                        "module": row.get("module"),
                        "dynamic_difficulty": row.get("dynamic_difficulty"),
                        "pass_count": row.get("pass_count"),
                        "rollout_count": row.get("rollout_count"),
                        "pass_rate": row.get("pass_rate"),
                        "rollout_stage": row.get("rollout_stage"),
                        "rollout_model_key": row.get("rollout_model_key"),
                        "rollout_config_hash": row.get("rollout_config_hash"),
                        "rollout_judge_version": row.get("rollout_judge_version"),
                        "replay_use_count": row.get("replay_use_count"),
                        "correct_flags": [],
                        "dynamic_difficulty_votes": [],
                    },
                    row,
                )
            merged[qid]["correct_flags"].append(row.get("correct", False))
            merged[qid]["dynamic_difficulty_votes"].append(row.get("dynamic_difficulty", "unknown"))

    # ── 2. 统计每道题的动态难度 ──
    round_id = int(state.get("round_id", 0) or 0)
    raw_pass_counts: list[int] = []
    histogram_pass_counts: list[int] = []
    max_rollout_count = 0
    min_rollout_count: int | None = None
    for item in merged.values():
        rollout_count = len(item["correct_flags"])
        pass_count = sum(1 for flag in item["correct_flags"] if flag)
        raw_pass_counts.append(pass_count)
        max_rollout_count = max(max_rollout_count, rollout_count)
        min_rollout_count = rollout_count if min_rollout_count is None else min(min_rollout_count, rollout_count)

    histogram_rollout_count = max_rollout_count
    mixed_rollout_counts = (
        min_rollout_count is not None
        and max_rollout_count > 0
        and min_rollout_count != max_rollout_count
    )
    if mixed_rollout_counts:
        for item in merged.values():
            rollout_count = len(item["correct_flags"])
            pass_count = sum(1 for flag in item["correct_flags"] if flag)
            normalized_count = round((pass_count / rollout_count) * max_rollout_count) if rollout_count else 0
            histogram_pass_counts.append(normalized_count)
        print(
            "[rollout_aggregator] Mixed rollout counts detected; "
            f"normalizing pass-count histogram to denominator={max_rollout_count} "
            f"from min={min_rollout_count} max={max_rollout_count}"
        )
    else:
        histogram_pass_counts = list(raw_pass_counts)

    rollout_pass_count_histogram = build_pass_count_histogram(histogram_pass_counts, histogram_rollout_count)
    threshold_policy = derive_threshold_policy_from_histogram(
        histogram_pass_counts,
        histogram_rollout_count,
        round_id=round_id,
    )
    if mixed_rollout_counts:
        threshold_policy = {
            **threshold_policy,
            "reason": f"{threshold_policy.get('reason', '')}; normalized mixed rollout counts".strip("; "),
        }
    difficulty_counts = {}
    for item in merged.values():
        rollout_count = len(item["correct_flags"])
        pass_count = sum(1 for flag in item["correct_flags"] if flag)
        if IS_CODE_DOMAIN:
            # Code difficulty is defined purely by test execution results:
            # all pass = easy, partial = medium, all fail = hard, no test = unknown.
            if rollout_count > 0 and pass_count == rollout_count:
                difficulty = "easy"
            elif pass_count > 0:
                difficulty = "medium"
            else:
                difficulty = "hard"
        else:
            difficulty = next(
                (
                    str(value)
                    for value in item.get("dynamic_difficulty_votes", [])
                    if str(value) in _DYNAMIC_DIFFICULTIES and str(value) != "unknown"
                ),
                "unknown",
            )
        item["pass_count"] = pass_count
        item["rollout_count"] = rollout_count
        item["pass_rate"] = pass_count / rollout_count if rollout_count else 0.0
        item["dynamic_difficulty"] = difficulty
        item.pop("dynamic_difficulty_votes", None)
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    # ── 3. 持久化打分结果并组装消息 ──
    trace_id = str(state.get("trace_id", ""))
    scored_questions = [QuestionScore.model_validate(v).model_dump() for v in merged.values()]
    scored_questions_ref = write_json_artifact(
        trace_id=trace_id,
        round_id=round_id,
        producer="rollout_aggregator",
        name="scored_questions",
        data=scored_questions,
    )
    payload = RolloutResultPayload(
        scored_questions=[],
        scored_questions_ref=scored_questions_ref,
    )

    # ── 4. 难度教师质量门 ──
    correct_count = sum(1 for v in merged.values() if all(v["correct_flags"]))
    hard_count = int(difficulty_counts.get("hard", 0) or 0)
    hard_ratio = hard_count / len(merged) if merged else 0.0
    difficulty_feedback = _decide_difficulty_feedback(
        state,
        difficulty_counts,
        hard_ratio,
        len(merged),
    )
    difficulty_feedback.update({
        "difficulty_distribution": difficulty_counts,
        "hard_ratio": hard_ratio,
        "candidate_count": len(merged),
        "threshold_policy": threshold_policy,
        "pass_count_histogram": rollout_pass_count_histogram,
    })
    retry_count = 0
    difficulty_feedback["accept_batch"] = True
    difficulty_feedback["retry_count"] = retry_count
    print(f"[rollout_aggregator] Aggregated {len(merged)} questions "
          f"across {len(all_runs)} rollouts, {correct_count} mastered, "
          f"dynamic_difficulty={difficulty_counts}, hard_ratio={hard_ratio:.3f}")

    receiver = AgentName.FILTER
    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.ROLLOUT_AGGREGATOR,
            receiver=receiver,
            message_type=MessageType.ROLLOUT_RESULT,
        ),
        payload=payload,
    )

    return {
        "pending_message": msg,
        "rollout_difficulty_distribution": difficulty_counts,
        "rollout_hard_ratio": hard_ratio,
        "rollout_pass_count_histogram": rollout_pass_count_histogram,
        "difficulty_threshold_policy": threshold_policy,
        "difficulty_teacher_feedback": difficulty_feedback,
        "difficulty_retry_count": retry_count,
    }
