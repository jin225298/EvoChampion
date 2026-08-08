from collections import Counter
from typing import Any

from config.settings import (
    DIFFICULTY_EARLY_EASY_MIN,   # 前期轮次："easy" 的最小答对次数
    DIFFICULTY_EARLY_MEDIUM_MIN, # 前期轮次："medium" 的最小答对次数
    DIFFICULTY_LATE_EASY_MIN,    # 后期轮次："easy" 的最小答对次数
    DIFFICULTY_LATE_MEDIUM_MIN,  # 后期轮次："medium" 的最小答对次数
    DIFFICULTY_LATE_ROUND,       # 从第几轮开始视为"后期"（使用更宽松的阈值）
    ROLLOUT_MAX_NEW_TOKENS,
)
from src.tools.inference_trace import build_inference_trace_row, write_inference_trace_rows
from src.tools.llm_judge import judge_predictions_with_llm_batch
from src.tools.model_runner import judge_answer, run_model_batch
from src.tools.question_fields import add_processed_question_fields


_THRESHOLD_KEYS = ("easy_min", "medium_min")
ROLLOUT_ANSWER_ONLY_SUFFIX = "\n\nOnly output the final answer in \\boxed{}."
EVALUATION_METHOD_GOLD = "gold"
EVALUATION_METHOD_LLM_JUDGE = "llm_judge"


def _rollout_prompt(question_text: str) -> str:
    return question_text


def _answer_only_rollout_prompt(question_text: str) -> str:
    return f"{question_text}{ROLLOUT_ANSWER_ONLY_SUFFIX}"


def rollout_stage_to_dynamic_difficulty(rollout_stage: str, correct: bool) -> str:
    """Map the cascade rollout outcome to training difficulty."""
    if not correct:
        return "hard"
    if rollout_stage == "answer_only":
        return "easy"
    if rollout_stage == "thinking":
        return "medium"
    return "unknown"


def canonical_difficulty_threshold_policy(
    rollout_count: int,
    round_id: int | None = None,
    source: str = "canonical_config",
) -> dict[str, Any]:
    """Return the fixed config-backed easy/medium thresholds.

    This is the stable policy used for evaluator/canonical labeling. Training
    may supply a rollout-derived policy, but evaluation should keep this fixed
    so per-difficulty metrics remain comparable across rounds.
    """

    count = max(1, int(rollout_count or 1))
    if round_id is not None and round_id >= DIFFICULTY_LATE_ROUND:
        easy_min = min(count, int(DIFFICULTY_LATE_EASY_MIN or count))
        medium_min = min(count, int(DIFFICULTY_LATE_MEDIUM_MIN or 1))
    else:
        easy_min = min(count, int(DIFFICULTY_EARLY_EASY_MIN or count))
        medium_min = min(count, int(DIFFICULTY_EARLY_MEDIUM_MIN or 1))
    medium_min = min(max(1, medium_min), count)
    easy_min = min(max(medium_min + 1, easy_min), count) if count > 1 else 1
    if count == 1:
        medium_min = 1
        easy_min = 1
    return {
        "version": 1,
        "source": source,
        "rollout_count": count,
        "easy_min": easy_min,
        "medium_min": medium_min,
        "reason": "fixed config thresholds",
    }


def sanitize_difficulty_threshold_policy(
    policy: dict[str, Any] | None,
    rollout_count: int,
    round_id: int | None = None,
) -> dict[str, Any]:
    """Clamp a threshold policy to valid rollout-count bounds."""

    count = max(1, int(rollout_count or 1))
    fallback = canonical_difficulty_threshold_policy(count, round_id=round_id)
    if not isinstance(policy, dict):
        return fallback
    try:
        easy_min = int(policy.get("easy_min", fallback["easy_min"]))
        medium_min = int(policy.get("medium_min", fallback["medium_min"]))
    except (TypeError, ValueError):
        return fallback
    medium_min = min(max(1, medium_min), count)
    if count <= 1:
        easy_min = 1
        medium_min = 1
    else:
        easy_min = min(max(medium_min + 1, easy_min), count)
        if medium_min >= easy_min:
            medium_min = max(1, easy_min - 1)
    return {
        "version": int(policy.get("version", 1) or 1),
        "source": str(policy.get("source") or fallback["source"]),
        "rollout_count": count,
        "easy_min": easy_min,
        "medium_min": medium_min,
        "reason": str(policy.get("reason") or fallback["reason"]),
    }


def build_pass_count_histogram(pass_counts: list[int], rollout_count: int) -> dict[str, int]:
    """Build an auditable 0/N..N/N pass-count histogram."""

    count = max(0, int(rollout_count or 0))
    histogram = {f"{idx}/{count}": 0 for idx in range(count + 1)}
    for raw in pass_counts:
        try:
            passed = int(raw)
        except (TypeError, ValueError):
            passed = 0
        passed = min(max(0, passed), count)
        key = f"{passed}/{count}"
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def derive_threshold_policy_from_histogram(
    pass_counts: list[int],
    rollout_count: int,
    round_id: int | None = None,
) -> dict[str, Any]:
    """Derive conservative training thresholds from rollout outcomes.

    The policy keeps the public easy/medium/hard contract while allowing the
    medium boundary to relax when many examples are near the model's capability
    frontier. It never lets an LLM choose raw thresholds and always clamps to
    1 <= medium_min < easy_min <= rollout_count when possible.
    """

    count = max(1, int(rollout_count or 1))
    canonical = canonical_difficulty_threshold_policy(count, round_id=round_id)
    if not pass_counts:
        return {**canonical, "source": "rollout_histogram_empty", "reason": "no rollout pass counts"}

    cleaned = [min(max(0, int(value)), count) for value in pass_counts]
    total = len(cleaned)
    impossible_ratio = sum(1 for value in cleaned if value <= 1) / total
    borderline_ratio = sum(1 for value in cleaned if 2 <= value < canonical["easy_min"]) / total

    easy_min = canonical["easy_min"]
    medium_min = canonical["medium_min"]
    reason = "canonical thresholds retained"
    source = "rollout_histogram"

    if count > 1 and borderline_ratio >= 0.25 and impossible_ratio < 0.70:
        medium_min = max(1, min(medium_min, 2))
        reason = "relaxed medium boundary for borderline rollout pass counts"
    elif count > 1 and impossible_ratio >= 0.70:
        medium_min = canonical["medium_min"]
        reason = "kept medium boundary because most items remain near-zero pass"

    return sanitize_difficulty_threshold_policy(
        {
            "version": 1,
            "source": source,
            "rollout_count": count,
            "easy_min": easy_min,
            "medium_min": medium_min,
            "reason": reason,
        },
        rollout_count=count,
        round_id=round_id,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# pass_rate_to_dynamic_difficulty
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 功能：根据"答对次数"和"总尝试次数"将题目标注为 easy / medium / hard
#
# 设计思路：
#   - 前期（round_id < DIFFICULTY_LATE_ROUND）：阈值较严格，
#     即需要相对更高的答对比例才会被判定为 easy，目的是在 system 能力
#     较弱时，把更多题目识别为 hard/medium，避免过早给模型"简单"题目。
#   - 后期（round_id >= DIFFICULTY_LATE_ROUND）：阈值较宽松，
#     即即使答对次数偏少也可能被算作 easy，目的是在系统能力上升后，
#     让模型接触更多"已经会了"的题目，减少低效训练。
#   - 这样做的目的：让难度标签能随系统进化动态调整，
#     避免在训练后期还在"死磕"已经掌握的题目。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pass_rate_to_dynamic_difficulty(
    pass_count: int,           # 答对次数
    rollout_count: int,        # 总尝试次数（rollout 的次数）
    round_id: int | None = None, # 当前训练轮次，None 表示使用前期阈值
    threshold_policy: dict[str, Any] | None = None,
) -> str:
    # 没有有效尝试，无法判断难度
    if rollout_count <= 0:
        return "unknown"

    policy = sanitize_difficulty_threshold_policy(
        threshold_policy,
        rollout_count=rollout_count,
        round_id=round_id,
    ) if threshold_policy is not None else canonical_difficulty_threshold_policy(
        rollout_count,
        round_id=round_id,
    )
    easy_min = int(policy["easy_min"])
    medium_min = int(policy["medium_min"])

    # 注意：min(rollout_count, threshold) 的作用——
    # 当 rollout 次数少于阈值时，阈值降为 rollout_count，即必须全部答对才能算 easy。
    # 这避免了"阈值比尝试次数还大"的无意义比较。

    if pass_count >= easy_min:
        return "easy"
    if pass_count >= medium_min:
        return "medium"
    return "hard"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _field — 通用字段提取器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 功能：从 dict 或对象中提取字段值，按优先级尝试多个可能的 key 名。
# 例如：_field(q, "question_text", "input", default="")
#       先尝试取 q["question_text"]，没有则取 q["input"]，都没有返回 ""
# 这样设计的原因是：数据来源不同（文件、API、dict、dataclass），字段名可能不一致。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _field(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, dict):
            value = item.get(name)
        else:
            value = getattr(item, name, None)
        if value is not None:
            return value
    return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _reference_solution(item: Any) -> str:
    return str(_field(item, "train_output", "process", "think", default="") or "").strip()


def _gold_answer_for_item(item: Any) -> str:
    return str(_field(item, "rollout_gold_answer", "gold_answer", "output", default="") or "").strip()


def _evaluation_method_for_item(item: Any, gold_answer: str, reference_solution: str) -> str:
    raw = str(_field(item, "evaluation_method", "judge_mode", default="") or "").strip().lower()
    if raw == EVALUATION_METHOD_LLM_JUDGE or _truthy(_field(item, "needs_judge", default=False)):
        return EVALUATION_METHOD_LLM_JUDGE
    if gold_answer:
        return EVALUATION_METHOD_GOLD
    if reference_solution:
        return EVALUATION_METHOD_LLM_JUDGE
    return EVALUATION_METHOD_GOLD


def _judge_predictions_batch(
    *,
    predictions: list[str],
    question_texts: list[str],
    gold_answers: list[str],
    reference_solutions: list[str],
    evaluation_methods: list[str],
    trace_id: str,
    round_id: int | None,
) -> list[dict[str, Any]]:
    judgements: list[dict[str, Any]] = [
        {
            "correct": False,
            "score": 0.0,
            "reason": "not judged",
            "source": "not_judged",
        }
        for _ in predictions
    ]
    llm_indices: list[int] = []
    for idx, prediction in enumerate(predictions):
        method = evaluation_methods[idx] if idx < len(evaluation_methods) else EVALUATION_METHOD_GOLD
        gold_answer = gold_answers[idx] if idx < len(gold_answers) else ""
        if method == EVALUATION_METHOD_LLM_JUDGE:
            llm_indices.append(idx)
            continue
        correct = judge_answer(prediction, gold_answer)
        judgements[idx] = {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "reason": "symbolic answer match",
            "source": "gold",
        }
    if llm_indices:
        llm_judgements = judge_predictions_with_llm_batch(
            predictions=[predictions[idx] for idx in llm_indices],
            question_texts=[question_texts[idx] if idx < len(question_texts) else "" for idx in llm_indices],
            gold_answers=[gold_answers[idx] if idx < len(gold_answers) else "" for idx in llm_indices],
            reference_solutions=[
                reference_solutions[idx] if idx < len(reference_solutions) else ""
                for idx in llm_indices
            ],
            state={"trace_id": trace_id, "round_id": int(round_id or 0)},
        )
        for original_idx, judgement in zip(llm_indices, llm_judgements, strict=False):
            judgements[original_idx] = judgement
    return judgements


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# tag_questions_by_pass_rate — 核心入口：两阶段 rollout + 难度标注
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 功能：用指定模型对一批题目进行单次两阶段 rollout：
#       先不开思考，失败时再开思考。根据成功阶段直接判断动态难度。
#
# 工作流程：
#   1. 对每道题先跑一次 answer-only
#   2. answer-only 不正确时再跑一次 thinking
#   3. answer-only 正确标 easy；thinking 正确标 medium；仍错误标 hard
#   4. 返回：标注后的题目列表、本次 cascade 记录、难度分布统计
#
# 设计要点：
#   - 批量推理（run_model_batch）复用 vLLM 引擎，避免每次单独加载模型
#   - rollout_count 参数保留为兼容旧调用，但难度评估始终只做一次 cascade
#   - rollout_rows 记录本次 cascade 的细节，用于后续分析和审计
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tag_questions_by_pass_rate(
    questions: list[Any],       # 待标注的题目列表（dict 或 dataclass）
    model_path: str,            # 模型路径，用于 run_model_batch 加载模型
    rollout_count: int,         # 兼容旧调用；当前难度评估固定为单次 cascade
    temperature: float,         # 推理温度参数
    top_p: float,               # 推理 top_p 参数
    max_new_tokens: int | None = None, # 单次推理最大生成 token 数
    round_id: int | None = None,      # 当前训练轮次，传递给难度判断函数
    rollout_start_idx: int = 0,       # 并行 worker 的全局 rollout 起始编号
    threshold_policy: dict[str, Any] | None = None,
    trace_id: str = "",
    trace_stage: str = "rollout",
    model_role: str = "reference",
    rollout_config_hash: str = "",
    rollout_judge_version: str = "",
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Tag questions with stage-based dynamic difficulty using batched model inference.

    This is the shared mechanism for first-step data difficulty labeling and
    probe difficulty labeling. When USE_VLLM=1, run_model_batch reuses the
    existing vLLM backend and engine cache.
    """

    _ = rollout_count
    repeat_count = 1

    # 从题目中提取 prompt 文本（尝试 question_text / input 两个可能的字段名）
    original_prompts = [_rollout_prompt(str(_field(q, "question_text", "input", default=""))) for q in questions]
    answer_only_prompts = [_answer_only_rollout_prompt(prompt) for prompt in original_prompts]
    # 从题目中提取判题配置：有真实 gold 走符号判题；无 gold 但有过程/证明时走 LLM judge。
    gold_answers = [_gold_answer_for_item(q) for q in questions]
    reference_solutions = [_reference_solution(q) for q in questions]
    evaluation_methods = [
        _evaluation_method_for_item(q, gold_answers[idx], reference_solutions[idx])
        for idx, q in enumerate(questions)
    ]
    needs_judge_flags = [method == EVALUATION_METHOD_LLM_JUDGE for method in evaluation_methods]

    # 每道题的答对计数，初始化为 0
    pass_counts = [0] * len(questions)
    per_question_difficulties = ["unknown"] * len(questions)
    # 记录每一次 rollout 的详细信息（用于日志、审计、分析）
    rollout_rows: list[dict] = []
    trace_rows: list[dict] = []
    effective_max_new_tokens = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else int(ROLLOUT_MAX_NEW_TOKENS)
    )

    # ── 单次级联：先 answer-only，失败时再 thinking ──
    start_idx = max(0, int(rollout_start_idx or 0))
    for local_rollout_idx in range(repeat_count):
        _ = local_rollout_idx
        # 第一阶段：短答粗筛。短答答错的题才进入 thinking 二次判断。
        answer_only_predictions = run_model_batch(
            model_path=model_path,
            prompts=answer_only_prompts,
            max_new_tokens=effective_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            disable_thinking=True,
        )
        predictions = list(answer_only_predictions)
        rollout_stages = ["answer_only"] * len(questions)
        answer_only_judgements = _judge_predictions_batch(
            predictions=answer_only_predictions,
            question_texts=original_prompts,
            gold_answers=gold_answers,
            reference_solutions=reference_solutions,
            evaluation_methods=evaluation_methods,
            trace_id=trace_id,
            round_id=round_id,
        )
        thinking_indices = [
            idx
            for idx, judgement in enumerate(answer_only_judgements)
            if not bool(judgement.get("correct"))
        ]
        if thinking_indices:
            thinking_predictions = run_model_batch(
                model_path=model_path,
                prompts=[original_prompts[idx] for idx in thinking_indices],
                max_new_tokens=effective_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                disable_thinking=False,
            )
            for local_idx, pred in zip(thinking_indices, thinking_predictions, strict=False):
                predictions[local_idx] = pred
                rollout_stages[local_idx] = "thinking"
            thinking_judgements = _judge_predictions_batch(
                predictions=thinking_predictions,
                question_texts=[original_prompts[idx] for idx in thinking_indices],
                gold_answers=[gold_answers[idx] for idx in thinking_indices],
                reference_solutions=[reference_solutions[idx] for idx in thinking_indices],
                evaluation_methods=[evaluation_methods[idx] for idx in thinking_indices],
                trace_id=trace_id,
                round_id=round_id,
            )
            final_thinking_judgements = {
                original_idx: judgement
                for original_idx, judgement in zip(thinking_indices, thinking_judgements, strict=False)
            }
        else:
            final_thinking_judgements = {}

        # 逐题判断预测是否正确
        for idx, pred in enumerate(predictions):
            rollout_idx = start_idx + idx
            if rollout_stages[idx] == "answer_only":
                judgement = answer_only_judgements[idx]
            else:
                judgement = final_thinking_judgements.get(idx, answer_only_judgements[idx])
            correct = bool(judgement.get("correct"))
            if correct:
                pass_counts[idx] += 1  # 答对，累加
            dynamic_difficulty = rollout_stage_to_dynamic_difficulty(rollout_stages[idx], correct)
            per_question_difficulties[idx] = dynamic_difficulty

            # 记录本次 rollout 的详细信息
            rollout_row = {
                "question_id": _field(questions[idx], "question_id", default=""),
                "question_text": original_prompts[idx],
                "gold_answer": gold_answers[idx],
                "reference_solution": reference_solutions[idx],
                "evaluation_method": evaluation_methods[idx],
                "needs_judge": needs_judge_flags[idx],
                "source_dataset_id": _field(questions[idx], "source_dataset_id"),
                "source_dataset_row_id": _field(questions[idx], "source_dataset_row_id"),
                "source_dataset_split": _field(questions[idx], "source_dataset_split"),
                "source_dataset_subset": _field(questions[idx], "source_dataset_subset"),
                "source_dataset_requested_split": _field(questions[idx], "source_dataset_requested_split"),
                "source_dataset_split_names": _field(questions[idx], "source_dataset_split_names", default=[]),
                "source_dataset_columns": _field(questions[idx], "source_dataset_columns", default=[]),
                "source_dataset_first_row": _field(questions[idx], "source_dataset_first_row", default={}),
                "source_dataset_schema": _field(questions[idx], "source_dataset_schema", default={}),
                "dataset_window_id": _field(questions[idx], "dataset_window_id"),
                "dataset_window_offset": _field(questions[idx], "dataset_window_offset"),
                "dataset_window_limit": _field(questions[idx], "dataset_window_limit"),
                "source_role": _field(questions[idx], "source_role"),
                "module": _field(questions[idx], "module"),
                "category": _field(questions[idx], "category"),
                "replay_use_count": _field(questions[idx], "replay_use_count"),
                "correct": correct,
                "judge_score": judgement.get("score"),
                "judge_reason": judgement.get("reason"),
                "judge_source": judgement.get("source"),
                "judge_raw_text": judgement.get("judge_raw_text", ""),
                "judge_fallback_used": bool(judgement.get("fallback_used", False)),
                "judge_schema_errors": list(judgement.get("schema_errors") or []),
                "rollout_idx": rollout_idx,
                "rollout_stage": rollout_stages[idx],
                "rollout_model_key": model_path,
                "rollout_config_hash": rollout_config_hash,
                "rollout_judge_version": rollout_judge_version,
                "dynamic_difficulty": dynamic_difficulty,
            }
            add_processed_question_fields(rollout_row, questions[idx])
            rollout_rows.append(rollout_row)
            if trace_id:
                trace_rows.append(build_inference_trace_row(
                    trace_id=trace_id,
                    round_id=int(round_id or 0),
                    stage=trace_stage,
                    model_role=model_role,
                    model_path=model_path,
                    question_id=str(_field(questions[idx], "question_id", default="")),
                    prompt=original_prompts[idx],
                    gold_answer=gold_answers[idx],
                    prediction=pred,
                    correct=correct,
                    max_new_tokens=effective_max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    rollout_idx=rollout_idx,
                    split_role=str(_field(questions[idx], "split_role", "source_role", default="") or ""),
                    module=str(_field(questions[idx], "module", default="") or ""),
                    dynamic_difficulty=dynamic_difficulty,
                    metadata={
                        "source_dataset_id": _field(questions[idx], "source_dataset_id"),
                        "source_dataset_row_id": _field(questions[idx], "source_dataset_row_id"),
                        "dataset_window_id": _field(questions[idx], "dataset_window_id"),
                        "rollout_stage": rollout_stages[idx],
                        "evaluation_method": evaluation_methods[idx],
                        "needs_judge": needs_judge_flags[idx],
                        "judge_score": judgement.get("score"),
                        "judge_reason": judgement.get("reason"),
                        "judge_source": judgement.get("source"),
                        "judge_raw_text": judgement.get("judge_raw_text", ""),
                        "fallback_used": bool(judgement.get("fallback_used", False)),
                        "schema_errors": list(judgement.get("schema_errors") or []),
                        "judge_fallback_used": bool(judgement.get("fallback_used", False)),
                        "judge_schema_errors": list(judgement.get("schema_errors") or []),
                        "reference_solution": reference_solutions[idx],
                    },
                ))

    # ── 标注阶段：给每道题打上难度标签 ──
    tagged: list[dict] = []
    for idx, question in enumerate(questions):
        # 统一转为 dict，方便后续添加字段
        item = dict(question) if isinstance(question, dict) else question.model_dump()
        pass_count = pass_counts[idx]
        difficulty = per_question_difficulties[idx]

        # 标注信息写入题目
        item["pass_count"] = pass_count
        item["rollout_count"] = repeat_count
        item["pass_rate"] = pass_count / repeat_count
        item["dynamic_difficulty"] = difficulty
        item["evaluation_method"] = evaluation_methods[idx]
        item["needs_judge"] = needs_judge_flags[idx]
        item["rollout_model_key"] = model_path
        item["rollout_config_hash"] = rollout_config_hash
        item["rollout_judge_version"] = rollout_judge_version
        tagged.append(item)

    # 统计各难度级别的题目数量（如 {"easy": 30, "medium": 20, "hard": 10}）
    difficulty_counts = dict(Counter(q["dynamic_difficulty"] for q in tagged))
    if trace_rows:
        write_inference_trace_rows(
            trace_id=trace_id,
            round_id=int(round_id or 0),
            stage=trace_stage,
            rows=trace_rows,
        )

    return tagged, rollout_rows, difficulty_counts
