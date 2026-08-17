"""
评估器模块 (Evaluator) - 自进化训练系统的质检门。

============================================================
模块职责
============================================================
本模块实现了训练循环中的「评估节点」(evaluator_node)，是 LangGraph
控制流中的关键环节。它在每轮训练完成后执行，负责：

1. **多维度对比评估**：分别在测试集(test)、互补测试集(cotest)、
   固定探针集(frozen probe)上评估冠军模型 vs 候选模型的表现。
2. **灾难性遗忘检测**：通过对比训练前后在旧能力(old_ability)题目
   上的准确率变化，判断候选模型是否丢失了旧知识。
3. **新能力增益评估**：判断候选模型在新能力(new_ability)题目上
   是否比冠军模型有显著提升。
4. **门控决策**：综合旧能力保持、新能力提升、探针准确率三个维度，
   产出是否晋升候选模型、是否终止训练循环的决策。
5. **多次采样(multi-rollout)稳健性评估**：对 probe set 做多次采样，
   计算 pass@1、pass@2/3、pass@3/3 统计，减少单次采样偶然性。

============================================================
评估流程（evaluator_node）
============================================================
训练完成 → 校验模型可加载 → 冠军+候选分别在
{test, cotest, frozen} 上批量推理 → 按 split_role 拆分
old_ability/new_ability 计算遗忘/增益 → 多轮采样评估 →
调用 decide_evaluation_gates 做门控决策 → 发送 EvalResultPayload
给 StrategyInspector 做最终晋升/剪枝/回退决策

============================================================
辅助函数
============================================================
- _load_probe_items / _load_alpaca_eval_items: 从 JSON/JSONL 加载评估数据
- _load_mastered_eval_items: 加载已掌握题目集
- _load_old_eval_items: 加载旧能力评估数据(holdout+mastered)
- tag_probe_dynamic_difficulty: 用参考模型标注探针题目难度(easy/medium/hard)
- evaluate_old_mastered_set: 评估候选模型在已掌握题目上的回归
- evaluate_new_skill_gain: 对比冠军 vs 候选在新题上的准确率
- evaluate_probe_set / evaluate_probe_set_detailed: 探针集评估
- _llm_judge_score: LLM 裁判评分(步骤+结果联合评分)
- _accuracy_from_predictions: 从推理结果计算准确率
- _probe_stats_from_predictions: 从推理结果计算探针统计(含失败样例)
- _run_frozen_multi_rollout: frozen probe 多次采样评估
- _save_test_accuracy: 持久化测试准确率明细
- _persist_teacher_decision: 持久化 teacher 决策参数
"""

import hashlib
import json
import time
from pathlib import Path

from config.settings import (
    EVAL_MAX_NEW_TOKENS,
    EVAL_COTEST_MAX_ITEMS,
    EVAL_MASTERED_MAX_ITEMS,
    EVAL_TEST_MAX_ITEMS,
    EXTERNAL_PROBE_EVAL_MAX_ITEMS,
    FROZEN_PROBE_EVAL_METHOD,
    HOLDOUT_EVAL_SIZE,
    PROBE_DIFFICULTY_ROLLOUTS,
    PROBE_EVAL_MAX_ITEMS,
    ROLLOUT_TEMPERATURE,
    ROLLOUT_TOP_P,
    TEST_FORGETTING_TOLERANCE,
    TEST_ROLLOUT_TIMES,
    USE_LLM_AS_JUDGE,
    get_session_dir,
)
from src.models.messages import (
    AgentName,
    EvalResultPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
    TrainResultPayload,
)
from src.models.state import EvoState
from src.tools.model_runner import judge_answer, run_model_batch
from src.tools.code_execution import (
    EVALUATION_METHOD_CODE,
    extract_code_test_fields,
    judge_code_candidate,
    judge_predictions_code_batch,
)
from src.tools.difficulty_tagger import tag_questions_by_pass_rate
from src.tools.agent_prompts import EVALUATOR_JUDGE_PROMPT
from src.tools.data_pipeline.log_parser import parse_structured_training_logs, parse_training_log
from src.tools.inference_trace import build_inference_trace_row, write_inference_trace_rows
from src.tools.llm_decision import decide_json, prompt_for_agent
from src.tools.llm_judge import is_correct_by_score as _shared_is_correct_by_score
from src.tools.llm_judge import judge_predictions_with_llm_batch
from src.tools.llm_judge import llm_judge_score
from src.tools.mathbench_probe import MathBenchProbeResult, mathbench_probe_enabled, run_mathbench_probe
from src.tools.strategy_policy import decide_evaluation_gates


def _run_eval_batch(
    model_path: str,
    prompts: list[str],
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> list[str]:
    return run_model_batch(
        model_path,
        prompts,
        max_new_tokens=int(EVAL_MAX_NEW_TOKENS),
        temperature=temperature,
        top_p=top_p,
        disable_thinking=True,
    )


def _eval_prompt_from_item(item: dict) -> str:
    """Return the non-empty prompt text from supported evaluation record shapes."""
    for key in ("input", "question_text", "instruction", "prompt"):
        value = item.get(key)
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    return ""


def _training_summary_from_result(train_result: TrainResultPayload) -> dict:
    if train_result.train_log_path and Path(train_result.train_log_path).exists():
        parsed_log = parse_training_log(train_result.train_log_path)
        summary = parsed_log.get("training_summary")
        if isinstance(summary, dict):
            if not train_result.success and summary.get("status") in {None, "unknown", "success"}:
                summary = {
                    **summary,
                    "status": "failed",
                    "failure_kind": "trainer_error" if train_result.error_message else "unknown_failure",
                    "failure_reason": train_result.error_message or "training failed",
                }
            return summary

    parsed = parse_structured_training_logs(
        trainer_log_jsonl_path=train_result.trainer_log_jsonl_path,
        training_loss_jsonl_path=train_result.training_loss_jsonl_path,
        all_results_path=train_result.all_results_path,
        trainer_state_path=train_result.trainer_state_path,
        train_results_path=train_result.train_results_path,
        log_text=train_result.error_message,
        exit_code=None,
    )
    summary = parsed.get("training_summary")
    if not isinstance(summary, dict):
        summary = {"available": False}
    if not train_result.success:
        summary = {
            **summary,
            "status": "failed",
            "failure_kind": summary.get("failure_kind") or "trainer_error",
            "failure_reason": summary.get("failure_reason") or train_result.error_message or "training failed",
        }
    return summary


def _probe_cache_key(model_path: str, probe_set_path: str, prompts: list[str], gold: list[str]) -> str:
    """生成探针难度标注的缓存键。

    将模型路径、探针集路径、采样参数、题目(prompt+gold)的 SHA1 哈希组合后计算全局 SHA1，
    确保相同模型+相同题目+相同采样参数下不重复计算动态难度。
    题目内容不直接参与哈希（太长），改用 prompt 的 SHA1 + gold 的组合。
    """
    payload = {
        "model_path": model_path,
        "probe_set_path": probe_set_path,
        "rollouts": PROBE_DIFFICULTY_ROLLOUTS,
        "temperature": ROLLOUT_TEMPERATURE,
        "top_p": ROLLOUT_TOP_P,
        "items": [
            {"prompt_hash": hashlib.sha1(p.encode("utf-8")).hexdigest(), "gold": g}
            for p, g in zip(prompts, gold)
        ],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_probe_items(
    probe_set_path: str,
    max_items: int | None = None,
) -> tuple[list[str], list[str], list[dict]]:
    """从 JSON 或 JSONL 文件加载探针数据。

    支持两种格式：
    - JSON: {"category": [{"question_text": ..., "gold_answer": ...}, ...]}
    - JSONL: 每行一个 {"input"/"question_text": ..., "output"/"gold_answer": ...}

    Returns:
        (prompts, gold_answers, raw_items) 三元组。数量受 max_items 或全局 PROBE_EVAL_MAX_ITEMS 限制。
    """
    if not probe_set_path:
        return [], [], []

    path = Path(probe_set_path)
    if not path.exists() or path.is_dir():
        return [], [], []
    if path.suffix == ".jsonl":
        probe_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    probe_data.append(item)
    else:
        with open(path, "r", encoding="utf-8") as f:
            probe_data = json.load(f)

    all_prompts = []
    all_gold = []
    all_questions: list[dict] = []

    if isinstance(probe_data, dict):
        for _category, questions in probe_data.items():
            for q in questions:
                prompt = _eval_prompt_from_item(q)
                gold = q.get("rollout_gold_answer") or q.get("gold_answer") or q.get("output", "")
                all_prompts.append(prompt)
                all_gold.append(gold)
                all_questions.append(q)
    elif isinstance(probe_data, list):
        for item in probe_data:
            prompt = _eval_prompt_from_item(item)
            gold = item.get("rollout_gold_answer") or item.get("gold_answer") or item.get("output", "")
            all_prompts.append(prompt)
            all_gold.append(gold)
            all_questions.append(item)

    limit = PROBE_EVAL_MAX_ITEMS if max_items is None else max_items
    if limit and limit > 0:
        all_prompts = all_prompts[:limit]
        all_gold = all_gold[:limit]
        all_questions = all_questions[:limit]

    return all_prompts, all_gold, all_questions


def _load_frozen_probe_items(probe_set_path: str) -> tuple[list[str], list[str], list[dict]]:
    """Load the fixed frozen probe without applying the dynamic probe eval cap."""
    if mathbench_probe_enabled(FROZEN_PROBE_EVAL_METHOD):
        return [], [], []
    return _load_probe_items(probe_set_path, max_items=0)


def _load_external_probe_items(probe_set_path: str) -> tuple[list[str], list[str], list[dict]]:
    """加载外部探针数据，使用专门的 EXTERNAL_PROBE_EVAL_MAX_ITEMS 上限。"""
    return _load_probe_items(probe_set_path, max_items=EXTERNAL_PROBE_EVAL_MAX_ITEMS)


def _mathbench_probe_metadata(result: MathBenchProbeResult) -> dict:
    return {
        "score": result.score,
        "metric_name": result.metric_name,
        "summary_path": str(result.summary_path),
        "work_dir": str(result.work_dir),
        "raw_metrics": result.raw_metrics,
    }


def tag_probe_dynamic_difficulty(
    reference_model_path: str,
    probe_set_path: str,
    prompts: list[str],
    gold_answers: list[str],
    questions: list[dict],
    trace_id: str,
    round_id: int = 0,
) -> list[str]:
    """用参考模型标注探针题目的动态难度(easy/medium/hard)。

    工作流程：
    1. 若 questions 中已有 dynamic_difficulty 字段且全是有效值，直接返回。
    2. 检查磁盘缓存(session_dir/probe_difficulty_cache/)，
       相同模型+题目+参数下复用上次标注结果，避免重复推理。
    3. 缓存未命中时，用参考模型对每道题做 k=PROBE_DIFFICULTY_ROLLOUTS 次采样，
       按 pass_rate 分档：高通过率→easy，中→medium，低→hard。
    4. 结果写回磁盘缓存。
    """
    existing = [
        str(q.get("dynamic_difficulty") or "")
        for q in questions
    ]
    if existing and all(item in {"easy", "medium", "hard"} for item in existing):
        return existing

    if not prompts or not reference_model_path:
        return ["unknown"] * len(prompts)

    session_dir = get_session_dir(trace_id)
    cache_dir = session_dir / "probe_difficulty_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_probe_cache_key(reference_model_path, probe_set_path, prompts, gold_answers)}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            difficulties = cached.get("difficulties", [])
            if len(difficulties) == len(prompts):
                return [str(item) for item in difficulties]
        except (json.JSONDecodeError, OSError):
            pass

    rollout_count = max(1, PROBE_DIFFICULTY_ROLLOUTS)
    probe_questions = []
    for idx, question in enumerate(questions):
        item = dict(question)
        item["question_text"] = prompts[idx]
        item["gold_answer"] = gold_answers[idx]
        probe_questions.append(item)
    tagged, _rollout_rows, difficulty_distribution = tag_questions_by_pass_rate(
        questions=probe_questions,
        model_path=reference_model_path,
        rollout_count=rollout_count,
        temperature=ROLLOUT_TEMPERATURE,
        top_p=ROLLOUT_TOP_P,
        round_id=round_id,
        trace_id=trace_id,
        trace_stage="probe_difficulty_rollout",
        model_role="reference",
    )
    difficulties = [str(item.get("dynamic_difficulty", "unknown")) for item in tagged]
    pass_counts = [int(item.get("pass_count", 0) or 0) for item in tagged]
    cache_payload = {
        "reference_model_path": reference_model_path,
        "probe_set_path": probe_set_path,
        "rollout_count": rollout_count,
        "temperature": ROLLOUT_TEMPERATURE,
        "top_p": ROLLOUT_TOP_P,
        "difficulty_distribution": difficulty_distribution,
        "difficulties": difficulties,
        "pass_counts": pass_counts,
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[evaluator] Tagged probe dynamic difficulty: "
        f"{cache_payload['difficulty_distribution']} with k={rollout_count}"
    )
    return difficulties


def evaluate_old_mastered_set(candidate_model_path: str, mastered_set_path: str) -> tuple[int, float, int]:
    """评估候选模型在已掌握题目集上的表现（回归检测）。

    用于检测灾难性遗忘：对之前已经掌握的题目，新模型是否还能做对。
    最多评估 EVAL_MASTERED_MAX_ITEMS 道题（默认50），用 judge_answer 做符号匹配。

    Returns:
        (error_count, error_rate, eval_count) — error_rate = 错误数/总题数。
    """
    t0 = time.time()

    if not mastered_set_path or not Path(mastered_set_path).exists():
        return 0, 0.0, 0

    with open(mastered_set_path, "r", encoding="utf-8") as f:
        mastered = json.load(f)

    questions = []
    if isinstance(mastered, dict):
        for v in mastered.values():
            if isinstance(v, list):
                questions.extend(v)
    elif isinstance(mastered, list):
        questions = mastered

    total = len(questions)
    if total == 0:
        return 0, 0.0, 0

    # Cap at 50 questions for efficiency
    eval_questions = questions[:EVAL_MASTERED_MAX_ITEMS]

    # Batch inference
    prompts = [q.get("question_text", "") for q in eval_questions]
    gold_answers = [q.get("rollout_gold_answer") or q.get("gold_answer", "") for q in eval_questions]
    predictions = _run_eval_batch(candidate_model_path, prompts)

    errors = 0
    for pred, gold, item in zip(predictions, gold_answers, eval_questions):
        if not _judge_eval_prediction(pred, gold, item=item, allow_llm_judge=False):
            errors += 1

    error_rate = errors / total
    elapsed = time.time() - t0
    print(f"[evaluator] evaluate_old_mastered_set: {len(eval_questions)}/{total} questions in {elapsed:.2f}s, error_rate={error_rate:.3f}")
    return errors, error_rate, len(eval_questions)


def evaluate_new_skill_gain(
    champion_model_path: str,
    candidate_model_path: str,
    test_path: str,
    max_items: int = 60,
) -> tuple[float, float]:
    """对比冠军模型与候选模型在新技能测试集上的准确率。

    分别用冠军和候选模型对同一批题目做推理，用 judge_answer 判断对错，
    返回 (冠军准确率, 候选准确率)。候选高于冠军即为正向增益。
    最多评估 max_items 道题（默认60）。

    Returns:
        (champion_accuracy, candidate_accuracy)
    """
    t0 = time.time()

    if not Path(test_path).exists():
        return 0.0, 0.0

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    eval_items = test_data[:max_items]
    if not eval_items:
        return 0.0, 0.0

    prompts = [_eval_prompt_from_item(item) for item in eval_items]
    gold_answers = [item.get("rollout_gold_answer") or item.get("gold_answer") or item.get("output", "") for item in eval_items]

    before_correct, before_total = 0, 0
    after_correct, after_total = 0, 0

    # Batch inference for champion
    if champion_model_path:
        before_preds = _run_eval_batch(champion_model_path, prompts)
        for pred, gold, item in zip(before_preds, gold_answers, eval_items):
            if _judge_eval_prediction(pred, gold, item=item, allow_llm_judge=False):
                before_correct += 1
            before_total += 1

    # Batch inference for candidate
    if candidate_model_path:
        after_preds = _run_eval_batch(candidate_model_path, prompts)
        for pred, gold, item in zip(after_preds, gold_answers, eval_items):
            if _judge_eval_prediction(pred, gold, item=item, allow_llm_judge=False):
                after_correct += 1
            after_total += 1

    before_acc = before_correct / before_total if before_total > 0 else 0.0
    after_acc = after_correct / after_total if after_total > 0 else 0.0

    elapsed = time.time() - t0
    print(f"[evaluator] evaluate_new_skill_gain: {len(eval_items)} items, "
          f"champion_acc={before_acc:.3f}, candidate_acc={after_acc:.3f} in {elapsed:.2f}s")
    return before_acc, after_acc


def evaluate_probe_set_detailed(
    candidate_model_path: str,
    probe_set_path: str,
    reference_model_path: str = "",
    trace_id: str = "",
) -> tuple[float, dict[str, float], list[dict]]:
    """评估候选模型在探针集上的表现（详细版，含失败样例）。

    1. 加载探针数据 → 标注动态难度(easy/medium/hard)
    2. 候选模型批量推理
    3. 按难度统计准确率，收集失败样例

    Returns:
        (overall_accuracy, {difficulty: accuracy, ...}, [failed_examples, ...])
        failed_examples 每项包含 question_id, question_text, gold_answer,
        prediction, module, dynamic_difficulty。
    """
    t0 = time.time()

    all_prompts, all_gold, all_questions = _load_probe_items(probe_set_path)

    if not all_prompts:
        return 0.0, {}, []

    question_difficulties = tag_probe_dynamic_difficulty(
        reference_model_path or candidate_model_path,
        probe_set_path,
        all_prompts,
        all_gold,
        all_questions,
        trace_id,
        round_id=0,
    )

    predictions = _run_eval_batch(candidate_model_path, all_prompts)

    correct = 0
    difficulty_stats: dict[str, list[bool]] = {}
    failed_examples: list[dict] = []
    for pred, gold, difficulty, question in zip(
        predictions,
        all_gold,
        question_difficulties,
        all_questions,
    ):
        is_correct = _judge_eval_prediction(pred, gold, item=question, allow_llm_judge=False)
        if is_correct:
            correct += 1
        else:
            failed_examples.append({
                "question_id": question.get("question_id", ""),
                "question_text": question.get("question_text", question.get("input", "")),
                "gold_answer": gold,
                "prediction": pred,
                "module": question.get("module", "unknown"),
                "dynamic_difficulty": difficulty,
            })
        difficulty_stats.setdefault(difficulty, []).append(is_correct)

    accuracy = correct / len(all_prompts) if all_prompts else 0.0

    per_difficulty_acc = {}
    for difficulty_name, results in sorted(difficulty_stats.items()):
        per_difficulty_acc[difficulty_name] = sum(results) / len(results) if results else 0.0

    elapsed = time.time() - t0
    print(f"[evaluator] evaluate_probe_set: {len(all_prompts)} questions in {elapsed:.2f}s, "
          f"accuracy={accuracy:.3f}, "
          f"per_dynamic_difficulty={per_difficulty_acc}")
    return accuracy, per_difficulty_acc, failed_examples


def evaluate_probe_set(
    candidate_model_path: str,
    probe_set_path: str,
    reference_model_path: str = "",
    trace_id: str = "",
) -> tuple[float, dict[str, float]]:
    """评估候选模型在探针集上的表现（简化版，只返回聚合指标）。

    内部调用 evaluate_probe_set_detailed，但丢弃失败样例列表。
    适用于只需要 overall accuracy 和 per_difficulty accuracy 的场景。

    Returns:
        (overall_accuracy, {difficulty: accuracy, ...})
    """
    accuracy, per_difficulty_acc, _ = evaluate_probe_set_detailed(
        candidate_model_path,
        probe_set_path,
        reference_model_path=reference_model_path,
        trace_id=trace_id,
    )
    return accuracy, per_difficulty_acc


def _load_alpaca_eval_items(path: str, max_items: int) -> tuple[list[str], list[str], list[dict]]:
    """从 Alpaca 格式 JSON 文件加载评估题目。

    预期格式：[{ "input"/"question_text"/"instruction": ..., "output"/"gold_answer": ... }, ...]
    返回 (prompts, gold_answers, raw_items)，数量上限为 max_items。
    """
    if not path or not Path(path).exists():
        return [], [], []
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, list):
        return [], [], []
    items = [item for item in loaded[:max_items] if isinstance(item, dict)]
    prompts = [_eval_prompt_from_item(item) for item in items]
    gold = [
        str(item.get("rollout_gold_answer") or item.get("gold_answer") or item.get("output", ""))
        for item in items
    ]
    return prompts, gold, items


def _load_mastered_eval_items(mastered_set_path: str, max_items: int = 50) -> tuple[list[str], list[str], int]:
    """从已掌握题目集 JSON 文件加载评估数据。

    支持两种结构：
    - dict: {"category": [{"question_text": ..., "gold_answer": ...}, ...]}
    - list: [{"question_text": ..., "gold_answer": ...}, ...]

    Returns:
        (prompts, gold_answers, eval_count) — 只取前 max_items 道。
    """
    if not mastered_set_path or not Path(mastered_set_path).exists():
        return [], [], 0
    with open(mastered_set_path, "r", encoding="utf-8") as f:
        mastered = json.load(f)
    questions = []
    if isinstance(mastered, dict):
        for v in mastered.values():
            if isinstance(v, list):
                questions.extend(q for q in v if isinstance(q, dict))
    elif isinstance(mastered, list):
        questions = [q for q in mastered if isinstance(q, dict)]
    eval_questions = questions[:max_items]
    prompts = [str(q.get("question_text", "")) for q in eval_questions]
    gold = [str(q.get("rollout_gold_answer") or q.get("gold_answer", "")) for q in eval_questions]
    return prompts, gold, len(eval_questions)


def _load_old_eval_items(
    holdout_path: str,
    mastered_set_path: str,
    max_items: int = HOLDOUT_EVAL_SIZE,
) -> tuple[list[str], list[str], int, int]:
    """加载旧能力评估数据（holdout + mastered 合并）。

    将 holdout 集和 mastered 集的 prompts/gold 拼接在一起，
    返回合并后的 (prompts, gold, holdout_count, mastered_count)。

    Returns:
        (all_prompts, all_gold, holdout_len, mastered_len)
    """
    holdout_prompts, holdout_gold, _holdout_items = _load_alpaca_eval_items(
        holdout_path,
        max_items=max_items,
    )
    mastered_prompts, mastered_gold, mastered_count = _load_mastered_eval_items(
        mastered_set_path,
        max_items=max_items,
    )
    return (
        holdout_prompts + mastered_prompts,
        holdout_gold + mastered_gold,
        len(holdout_gold),
        mastered_count,
    )


def _llm_judge_score(
    prediction: str,
    gold_answer: str,
    question_text: str = "",
    state: EvoState | None = None,
    reference_solution: str = "",
) -> float:
    """用 LLM 裁判对单次预测进行最终答案二值判定。

    评分逻辑：
    1. 先用 judge_answer(pred, gold) 做符号匹配（快速路径）。
    2. 若未启用 LLM 裁判(USE_LLM_AS_JUDGE=False)，直接返回符号匹配结果。
    3. 若启用，调用 LLM(evaluator_judge) 只比较最终答案是否一致/等价。
    4. LLM 调用失败时 fallback 到符号匹配结果。

    Returns:
        二值分数。1.0 视为正确，0.0 视为错误。
    """
    return llm_judge_score(
        prediction=prediction,
        gold_answer=gold_answer,
        question_text=question_text,
        reference_solution=reference_solution,
        state=dict(state) if isinstance(state, dict) else None,
        use_llm=USE_LLM_AS_JUDGE,
        judge_answer_func=judge_answer,
        decide_json_func=decide_json,
        prompt_for_agent_func=lambda prompt_state, agent_name, _default_prompt: prompt_for_agent(
            prompt_state,
            agent_name,
            EVALUATOR_JUDGE_PROMPT,
        ),
    )


def _is_correct_by_score(score: float) -> bool:
    """LLM 裁判分数 → 布尔判对。只有二值分数 1.0 视为答对。"""
    return _shared_is_correct_by_score(score)


def _truthy_eval_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _reference_solution_for_item(item: dict | None) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("train_output", "reference_solution", "process", "think"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _evaluation_method_for_item(item: dict | None) -> str:
    if not isinstance(item, dict):
        return "gold"
    raw = str(item.get("evaluation_method") or item.get("judge_mode") or "").strip().lower()
    raw = raw.replace("-", "_")
    if raw in {"llm_judge", "llm_as_judge", "llmasjudge"}:
        return "llm_judge"
    if raw == EVALUATION_METHOD_CODE:
        return EVALUATION_METHOD_CODE
    if _truthy_eval_flag(item.get("needs_judge", False)):
        return "llm_judge"
    # Code domain: an item carrying an executable test + entry point is judged
    # by execution (never by LLM subjective equivalence).
    if _item_has_code_test(item):
        return EVALUATION_METHOD_CODE
    return "gold"


def _item_has_code_test(item: dict | None) -> bool:
    if not isinstance(item, dict):
        return False
    from config.settings import use_code_execution_judging

    if not use_code_execution_judging():
        return False
    test_code, entry_point, _gold = extract_code_test_fields(item)
    return bool(test_code and entry_point)


def _judge_eval_prediction(
    prediction: str,
    gold_answer: str,
    *,
    prompt: str = "",
    item: dict | None = None,
    state: EvoState | None = None,
    allow_llm_judge: bool = True,
) -> bool:
    """Judge one eval prediction using the cleaned dataset's judge label."""
    method = _evaluation_method_for_item(item)
    if method == "llm_judge":
        if USE_LLM_AS_JUDGE and allow_llm_judge:
            return _is_correct_by_score(_llm_judge_score(
                prediction,
                gold_answer,
                prompt,
                state,
                reference_solution=_reference_solution_for_item(item),
            ))
        return judge_answer(prediction, gold_answer)
    if method == EVALUATION_METHOD_CODE:
        test_code, entry_point, _gold = extract_code_test_fields(item or {})
        from config.settings import CODE_JUDGE_TIMEOUT_SECONDS, CODE_JUDGE_MEMORY_MB
        result = judge_code_candidate(
            prediction, test_code, entry_point,
            timeout=CODE_JUDGE_TIMEOUT_SECONDS, memory_mb=CODE_JUDGE_MEMORY_MB,
        )
        return bool(result.passed)
    return judge_answer(prediction, gold_answer)


def _judgements_from_predictions(
    predictions: list[str],
    gold_answers: list[str],
    prompts: list[str] | None = None,
    items: list[dict] | None = None,
    state: EvoState | None = None,
) -> list[dict]:
    judgements: list[dict] = [
        {
            "correct": False,
            "score": 0.0,
            "reason": "not judged",
            "source": "not_judged",
            "judge_raw_text": "",
            "fallback_used": False,
            "schema_errors": [],
        }
        for _ in predictions
    ]
    llm_indices: list[int] = []
    code_indices: list[int] = []
    for idx, pred in enumerate(predictions):
        gold = gold_answers[idx] if idx < len(gold_answers) else ""
        prompt = prompts[idx] if prompts and idx < len(prompts) else ""
        item = items[idx] if items and idx < len(items) else {}
        method = _evaluation_method_for_item(item)
        if method == "llm_judge" and USE_LLM_AS_JUDGE:
            llm_indices.append(idx)
            continue
        if method == EVALUATION_METHOD_CODE:
            code_indices.append(idx)
            continue
        correct = _judge_eval_prediction(
            pred,
            gold,
            prompt=prompt,
            item=item,
            state=state,
            allow_llm_judge=False,
        )
        judgements[idx] = {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "reason": "symbolic answer match",
            "source": "gold",
            "judge_raw_text": "",
            "fallback_used": False,
            "schema_errors": [],
        }
    if code_indices:
        from config.settings import CODE_JUDGE_TIMEOUT_SECONDS, CODE_JUDGE_MEMORY_MB, CODE_JUDGE_MAX_WORKERS
        code_judgements = judge_predictions_code_batch(
            predictions=[predictions[idx] for idx in code_indices],
            items=[(items[idx] if items and idx < len(items) else {}) for idx in code_indices],
            timeout=CODE_JUDGE_TIMEOUT_SECONDS,
            memory_mb=CODE_JUDGE_MEMORY_MB,
            max_workers=CODE_JUDGE_MAX_WORKERS,
        )
        for original_idx, judgement in zip(code_indices, code_judgements, strict=False):
            judgements[original_idx] = judgement
    if llm_indices:
        batch_judgements = judge_predictions_with_llm_batch(
            predictions=[predictions[idx] for idx in llm_indices],
            question_texts=[
                prompts[idx] if prompts and idx < len(prompts) else ""
                for idx in llm_indices
            ],
            gold_answers=[
                gold_answers[idx] if idx < len(gold_answers) else ""
                for idx in llm_indices
            ],
            reference_solutions=[
                _reference_solution_for_item(items[idx] if items and idx < len(items) else {})
                for idx in llm_indices
            ],
            state=dict(state) if isinstance(state, dict) else None,
            use_llm=USE_LLM_AS_JUDGE,
            judge_answer_func=judge_answer,
            prompt_for_agent_func=lambda prompt_state, agent_name, _default_prompt: prompt_for_agent(
                prompt_state,
                agent_name,
                EVALUATOR_JUDGE_PROMPT,
            ),
        )
        for original_idx, judgement in zip(llm_indices, batch_judgements, strict=False):
            judgements[original_idx] = judgement
    return judgements


def _correctness_from_predictions(
    predictions: list[str],
    gold_answers: list[str],
    prompts: list[str] | None = None,
    items: list[dict] | None = None,
    state: EvoState | None = None,
) -> list[bool]:
    return [
        bool(judgement.get("correct"))
        for judgement in _judgements_from_predictions(
            predictions,
            gold_answers,
            prompts=prompts,
            items=items,
            state=state,
        )
    ]


def _split_test_by_role(
    test_items: list[dict],
) -> tuple[list[int], list[int]]:
    """按 split_role 字段拆分测试题为「旧能力」和「新能力」两组索引列表。

    用于区分评估维度：
    - "old_ability": 检测灾难性遗忘
    - "new_ability": 检测新技能增益

    Returns:
        (old_indices, new_indices) — 各自由题目索引组成的列表。
    """
    old_indices = [i for i, item in enumerate(test_items) if item.get("split_role") == "old_ability"]
    new_indices = [i for i, item in enumerate(test_items) if item.get("split_role") == "new_ability"]
    return old_indices, new_indices


def _accuracy_from_predictions(
    predictions: list[str],
    gold_answers: list[str],
    prompts: list[str] | None = None,
    items: list[dict] | None = None,
    state: EvoState | None = None,
) -> tuple[int, float]:
    """从批量推理结果计算准确率（correct_count, accuracy）。

    判对逻辑：
    - 题目清洗标签为 llm_judge/needs_judge 时，批量调用 LLM judge 综合评分
    - 否则用 judge_answer 做符号/MathVerify 匹配
    """
    total = len(gold_answers)
    correct = sum(_correctness_from_predictions(
        predictions,
        gold_answers,
        prompts=prompts,
        items=items,
        state=state,
    ))
    return correct, correct / total if total else 0.0


def _probe_stats_from_predictions(
    predictions: list[str],
    gold_answers: list[str],
    difficulties: list[str],
    questions: list[dict],
    state: EvoState | None = None,
) -> tuple[float, dict[str, float], list[dict]]:
    """从探针集批量推理结果计算出完整的探针统计。

    对每道题判对后：
    1. 按 difficulty 维度的正确列表，用于后续分组统计
    2. 收集失败样例（含 question_id, question_text, gold, prediction, module, difficulty）

    判对逻辑同 _accuracy_from_predictions：按题目清洗标签分流到 LLM 裁判或符号裁判。

    Returns:
        (overall_accuracy, {difficulty: acc, ...}, [failed_examples, ...])
    """
    correct = 0
    difficulty_stats: dict[str, list[bool]] = {}
    failed_examples: list[dict] = []
    results = _correctness_from_predictions(
        predictions,
        gold_answers,
        prompts=[
            question.get("question_text", question.get("input", ""))
            for question in questions
        ],
        items=questions,
        state=state,
    )
    for idx, (pred, gold, difficulty, question, is_correct) in enumerate(zip(
        predictions,
        gold_answers,
        difficulties,
        questions,
        results,
    )):
        if is_correct:
            correct += 1
        else:
            failed_examples.append({
                "question_id": question.get("question_id", ""),
                "question_text": question.get("question_text", question.get("input", "")),
                "gold_answer": gold,
                "prediction": pred,
                "module": question.get("module", "unknown"),
                "dynamic_difficulty": difficulty,
            })
        difficulty_stats.setdefault(difficulty, []).append(is_correct)

    accuracy = correct / len(gold_answers) if gold_answers else 0.0
    per_difficulty_acc = {
        difficulty_name: sum(results) / len(results) if results else 0.0
        for difficulty_name, results in sorted(difficulty_stats.items())
    }
    return accuracy, per_difficulty_acc, failed_examples


def _write_eval_trace_rows(
    *,
    trace_id: str,
    round_id: int,
    stage: str,
    model_role: str,
    model_path: str,
    prompts: list[str],
    gold_answers: list[str],
    predictions: list[str],
    items: list[dict],
    split_role: str = "",
    difficulties: list[str] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    metadata: dict | None = None,
    correct_results: list[bool] | None = None,
    judgements: list[dict] | None = None,
    state: EvoState | None = None,
) -> None:
    if not trace_id or not prompts:
        return
    rows = []
    for idx, prompt in enumerate(prompts):
        item = items[idx] if idx < len(items) else {}
        gold = gold_answers[idx] if idx < len(gold_answers) else ""
        pred = predictions[idx] if idx < len(predictions) else ""
        judgement = judgements[idx] if judgements is not None and idx < len(judgements) else {}
        is_correct = (
            bool(judgement.get("correct"))
            if judgement
            else
            correct_results[idx]
            if correct_results is not None and idx < len(correct_results)
            else _judge_eval_prediction(pred, gold, prompt=prompt, item=item, state=state)
        )
        difficulty = (
            difficulties[idx]
            if difficulties is not None and idx < len(difficulties)
            else item.get("dynamic_difficulty", "")
        )
        row_metadata = dict(metadata or {})
        row_metadata.update({
            "evaluation_method": _evaluation_method_for_item(item),
            "needs_judge": _evaluation_method_for_item(item) == "llm_judge",
            "reference_solution_present": bool(_reference_solution_for_item(item)),
        })
        if judgement:
            row_metadata.update({
                "judge_score": judgement.get("score"),
                "judge_reason": judgement.get("reason"),
                "judge_source": judgement.get("source"),
                "judge_raw_text": judgement.get("judge_raw_text", ""),
                "judge_fallback_used": bool(judgement.get("fallback_used", False)),
                "judge_schema_errors": list(judgement.get("schema_errors") or []),
            })
        rows.append(build_inference_trace_row(
            trace_id=trace_id,
            round_id=round_id,
            stage=stage,
            model_role=model_role,
            model_path=model_path,
            question_id=str(item.get("question_id", "")),
            prompt=prompt,
            gold_answer=gold,
            prediction=pred,
            correct=is_correct,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else int(EVAL_MAX_NEW_TOKENS),
            temperature=temperature,
            top_p=top_p,
            split_role=str(item.get("split_role", split_role) or split_role),
            module=str(item.get("module", "")),
            dynamic_difficulty=str(difficulty or ""),
            metadata=row_metadata,
        ))
    write_inference_trace_rows(
        trace_id=trace_id,
        round_id=round_id,
        stage=stage,
        rows=rows,
    )


def evaluator_node(state: EvoState) -> dict:
    """评估节点 — 自进化训练循环的质检门。

    ============================================================
    整体流程
    ============================================================
    1. 从 state 中提取训练结果(candidate_model_path)、冠军模型路径、
       测试集(test/co-test)、固定探针集(frozen probe)路径。
    2. 若训练失败或模型不可加载，走 _training_failure_eval_result 快速失败路径。
    3. 批量推理阶段：
       - 冠军+候选模型在 {test, cotest, frozen} 三组数据上批量推理
       - 两组模型用完全相同的 prompts（确保公平对比）
    4. 多维度评估：
       a) 旧能力保持(old_ability)：训练前后准确率变化，检测灾难性遗忘
       b) 新能力增益(new_ability)：训练前后准确率变化，判断学习效果
       c) co-test 准确率：互补测试集上的表现
       d) frozen probe 准确率：1-shot + 3-shot 多次采样
    5. 门控决策：
       调用 decide_evaluation_gates() 产出 6 个决策信号：
       - pass_old_skill_gate / pass_new_skill_gate / pass_probe_gate
       - should_promote_candidate / should_stop
    6. 持久化：
       - 测试准确率明细 → session_dir/round_{N}_datasets/test_accuracy.json
       - 构建 EvalResultPayload → StrategyInspector 做最终晋升/剪枝/回退

    ============================================================
    关键设计决策
    ============================================================
    - 冠军和候选在完全相同的 prompts 上推理，消除题目差异导致的评估偏差。
    - 旧能力/新能力通过 test_items 的 split_role 字段切分，而非单独加载数据集。
    - frozen probe 既做 1-shot（直接推理）又做 3-shot（多次采样），
      1-shot 用于 gate 判断，3-shot 用于日志观察模型稳健性。
    - judge-only 题目批量进入 LLM judge，避免空 gold 退回符号裁判造成系统性漏判。

    Returns:
        dict 更新到 LangGraph state，包含：
        - metrics_before: 训练前指标
        - metrics_after: 训练后指标
        - pending_message: EvalResultPayload → StrategyInspector
    """
    t_start = time.time()
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    train_result = TrainResultPayload.model_validate(pending_message.payload)
    training_summary = _training_summary_from_result(train_result)
    candidate_model_path = train_result.candidate_model_path
    champion_model_path = state.get("champion_model_path", "")
    test_path = state.get("test_path", "")
    cotest_path = state.get("cotest_path", "")
    frozen_probe_set_path = state.get("probe_frozen_set_path") or state.get("global_probe_set_path", "")
    use_mathbench_probe = mathbench_probe_enabled(FROZEN_PROBE_EVAL_METHOD)
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    session_dir = get_session_dir(trace_id)

    if not train_result.success or not _is_loadable_model_dir(candidate_model_path):
        return _training_failure_eval_result(
            state=state,
            train_result=train_result,
            training_summary=training_summary,
            candidate_model_path=candidate_model_path,
            champion_model_path=champion_model_path,
            frozen_probe_set_path=frozen_probe_set_path,
            t_start=t_start,
        )

    test_prompts, test_gold, test_items = _load_alpaca_eval_items(test_path, max_items=EVAL_TEST_MAX_ITEMS)
    cotest_prompts, cotest_gold, _cotest_items = _load_alpaca_eval_items(cotest_path, max_items=EVAL_COTEST_MAX_ITEMS)
    if use_mathbench_probe:
        frozen_prompts: list[str] = []
        frozen_gold: list[str] = []
        frozen_questions: list[dict] = []
    else:
        frozen_prompts, frozen_gold, frozen_questions = _load_frozen_probe_items(frozen_probe_set_path)

    champion_prompts = test_prompts + cotest_prompts + frozen_prompts
    champion_predictions = _run_eval_batch(champion_model_path, champion_prompts) if champion_prompts else []
    cursor = 0
    test_champion_preds = champion_predictions[cursor:cursor + len(test_prompts)]
    cursor += len(test_prompts)
    cotest_champion_preds = champion_predictions[cursor:cursor + len(cotest_prompts)]
    cursor += len(cotest_prompts)
    frozen_champion_preds = champion_predictions[cursor:cursor + len(frozen_prompts)]

    candidate_prompts = test_prompts + cotest_prompts + frozen_prompts
    candidate_predictions = _run_eval_batch(candidate_model_path, candidate_prompts) if candidate_prompts else []
    cursor = 0
    test_candidate_preds = candidate_predictions[cursor:cursor + len(test_prompts)]
    cursor += len(test_prompts)
    cotest_candidate_preds = candidate_predictions[cursor:cursor + len(cotest_prompts)]
    cursor += len(cotest_prompts)
    frozen_candidate_preds = candidate_predictions[cursor:cursor + len(frozen_prompts)]

    frozen_difficulties = (
        []
        if use_mathbench_probe
        else tag_probe_dynamic_difficulty(
            candidate_model_path, frozen_probe_set_path, frozen_prompts, frozen_gold, frozen_questions, trace_id, round_id=round_id,
        )
    )
    _test_champion_judgements = _judgements_from_predictions(
        test_champion_preds,
        test_gold,
        prompts=test_prompts,
        items=test_items,
        state=state,
    )
    _test_candidate_judgements = _judgements_from_predictions(
        test_candidate_preds,
        test_gold,
        prompts=test_prompts,
        items=test_items,
        state=state,
    )
    _cotest_champion_judgements = _judgements_from_predictions(
        cotest_champion_preds,
        cotest_gold,
        prompts=cotest_prompts,
        items=_cotest_items,
        state=state,
    )
    _cotest_candidate_judgements = _judgements_from_predictions(
        cotest_candidate_preds,
        cotest_gold,
        prompts=cotest_prompts,
        items=_cotest_items,
        state=state,
    )
    _test_champion_results = [bool(j.get("correct")) for j in _test_champion_judgements]
    _test_candidate_results = [bool(j.get("correct")) for j in _test_candidate_judgements]
    _cotest_champion_results = [bool(j.get("correct")) for j in _cotest_champion_judgements]
    _cotest_candidate_results = [bool(j.get("correct")) for j in _cotest_candidate_judgements]
    _write_eval_trace_rows(
        trace_id=trace_id,
        round_id=round_id,
        stage="eval_test_1shot",
        model_role="champion",
        model_path=champion_model_path,
        prompts=test_prompts,
        gold_answers=test_gold,
        predictions=test_champion_preds,
        items=test_items,
        split_role="test",
        correct_results=_test_champion_results,
        judgements=_test_champion_judgements,
    )
    _write_eval_trace_rows(
        trace_id=trace_id,
        round_id=round_id,
        stage="eval_test_1shot",
        model_role="candidate",
        model_path=candidate_model_path,
        prompts=test_prompts,
        gold_answers=test_gold,
        predictions=test_candidate_preds,
        items=test_items,
        split_role="test",
        correct_results=_test_candidate_results,
        judgements=_test_candidate_judgements,
    )
    _write_eval_trace_rows(
        trace_id=trace_id,
        round_id=round_id,
        stage="eval_cotest_1shot",
        model_role="champion",
        model_path=champion_model_path,
        prompts=cotest_prompts,
        gold_answers=cotest_gold,
        predictions=cotest_champion_preds,
        items=_cotest_items,
        split_role="cotest",
        correct_results=_cotest_champion_results,
        judgements=_cotest_champion_judgements,
    )
    _write_eval_trace_rows(
        trace_id=trace_id,
        round_id=round_id,
        stage="eval_cotest_1shot",
        model_role="candidate",
        model_path=candidate_model_path,
        prompts=cotest_prompts,
        gold_answers=cotest_gold,
        predictions=cotest_candidate_preds,
        items=_cotest_items,
        split_role="cotest",
        correct_results=_cotest_candidate_results,
        judgements=_cotest_candidate_judgements,
    )
    _write_eval_trace_rows(
        trace_id=trace_id,
        round_id=round_id,
        stage="eval_frozen_1shot",
        model_role="champion",
        model_path=champion_model_path,
        prompts=frozen_prompts,
        gold_answers=frozen_gold,
        predictions=frozen_champion_preds,
        items=frozen_questions,
        split_role="frozen",
        difficulties=frozen_difficulties,
    )
    _write_eval_trace_rows(
        trace_id=trace_id,
        round_id=round_id,
        stage="eval_frozen_1shot",
        model_role="candidate",
        model_path=candidate_model_path,
        prompts=frozen_prompts,
        gold_answers=frozen_gold,
        predictions=frozen_candidate_preds,
        items=frozen_questions,
        split_role="frozen",
        difficulties=frozen_difficulties,
    )

    test_champion_correct = sum(_test_champion_results)
    test_candidate_correct = sum(_test_candidate_results)

    old_indices, new_indices = _split_test_by_role(test_items)

    def _compute_correct(indices: list[int], results: list[bool]) -> int:
        return sum(1 for i in indices if i < len(results) and results[i])

    def _compute_acc(indices: list[int], results: list[bool]) -> float:
        if not indices:
            return 0.0
        return _compute_correct(indices, results) / len(indices)

    old_ability_correct_before = _compute_correct(old_indices, _test_champion_results)
    old_ability_correct_after = _compute_correct(old_indices, _test_candidate_results)
    new_ability_correct_before = _compute_correct(new_indices, _test_champion_results)
    new_ability_correct_after = _compute_correct(new_indices, _test_candidate_results)
    old_ability_acc_before = _compute_acc(old_indices, _test_champion_results)
    old_ability_acc_after = _compute_acc(old_indices, _test_candidate_results)
    new_ability_acc_before = _compute_acc(new_indices, _test_champion_results)
    new_ability_acc_after = _compute_acc(new_indices, _test_candidate_results)
    new_ability_gate_before = new_ability_acc_before if new_indices else None
    new_ability_gate_after = new_ability_acc_after if new_indices else None
    forgetting_delta = old_ability_acc_before - old_ability_acc_after

    new_skill_acc_before = test_champion_correct / len(test_gold) if test_gold else 0.0
    new_skill_acc_after = test_candidate_correct / len(test_gold) if test_gold else 0.0

    _cotest_before_correct = sum(_cotest_champion_results)
    _cotest_after_correct = sum(_cotest_candidate_results)
    cotest_acc_before = _cotest_before_correct / len(cotest_gold) if cotest_gold else 0.0
    cotest_acc_after = _cotest_after_correct / len(cotest_gold) if cotest_gold else 0.0

    mathbench_candidate_result: MathBenchProbeResult | None = None
    mathbench_champion_result: MathBenchProbeResult | None = None
    if use_mathbench_probe:
        mathbench_champion_result = run_mathbench_probe(
            champion_model_path,
            trace_id=trace_id,
            round_id=round_id,
            model_role="champion",
        )
        mathbench_candidate_result = run_mathbench_probe(
            candidate_model_path,
            trace_id=trace_id,
            round_id=round_id,
            model_role="candidate",
        )
        probe_acc_frozen = mathbench_candidate_result.score
        frozen_champion_acc = mathbench_champion_result.score
        frozen_per_difficulty_acc = {"mathbench": probe_acc_frozen}
        _frozen_failures: list[dict] = []
    else:
        probe_acc_frozen, frozen_per_difficulty_acc, _frozen_failures = _probe_stats_from_predictions(
            frozen_candidate_preds, frozen_gold, frozen_difficulties, frozen_questions, state,
        )
        frozen_champion_acc, _, _ = _probe_stats_from_predictions(
            frozen_champion_preds, frozen_gold, frozen_difficulties, frozen_questions, state,
        )
    champion_frozen_probe_error = 1.0 - frozen_champion_acc if (frozen_gold or use_mathbench_probe) else state.get("champion_frozen_probe_error")

    base_frozen_error_rate = state.get("base_frozen_error_rate")

    print(
        "[evaluator] batched evaluation "
        f"champion_items={len(champion_prompts)} candidate_items={len(candidate_prompts)} "
        f"old={old_ability_acc_before:.3f}->{old_ability_acc_after:.3f} "
        f"({old_ability_correct_before}/{len(old_indices)}->{old_ability_correct_after}/{len(old_indices)}) "
        f"new={new_ability_acc_before:.3f}->{new_ability_acc_after:.3f} "
        f"({new_ability_correct_before}/{len(new_indices)}->{new_ability_correct_after}/{len(new_indices)}) "
        f"test={new_skill_acc_before:.3f}->{new_skill_acc_after:.3f} "
        f"({test_champion_correct}/{len(test_gold)}->{test_candidate_correct}/{len(test_gold)}) "
        f"forget={forgetting_delta:+.3f} "
        f"frozen_1shot={probe_acc_frozen:.3f} champion_1shot={frozen_champion_acc:.3f} "
        f"base_frozen_error={base_frozen_error_rate}"
    )

    frozen_3shot = None if use_mathbench_probe else _run_frozen_multi_rollout(
        frozen_prompts, frozen_gold, candidate_model_path,
        trace_id=trace_id,
        round_id=round_id,
        questions=frozen_questions,
        difficulties=frozen_difficulties,
    )
    if frozen_3shot:
        total = frozen_3shot["total"]
        print(
            f"[evaluator] frozen 3-shot ({total}q): "
            f"3/3={frozen_3shot['correct_3of3']} "
            f"2/3={frozen_3shot['correct_2of3']} "
            f"1/3={frozen_3shot['correct_1of3']} "
            f"0/3={frozen_3shot['correct_0of3']} "
            f"pass@1={frozen_3shot['pass_at_1']:.3f}"
        )

    _save_test_accuracy(
        session_dir, round_id, test_prompts, test_gold, test_items,
        candidate_model_path, state,
        trace_id=trace_id,
    )

    gates = decide_evaluation_gates(
        old_error_rate=0.0,
        champion_old_error_rate=None,
        new_skill_acc_before=new_skill_acc_before,
        new_skill_acc_after=new_skill_acc_after,
        cotest_acc_before=cotest_acc_before,
        cotest_acc_after=cotest_acc_after,
        probe_acc_after=probe_acc_frozen,
        champion_probe_error_rate=champion_frozen_probe_error,
        old_skill_eval_count=len(old_indices),
        base_frozen_error_rate=base_frozen_error_rate,
        old_ability_acc_before=old_ability_acc_before,
        old_ability_acc_after=old_ability_acc_after,
        forgetting_delta=forgetting_delta,
        new_ability_acc_before=new_ability_gate_before,
        new_ability_acc_after=new_ability_gate_after,
        probe_acc_for_gate=frozen_3shot["pass_at_1"] if frozen_3shot else None,
    )

    payload = EvalResultPayload(
        old_error_count=0,
        old_error_rate=0.0,
        new_skill_acc_before=new_skill_acc_before,
        new_skill_acc_after=new_skill_acc_after,
        new_skill_correct_before=test_champion_correct,
        new_skill_correct_after=test_candidate_correct,
        new_skill_eval_count=len(test_gold),
        cotest_acc_before=cotest_acc_before,
        cotest_acc_after=cotest_acc_after,
        old_ability_acc_before=old_ability_acc_before,
        old_ability_acc_after=old_ability_acc_after,
        old_ability_correct_before=old_ability_correct_before,
        old_ability_correct_after=old_ability_correct_after,
        old_ability_eval_count=len(old_indices),
        new_ability_acc_before=new_ability_acc_before,
        new_ability_acc_after=new_ability_acc_after,
        new_ability_correct_before=new_ability_correct_before,
        new_ability_correct_after=new_ability_correct_after,
        new_ability_eval_count=len(new_indices),
        forgetting_delta=forgetting_delta,
        probe_acc_after=probe_acc_frozen,
        probe_acc_frozen=probe_acc_frozen,
        probe_acc_champion=frozen_champion_acc,
        external_probe_acc=0.0,
        external_probe_acc_champion=0.0,
        probe_easy_acc=frozen_per_difficulty_acc.get("easy"),
        probe_medium_acc=frozen_per_difficulty_acc.get("medium"),
        probe_hard_acc=frozen_per_difficulty_acc.get("hard"),
        champion_probe_error_rate=champion_frozen_probe_error,
        base_frozen_error_rate=base_frozen_error_rate,
        pass_old_skill_gate=gates.pass_old_skill_gate,
        pass_new_skill_gate=gates.pass_new_skill_gate,
        pass_probe_gate=gates.pass_probe_gate,
        pass_frozen_gate=gates.pass_frozen_gate,
        should_stop=gates.should_stop,
        should_promote_candidate=gates.should_promote_candidate,
        training_summary=training_summary,
    )

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.EVALUATOR,
            receiver=AgentName.STRATEGY_INSPECTOR,
            message_type=MessageType.EVAL_RESULT,
        ),
        payload=payload,
    )

    elapsed = time.time() - t_start
    print(f"[evaluator] Total evaluation time: {elapsed:.2f}s")

    return {
        "metrics_before": {
            "new_skill_acc": new_skill_acc_before,
            "cotest_acc": cotest_acc_before,
            "old_ability_acc": old_ability_acc_before,
            "new_ability_acc": new_ability_acc_before,
            "old_ability_correct": old_ability_correct_before,
            "old_ability_eval_count": len(old_indices),
            "new_ability_correct": new_ability_correct_before,
            "new_ability_eval_count": len(new_indices),
            "new_skill_correct": test_champion_correct,
            "new_skill_eval_count": len(test_gold),
            "champion_frozen_probe_error": champion_frozen_probe_error,
        },
        "metrics_after": {
            "new_skill_acc": new_skill_acc_after,
            "cotest_acc": cotest_acc_after,
            "old_ability_acc": old_ability_acc_after,
            "new_ability_acc": new_ability_acc_after,
            "old_ability_correct": old_ability_correct_after,
            "old_ability_eval_count": len(old_indices),
            "new_ability_correct": new_ability_correct_after,
            "new_ability_eval_count": len(new_indices),
            "new_skill_correct": test_candidate_correct,
            "new_skill_eval_count": len(test_gold),
            "forgetting_delta": forgetting_delta,
            "probe_acc": probe_acc_frozen,
            "probe_acc_frozen": probe_acc_frozen,
            "probe_acc_champion": frozen_champion_acc,
            "probe_acc_for_gate": gates.probe_acc_for_gate,
            "frozen_degrade_tolerance_used": gates.frozen_degrade_tolerance_used,
            "champion_frozen_probe_error": champion_frozen_probe_error,
            "per_difficulty_acc_frozen": frozen_per_difficulty_acc,
            "probe_easy_acc": frozen_per_difficulty_acc.get("easy"),
            "probe_medium_acc": frozen_per_difficulty_acc.get("medium"),
            "probe_hard_acc": frozen_per_difficulty_acc.get("hard"),
            "mathbench_probe": _mathbench_probe_metadata(mathbench_candidate_result) if mathbench_candidate_result else None,
            "training_summary": training_summary,
        },
        "old_skill_eval_count": len(old_indices),
        "champion_frozen_probe_error": champion_frozen_probe_error,
        "pending_message": msg,
    }


def _is_loadable_model_dir(model_path: str) -> bool:
    """检查模型目录是否可加载。

    条件：
    - 路径存在且为目录时：必须有 config.json/adapter_config.json +
      至少一个权重文件(.safetensors / pytorch_model*.bin / adapter_model.bin)
    - 路径为文件时：直接视为有效（可能是单文件 checkpoint）

    Returns:
        True 如果模型目录可被 transformers 加载。
    """
    if not model_path:
        return False
    path = Path(model_path)
    if path.is_dir():
        has_config = (path / "config.json").exists() or (path / "adapter_config.json").exists()
        has_weights = (
            bool(list(path.glob("*.safetensors")))
            or bool(list(path.glob("pytorch_model*.bin")))
            or (path / "adapter_model.bin").exists()
        )
        return has_config and has_weights
    return True


def _training_failure_eval_result(
    state: EvoState,
    train_result: TrainResultPayload,
    training_summary: dict,
    candidate_model_path: str,
    champion_model_path: str,
    frozen_probe_set_path: str,
    t_start: float,
) -> dict:
    """训练失败时的评估结果生成。

    当训练产出不可用时（模型训练失败或无法加载），跳过候选模型推理，
    只用冠军模型在 frozen probe 上做评估作为基线指标。
    返回的 EvalResultPayload 中 pass_new_skill_gate=False，
    确保本轮不会被晋升。
    """
    print(
        "[evaluator] Training failed or produced an unloadable candidate; "
        f"candidate={candidate_model_path} error={train_result.error_message!r}. "
        "Skipping candidate inference."
    )
    if mathbench_probe_enabled(FROZEN_PROBE_EVAL_METHOD):
        champion_result = run_mathbench_probe(
            champion_model_path,
            trace_id=str(state.get("trace_id", "")),
            round_id=int(state.get("round_id", 0) or 0),
            model_role="champion",
        )
        probe_acc_frozen = champion_result.score
        champion_frozen_probe_error = 1.0 - probe_acc_frozen
        payload = EvalResultPayload(
            old_error_count=0, old_error_rate=0.0,
            new_skill_acc_before=0.0, new_skill_acc_after=0.0,
            cotest_acc_before=0.0, cotest_acc_after=0.0,
            probe_acc_after=probe_acc_frozen, probe_acc_frozen=probe_acc_frozen,
            probe_acc_champion=probe_acc_frozen,
            external_probe_acc=0.0, external_probe_acc_champion=0.0,
            probe_easy_acc=None,
            probe_medium_acc=None,
            probe_hard_acc=None,
            pass_old_skill_gate=True, pass_new_skill_gate=False,
            pass_probe_gate=False, pass_frozen_gate=True,
            should_stop=False, should_promote_candidate=False,
            training_summary=training_summary,
        )
        msg = RoutedMessage(
            header=MessageHeader(
                trace_id=str(state.get("trace_id", "")), round_id=int(state.get("round_id", 0) or 0),
                sender=AgentName.EVALUATOR, receiver=AgentName.STRATEGY_INSPECTOR,
                message_type=MessageType.EVAL_RESULT,
            ),
            payload=payload,
        )
        elapsed = time.time() - t_start
        print(f"[evaluator] Training failure MathBench evaluation completed in {elapsed:.2f}s, score={probe_acc_frozen:.3f}")
        return {
            "metrics_before": {
                "training_failed": True,
                "champion_frozen_probe_error": champion_frozen_probe_error,
                "mathbench_probe": _mathbench_probe_metadata(champion_result),
            },
            "metrics_after": {
                "training_failed": True,
                "probe_acc_frozen": probe_acc_frozen,
                "champion_frozen_probe_error": champion_frozen_probe_error,
                "training_summary": training_summary,
                "mathbench_probe": _mathbench_probe_metadata(champion_result),
            },
            "old_skill_eval_count": 0, "champion_frozen_probe_error": champion_frozen_probe_error,
            "pending_message": msg,
        }

    frozen_prompts, frozen_gold, frozen_questions = _load_frozen_probe_items(frozen_probe_set_path)
    champion_prompts = frozen_prompts
    champion_predictions = _run_eval_batch(champion_model_path, champion_prompts) if champion_prompts else []
    frozen_champion_preds = champion_predictions[:len(frozen_prompts)]
    frozen_difficulties = tag_probe_dynamic_difficulty(
        champion_model_path, frozen_probe_set_path, frozen_prompts, frozen_gold, frozen_questions, state.get("trace_id", ""),
        round_id=int(state.get("round_id", 0) or 0),
    ) if frozen_prompts else []
    _write_eval_trace_rows(
        trace_id=str(state.get("trace_id", "")),
        round_id=int(state.get("round_id", 0) or 0),
        stage="eval_failure_frozen_1shot",
        model_role="champion",
        model_path=champion_model_path,
        prompts=frozen_prompts,
        gold_answers=frozen_gold,
        predictions=frozen_champion_preds,
        items=frozen_questions,
        split_role="frozen",
        difficulties=frozen_difficulties,
    )
    probe_acc_frozen, frozen_per_difficulty_acc, _ = _probe_stats_from_predictions(
        frozen_champion_preds, frozen_gold, frozen_difficulties, frozen_questions, state,
    )
    champion_frozen_probe_error = 1.0 - probe_acc_frozen if frozen_gold else state.get("champion_frozen_probe_error")

    payload = EvalResultPayload(
        old_error_count=0, old_error_rate=0.0,
        new_skill_acc_before=0.0, new_skill_acc_after=0.0,
        cotest_acc_before=0.0, cotest_acc_after=0.0,
        probe_acc_after=probe_acc_frozen, probe_acc_frozen=probe_acc_frozen,
        external_probe_acc=0.0, external_probe_acc_champion=0.0,
        probe_easy_acc=frozen_per_difficulty_acc.get("easy"),
        probe_medium_acc=frozen_per_difficulty_acc.get("medium"),
        probe_hard_acc=frozen_per_difficulty_acc.get("hard"),
        pass_old_skill_gate=True, pass_new_skill_gate=False,
        pass_probe_gate=False, pass_frozen_gate=True,
        should_stop=False, should_promote_candidate=False,
        training_summary=training_summary,
    )

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=str(state.get("trace_id", "")), round_id=int(state.get("round_id", 0) or 0),
            sender=AgentName.EVALUATOR, receiver=AgentName.STRATEGY_INSPECTOR,
            message_type=MessageType.EVAL_RESULT,
        ),
        payload=payload,
    )

    elapsed = time.time() - t_start
    print(f"[evaluator] Training failure evaluation completed in {elapsed:.2f}s, frozen_probe={probe_acc_frozen:.3f}")
    return {
        "metrics_before": {"training_failed": True, "champion_frozen_probe_error": champion_frozen_probe_error},
        "metrics_after": {"training_failed": True, "probe_acc_frozen": probe_acc_frozen, "champion_frozen_probe_error": champion_frozen_probe_error, "training_summary": training_summary},
        "old_skill_eval_count": 0, "champion_frozen_probe_error": champion_frozen_probe_error,
        "pending_message": msg,
    }


def _run_frozen_multi_rollout(
    frozen_prompts: list[str],
    frozen_gold: list[str],
    candidate_model_path: str,
    trace_id: str = "",
    round_id: int = 0,
    questions: list[dict] | None = None,
    difficulties: list[str] | None = None,
) -> dict | None:
    """对 frozen probe 做多次采样评估，计算 pass@k 统计。

    对每道题做 TEST_ROLLOUT_TIMES 次推理（默认3次），统计：
    - 3/3 全对的题目数
    - 2/3 对的题目数
    - 1/3 对的题目数
    - 0/3 全错的题目数
    - pass@1 = (total - 0/3次数) / total（至少答对一次的比例）

    3-shot 统计用于低 1-shot 准确率阶段的稳健 gate 信号。
    当 1-shot frozen probe 仍较低时，strategy_policy 可使用 pass@1 减少噪声。

    Returns:
        None 如果 prompts 为空或模型路径为空；
        否则返回 {"total", "correct_3of3", "correct_2of3", "correct_1of3",
                  "correct_0of3", "pass_at_1"}。
    """
    rollout_times = max(1, TEST_ROLLOUT_TIMES)
    if not frozen_prompts or not candidate_model_path:
        return None

    all_prompts: list[str] = []
    for p in frozen_prompts:
        all_prompts.extend([p] * rollout_times)

    predictions = _run_eval_batch(
        candidate_model_path, all_prompts,
        temperature=ROLLOUT_TEMPERATURE, top_p=ROLLOUT_TOP_P,
    )

    stats = {3: 0, 2: 0, 1: 0, 0: 0}
    trace_rows = []
    for idx in range(len(frozen_prompts)):
        gold = frozen_gold[idx] if idx < len(frozen_gold) else ""
        item = questions[idx] if questions and idx < len(questions) else {}
        chunk = predictions[idx * rollout_times:(idx + 1) * rollout_times]
        correct_count = sum(1 for pred in chunk if _judge_eval_prediction(pred, gold, item=item, allow_llm_judge=False))
        stats[correct_count] = stats.get(correct_count, 0) + 1
        if trace_id:
            difficulty = (
                difficulties[idx]
                if difficulties is not None and idx < len(difficulties)
                else item.get("dynamic_difficulty", "")
            )
            for rollout_idx, pred in enumerate(chunk):
                trace_rows.append(build_inference_trace_row(
                    trace_id=trace_id,
                    round_id=round_id,
                    stage="eval_frozen_multi_rollout",
                    model_role="candidate",
                    model_path=candidate_model_path,
                    question_id=str(item.get("question_id", "")),
                    prompt=frozen_prompts[idx],
                    gold_answer=gold,
                    prediction=pred,
                    correct=_judge_eval_prediction(pred, gold, item=item, allow_llm_judge=False),
                    max_new_tokens=int(EVAL_MAX_NEW_TOKENS),
                    temperature=ROLLOUT_TEMPERATURE,
                    top_p=ROLLOUT_TOP_P,
                    rollout_idx=rollout_idx,
                    split_role="frozen",
                    module=str(item.get("module", "")),
                    dynamic_difficulty=str(difficulty or ""),
                    metadata={"rollout_count_for_question": correct_count},
                ))
    if trace_rows:
        write_inference_trace_rows(
            trace_id=trace_id,
            round_id=round_id,
            stage="eval_frozen_multi_rollout",
            rows=trace_rows,
        )

    total = len(frozen_prompts)
    pass1 = (total - stats[0]) / total if total else 0.0
    return {
        "total": total,
        "correct_3of3": stats[3],
        "correct_2of3": stats[2],
        "correct_1of3": stats[1],
        "correct_0of3": stats[0],
        "pass_at_1": pass1,
    }


def _save_test_accuracy(
    session_dir: Path, round_id: int,
    base_prompts: list[str], gold_answers: list[str], test_items: list[dict],
    candidate_model_path: str,
    state: EvoState | None = None,
    trace_id: str = "",
) -> None:
    """持久化 test set 的逐题准确率明细到 JSON 文件。

    对每道题做 TEST_ROLLOUT_TIMES 次推理，记录：
    - correct: 第一次推理是否答对（用于 1-shot 准确率）
    - correct_count: TEST_ROLLOUT_TIMES 次中答对的次数
    - split_role: 题目类别(old_ability/new_ability/test)

    输出路径: session_dir/round_{round_id}_datasets/test_accuracy.json
    """
    rollout_times = max(1, TEST_ROLLOUT_TIMES)
    if not base_prompts or not candidate_model_path:
        return

    all_prompts: list[str] = []
    for p in base_prompts:
        all_prompts.extend([p] * rollout_times)

    predictions = _run_eval_batch(
        candidate_model_path, all_prompts,
        temperature=ROLLOUT_TEMPERATURE, top_p=ROLLOUT_TOP_P,
    )
    expanded_gold: list[str] = []
    expanded_items: list[dict] = []
    for idx in range(len(base_prompts)):
        gold = gold_answers[idx] if idx < len(gold_answers) else ""
        item = test_items[idx] if idx < len(test_items) else {}
        expanded_gold.extend([gold] * rollout_times)
        expanded_items.extend([item] * rollout_times)
    expanded_judgements = _judgements_from_predictions(
        predictions,
        expanded_gold,
        prompts=all_prompts,
        items=expanded_items,
        state=state,
    )
    expanded_results = [bool(judgement.get("correct")) for judgement in expanded_judgements]

    rows: list[dict] = []
    trace_rows = []
    for idx in range(len(base_prompts)):
        gold = gold_answers[idx] if idx < len(gold_answers) else ""
        chunk_start = idx * rollout_times
        chunk_preds = predictions[chunk_start:chunk_start + rollout_times]
        chunk_results = expanded_results[chunk_start:chunk_start + rollout_times]
        chunk_judgements = expanded_judgements[chunk_start:chunk_start + rollout_times]
        correct_count = sum(chunk_results)
        item = test_items[idx] if idx < len(test_items) else {}
        is_correct = chunk_results[0] if chunk_results else False
        rows.append({
            "question_id": item.get("question_id", ""),
            "question": {
                "question_text": _eval_prompt_from_item(item) or base_prompts[idx],
                "gold_answer": gold,
                "module": item.get("module", "unknown"),
                "dynamic_difficulty": item.get("dynamic_difficulty", "unknown"),
            },
            "correct": is_correct,
            "correct_count": correct_count,
            "rollout_count": rollout_times,
            "split_role": item.get("split_role", "test"),
        })
        if trace_id:
            for rollout_idx, pred in enumerate(chunk_preds):
                judgement = chunk_judgements[rollout_idx] if rollout_idx < len(chunk_judgements) else {}
                row_metadata = {
                    "rollout_count_for_question": correct_count,
                    "evaluation_method": _evaluation_method_for_item(item),
                    "needs_judge": _evaluation_method_for_item(item) == "llm_judge",
                    "reference_solution_present": bool(_reference_solution_for_item(item)),
                    "judge_score": judgement.get("score"),
                    "judge_reason": judgement.get("reason"),
                    "judge_source": judgement.get("source"),
                    "judge_raw_text": judgement.get("judge_raw_text", ""),
                    "judge_fallback_used": bool(judgement.get("fallback_used", False)),
                    "judge_schema_errors": list(judgement.get("schema_errors") or []),
                }
                trace_rows.append(build_inference_trace_row(
                    trace_id=trace_id,
                    round_id=round_id,
                    stage="eval_test_multi_rollout",
                    model_role="candidate",
                    model_path=candidate_model_path,
                    question_id=str(item.get("question_id", "")),
                    prompt=base_prompts[idx],
                    gold_answer=gold,
                    prediction=pred,
                    correct=chunk_results[rollout_idx] if rollout_idx < len(chunk_results) else False,
                    max_new_tokens=int(EVAL_MAX_NEW_TOKENS),
                    temperature=ROLLOUT_TEMPERATURE,
                    top_p=ROLLOUT_TOP_P,
                    rollout_idx=rollout_idx,
                    split_role=str(item.get("split_role", "test")),
                    module=str(item.get("module", "")),
                    dynamic_difficulty=str(item.get("dynamic_difficulty", "")),
                    metadata=row_metadata,
                ))
    if trace_rows:
        write_inference_trace_rows(
            trace_id=trace_id,
            round_id=round_id,
            stage="eval_test_multi_rollout",
            rows=trace_rows,
        )
    path = session_dir / f"round_{round_id}_datasets" / "test_accuracy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _persist_teacher_decision(
    state: EvoState,
    per_difficulty_acc: dict[str, float],
    frozen_per_difficulty_acc: dict[str, float],
    external_probe_acc: float | None = None,
    external_probe_acc_champion: float | None = None,
    external_per_difficulty_acc: dict[str, float] | None = None,
) -> None:
    """追加 per-difficulty 准确率到 teacher_decisions.jsonl，用于可观测性。

    每轮评估结束后，将本轮的各维度难度准确率、外部探针准确率、
    sampling_plan 等信息追加写入 teacher_decisions.jsonl，
    便于事后分析 teacher 的决策质量和评估趋势。

    注意：此函数当前在 evaluator_node 中未被调用（预留扩展点）。
    """
    session_dir = get_session_dir(str(state.get("trace_id", "")))
    decisions_path = session_dir / "teacher_decisions.jsonl"
    entry = {
        "round": state.get("round_id", 0),
        "input_per_difficulty_acc": per_difficulty_acc,
        "input_per_difficulty_acc_frozen": frozen_per_difficulty_acc,
        "input_per_difficulty_acc_external": external_per_difficulty_acc or {},
        "external_probe_acc": external_probe_acc,
        "external_probe_acc_champion": external_probe_acc_champion,
        "external_probe_acc_delta": (
            external_probe_acc - external_probe_acc_champion
            if external_probe_acc is not None and external_probe_acc_champion is not None
            else None
        ),
        "probe_easy_acc": frozen_per_difficulty_acc.get("easy"),
        "probe_medium_acc": frozen_per_difficulty_acc.get("medium"),
        "probe_hard_acc": frozen_per_difficulty_acc.get("hard"),
        "sampling_plan": state.get("sampling_plan") or {},
        "weakest_bucket": None,
        "decision": None,
    }
    with open(decisions_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
