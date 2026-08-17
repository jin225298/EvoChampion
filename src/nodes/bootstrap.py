"""
Bootstrap 节点 —— 系统初始化与跨轮次状态恢复

核心职责：
1. 创建或恢复 LangGraph 图执行的初始 State：trace_id、round_id、DAG、replay buffer、
   数据窗口参数、probe set 路径等
2. 构建全局探针集（global probe set）和固定基准集（frozen probe set），
   支持两种来源：benchmark holdout / seed 题库
3. 将探针题注册到 question_registry，标记为 probe_holdout / external_probe，
   防止跨轮次数据泄露
4. 评估基础模型在 frozen probe 上的基线表现（1-shot + 3-shot），
   作为后续 evaluator 计算 forgetting score 的参考值
5. 发送首条 GOAL_REQUEST 消息给 TEACHER，启动训练循环

执行顺序：
  加载 checkpoint / DAG / replay → 确定 round_id → 确定 champion 模型路径 →
  构建 global probe set → 构建 frozen probe set → 注册 holdout 题 →
  评估 baseline → 组装 State → 发出 GOAL_REQUEST → 返回

在整个 graph 中的位置：
  __start__ → bootstrap_node → teacher → searcher → ... → strategy_inspector → 回环
"""

import json
import uuid
from pathlib import Path

from config.settings import (
    BENCHMARK_ANSWER_KEY,
    BENCHMARK_DATASET_ID,
    BENCHMARK_QUESTION_KEY,
    BENCHMARK_TEST_KEY,
    BENCHMARK_ENTRY_POINT_KEY,
    IS_CODE_DOMAIN,
    CANDIDATE_MODEL_DIR,
    CHAMPION_MODEL_PATH,
    EVAL_MAX_NEW_TOKENS,
    get_classifier_labels,
    EXTERNAL_PROBE_MANIFEST_PATH,
    EXTERNAL_PROBE_PATH,
    EVOLVE_TRACE_ID,
    FIXED_BENCHMARK_SEED,
    FIXED_BENCHMARK_SIZE,
    FIXED_BENCHMARK_SOURCE,
    FROZEN_PROBE_EVAL_METHOD,
    FROZEN_PROBE_SIZE,
    GLOBAL_PROBE_SIZE,
    GLOBAL_PROBE_SOURCE,
    GLOBAL_PROBE_STRATIFIED_BY_MODULE,
    DATA_WINDOW_SIZE,
    get_session_dir,
    MAX_PROFILE_ITEMS_PER_ROUND,
    MAX_PROFILE_WINDOWS_PER_ROUND,
    PROBE_BANK_PATH,
    PROBE_SET_SIZE_PER_CLASS,
)
from src.models.messages import (
    AgentName,
    GoalRequestPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
)
from src.models.state import EvoState
from src.tools.model_runner import warmup_model
from src.tools.model_runner import judge_answer, judge_prediction_for_item
from src.tools.question_registry import mark_questions_active_holdout, mark_questions_probe_holdout
from src.tools.question_registry import mark_questions_external_probe
from src.tools.replay_buffer import load_replay_buffer
from src.tools.search_dag import create_root_node, load_search_dag
from src.tools.dataset_bank import build_probe_from_bank, load_dataset_bank
from src.tools.mathbench_probe import build_mathbench_probe_marker, mathbench_probe_enabled, run_mathbench_probe


# ── Checkpoint / 状态恢复 ───────────────────────────────────

def _load_evolution_checkpoint(trace_id: str) -> dict:
    """加载进化检查点文件（evolution_checkpoint.json）。

    检查点由 strategy_inspector 在每轮结束时写入，包含：
    - next_round_id / data_window_offset / data_window_size
    - champion_model_path / replay_sample_ratio_override
    - rollout 统计、diagnostic 路径、target_bucket 等

    Args:
        trace_id: 会话标识

    Returns:
        检查点字典，不存在时返回空字典
    """
    checkpoint_path = get_session_dir(trace_id) / "evolution_checkpoint.json"
    if not checkpoint_path.exists():
        return {}
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _infer_next_round_id(search_dag_edges: list[dict]) -> int:
    """从 DAG 边中推断下一轮 ID。

    解析所有 action_summary 中的 "round_N" 前缀，取最大 N + 1。

    Args:
        search_dag_edges: DAG 边列表

    Returns:
        下一轮 ID，无边时返回 0
    """
    round_ids = []
    for edge in search_dag_edges:
        summary = edge.get("action_summary", "")
        if not isinstance(summary, str) or not summary.startswith("round_"):
            continue
        raw_round = summary.split(" ", 1)[0].removeprefix("round_")
        if raw_round.isdigit():
            round_ids.append(int(raw_round))
    return max(round_ids) + 1 if round_ids else 0


def _extract_round_and_decision(edge: dict) -> tuple[int | None, str]:
    """从 DAG 边中提取轮次和决策类型。

    解析 action_summary 格式 "round_N <decision> ..."。

    Args:
        edge: DAG 边字典

    Returns:
        (round_id, decision) 或 (None, "")
    """
    summary = edge.get("action_summary", "")
    if not isinstance(summary, str) or not summary.startswith("round_"):
        return None, ""
    parts = summary.split(" ", 2)
    if len(parts) < 2:
        return None, ""
    raw_round = parts[0].removeprefix("round_")
    if not raw_round.isdigit():
        return None, ""
    return int(raw_round), parts[1]


def _infer_latest_promoted_champion(trace_id: str, search_dag_edges: list[dict]) -> str:
    """推断最近一次被 promote 的 champion 模型路径。

    遍历所有 DAG 边，找到决策为 "promote" 的最大轮次，
    检查 CANDIDATE_MODEL_DIR 下是否存在对应模型目录。

    Args:
        trace_id: 会话标识
        search_dag_edges: DAG 边列表

    Returns:
        champion 模型路径，无 promote 记录时返回空字符串
    """
    promoted_rounds = []
    for edge in search_dag_edges:
        round_id, decision = _extract_round_and_decision(edge)
        if round_id is not None and decision == "promote":
            promoted_rounds.append(round_id)
    for round_id in sorted(promoted_rounds, reverse=True):
        candidate_path = Path(CANDIDATE_MODEL_DIR) / f"candidate_{trace_id}_round{round_id}"
        if candidate_path.exists():
            return str(candidate_path)
    return ""


# ── 探针集构建 ──────────────────────────────────────────────

def _build_seed_probe_set() -> dict:
    """从种子题库构建探针集。

    按分类标签从 PROBE_BANK_PATH 中每类取 PROBE_SET_SIZE_PER_CLASS 题。

    Returns:
        {label: [question_dict, ...]} 格式的探针集
    """
    probe_bank_path = Path(PROBE_BANK_PATH)
    if probe_bank_path.exists():
        with open(probe_bank_path, "r", encoding="utf-8") as f:
            probe_bank = json.load(f)
    else:
        probe_bank = {label: [] for label in get_classifier_labels()}

    probe_set = {}
    for label in get_classifier_labels():
        questions = probe_bank.get(label, [])
        sample_size = min(len(questions), PROBE_SET_SIZE_PER_CLASS)
        probe_set[label] = questions[:sample_size]
    return probe_set


def _normalize_item(item: dict, idx: int, split: str) -> dict:
    """标准化一条原始数据集记录为系统内部题目格式。

    Args:
        item: 原始记录
        idx: 索引
        split: 数据集分片名

    Returns:
        标准化题目字典
    """
    safe_id = BENCHMARK_DATASET_ID.replace("/", "_")
    question_text = str(item.get(BENCHMARK_QUESTION_KEY, item.get("question", item.get("input", item.get("problem", "")))))
    gold_answer = str(item.get(BENCHMARK_ANSWER_KEY, item.get("answer", item.get("output", item.get("target", "")))))
    subject = str(item.get("subject", item.get("category", "")))
    if not subject:
        from src.tools.dataset_bank import infer_module
        subject = infer_module(question_text)
    # Code-domain fields: executable test + entry point, carried through the
    # probe/frozen sets so the evaluator can execute candidate code.
    test_code = str(item.get(BENCHMARK_TEST_KEY, "") or "") if IS_CODE_DOMAIN else ""
    entry_point = str(item.get(BENCHMARK_ENTRY_POINT_KEY, "") or "") if IS_CODE_DOMAIN else ""
    evaluation_method = "code_exec" if IS_CODE_DOMAIN else "gold"
    return {
        "question_id": f"{safe_id}_{split}_{idx}",
        "question_text": question_text,
        "gold_answer": gold_answer,
        "rollout_gold_answer": gold_answer,
        "train_output": gold_answer,
        "target_style": "answer",
        "evaluation_method": evaluation_method,
        "needs_judge": False,
        "test": test_code,
        "entry_point": entry_point,
        "source_dataset_id": f"{BENCHMARK_DATASET_ID}/{split}",
        "source_dataset_row_id": str(idx),
        "source_dataset_split": split,
        "source_dataset_subset": None,
        "source_dataset_requested_split": split,
        "module": subject,
        "dynamic_difficulty": str(item.get("level", "")),
    }


def _build_benchmark_holdout_probe_set(dataset_id: str, subset: str, split: str) -> list[dict]:
    """构建非分层的 global probe set，直接取前 N 题。

    Args:
        dataset_id: HuggingFace 数据集 ID
        subset: 子集名
        split: 分片名

    Returns:
        扁平题目列表，长度不超过 GLOBAL_PROBE_SIZE
    """
    from src.tools.dataset_adapter import load_hf_dataset_with_fallback

    dataset = load_hf_dataset_with_fallback(dataset_id, subset, split)
    questions: list[dict] = []
    for idx, item in enumerate(dataset):
        q = _normalize_item(item, idx, split)
        if q["question_text"] and q["gold_answer"]:
            questions.append(q)
        if len(questions) >= GLOBAL_PROBE_SIZE:
            break

    print(
        "[bootstrap] Built holdout global probe "
        f"dataset={dataset_id} split={split} size={len(questions)}"
    )
    return questions


def build_global_probe_set(trace_id: str) -> str:
    """构建全局探针集，写入 session 目录，返回文件路径。

    两种来源（由 GLOBAL_PROBE_SOURCE 配置控制）：
    - "benchmark_holdout"：从 BENCHMARK_DATASET_ID 的 holdout 分片抽取
    - 其他：使用 seed 题库

    分层模式（GLOBAL_PROBE_STRATIFIED_BY_MODULE）：
    - True：通过 load_dataset_bank → build_probe_from_bank 做 module-aware 轮询采样
    - False：直接取前 GLOBAL_PROBE_SIZE 题

    Args:
        trace_id: 会话标识

    Returns:
        探针集 JSON 文件路径
    """
    session_dir = get_session_dir(trace_id)
    probe_set_path = session_dir / "global_probe_set.json"

    if mathbench_probe_enabled(FROZEN_PROBE_EVAL_METHOD):
        with open(probe_set_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(
            "[bootstrap] Skipping local global probe construction because "
            f"frozen probe evaluation uses {FROZEN_PROBE_EVAL_METHOD}"
        )
        return str(probe_set_path)

    if GLOBAL_PROBE_SOURCE == "benchmark_holdout":
        try:
            from config.settings import BENCHMARK_DATASET_ID, BENCHMARK_SUBSET, GLOBAL_PROBE_SPLIT
            if GLOBAL_PROBE_STRATIFIED_BY_MODULE:
                bank_rows = load_dataset_bank(session_dir, dataset_id=BENCHMARK_DATASET_ID, subset=BENCHMARK_SUBSET, split=GLOBAL_PROBE_SPLIT)
                probe_set = build_probe_from_bank(bank_rows)
                selected_count = sum(len(v) for v in probe_set.values())
                print(
                    "[bootstrap] Built holdout global probe by module and legacy bucket "
                    f"size={selected_count} buckets={ {k: len(v) for k, v in probe_set.items()} }"
                )
            else:
                probe_set = _build_benchmark_holdout_probe_set(BENCHMARK_DATASET_ID, BENCHMARK_SUBSET, GLOBAL_PROBE_SPLIT)
        except Exception as exc:
            print(f"[bootstrap] Benchmark holdout probe failed ({type(exc).__name__}: {exc}), using seed probe")
            probe_set = _build_seed_probe_set()
    else:
        probe_set = _build_seed_probe_set()

    with open(probe_set_path, "w", encoding="utf-8") as f:
        json.dump(probe_set, f, ensure_ascii=False, indent=2)

    return str(probe_set_path)


def build_frozen_probe_set(trace_id: str, global_probe_path: str) -> str:
    """构建固定基准探针集（frozen probe set），用于评估旧能力保持。

    三种来源：
    - "benchmark_eval"：从基准数据集的 eval 分片 module-stratified 采样
    - 已有 frozen_path 缓存 → 直接返回
    - 否则：从 global probe set 中取前 FROZEN_PROBE_SIZE 题

    Args:
        trace_id: 会话标识
        global_probe_path: 全局探针集路径（fallback 数据源）

    Returns:
        frozen probe set 文件路径
    """
    session_dir = get_session_dir(trace_id)
    if mathbench_probe_enabled(FROZEN_PROBE_EVAL_METHOD):
        return build_mathbench_probe_marker(session_dir / "mathbench_frozen_probe.jsonl")

    frozen_path = session_dir / "fixed_benchmark_test.jsonl"
    if frozen_path.exists():
        return str(frozen_path)

    if FIXED_BENCHMARK_SOURCE == "benchmark_eval":
        try:
            from config.settings import BENCHMARK_DATASET_ID, BENCHMARK_EVAL_SPLIT, BENCHMARK_SUBSET
            from src.tools.dataset_adapter import load_hf_dataset_with_fallback

            dataset = load_hf_dataset_with_fallback(
                BENCHMARK_DATASET_ID,
                BENCHMARK_SUBSET,
                BENCHMARK_EVAL_SPLIT,
            )
            rows = []
            for idx, item in enumerate(dataset):
                q = _normalize_item(item, idx, BENCHMARK_EVAL_SPLIT)
                if q["question_text"] and q["gold_answer"]:
                    rows.append(q)

            from scripts.build_external_probe import _select_probe, DEFAULT_MODULE_WEIGHTS

            selected = _select_probe(rows, len(rows), DEFAULT_MODULE_WEIGHTS, seed=FIXED_BENCHMARK_SEED)
            with open(frozen_path, "w", encoding="utf-8") as f:
                for row in selected:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[bootstrap] Built fixed benchmark path={frozen_path} size={len(selected)}")
            return str(frozen_path)
        except Exception as exc:
            print(f"[bootstrap] Fixed benchmark failed ({type(exc).__name__}: {exc}), using global probe fallback")

    frozen_path = session_dir / "probe_frozen_set.json"
    if frozen_path.exists():
        return str(frozen_path)

    questions = _load_probe_questions(global_probe_path)
    frozen_questions = questions[: max(0, FROZEN_PROBE_SIZE)]

    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(frozen_questions, f, ensure_ascii=False, indent=2)

    print(
        "[bootstrap] Frozen trusted probe "
        f"size={len(frozen_questions)}"
    )
    return str(frozen_path)


# ── 基准评估 ────────────────────────────────────────────────

def _evaluate_base_model_frozen(
    champion_model_path: str, frozen_probe_path: str, trace_id: str, checkpoint: dict,
) -> float | None:
    """评估基础模型在 frozen probe 上的 1-shot 和 3-shot 表现。

    1-shot 结果作为 base_frozen_error_rate 供后续 evaluator 计算 forgetting score。
    checkpoint 中已有值时跳过评估（避免重复）。

    Args:
        champion_model_path: champion 模型路径
        frozen_probe_path: frozen probe set 路径
        trace_id: 会话标识
        checkpoint: 检查点字典

    Returns:
        1-shot 错误率（float），无数据时返回 None
    """
    if mathbench_probe_enabled(FROZEN_PROBE_EVAL_METHOD):
        result = run_mathbench_probe(
            champion_model_path,
            trace_id=trace_id,
            round_id=0,
            model_role="base",
        )
        error_rate = 1.0 - result.score
        print(
            "[bootstrap] Base model MathBench frozen probe: "
            f"{result.metric_name}={result.score:.4f}, error_rate={error_rate:.4f}"
        )
        return error_rate

    if checkpoint.get("base_frozen_error_rate") is not None:
        return float(checkpoint["base_frozen_error_rate"])
    frozen_prompts, frozen_gold, _questions = _load_probe_items_for_eval(frozen_probe_path)
    if not frozen_prompts:
        return None
    predictions = warmup_and_eval_batch(champion_model_path, frozen_prompts)
    correct = sum(1 for p, g, q in zip(predictions, frozen_gold, _questions) if judge_prediction_for_item(p, q, g))
    error_rate = 1.0 - correct / len(frozen_gold) if frozen_gold else None
    print(f"[bootstrap] Base model frozen probe 1-shot: {correct}/{len(frozen_gold)} correct, error_rate={error_rate:.4f}")

    from src.nodes.evaluator import _run_frozen_multi_rollout
    frozen_3shot = _run_frozen_multi_rollout(frozen_prompts, frozen_gold, champion_model_path)
    if frozen_3shot:
        total = frozen_3shot["total"]
        print(
            f"[bootstrap] Base model frozen 3-shot ({total}q): "
            f"3/3={frozen_3shot['correct_3of3']} "
            f"2/3={frozen_3shot['correct_2of3']} "
            f"1/3={frozen_3shot['correct_1of3']} "
            f"0/3={frozen_3shot['correct_0of3']} "
            f"pass@1={frozen_3shot['pass_at_1']:.3f}"
        )

    return error_rate


def _load_probe_items_for_eval(probe_path: str) -> tuple[list[str], list[str], list[dict]]:
    """加载探针集数据为评估用的 (prompts, golds, questions) 三元组。

    封装 evaluator._load_probe_items，避免模块级循环导入。
    传 max_items=0 表示不设上限，由数据集实际大小决定。
    """
    from src.nodes.evaluator import _load_probe_items
    return _load_probe_items(probe_path, max_items=0)


def warmup_and_eval_batch(model_path: str, prompts: list[str]) -> list[str]:
    """批量推理包装：预热模型后执行批量推理。

    Args:
        model_path: 模型路径
        prompts: 输入 prompt 列表

    Returns:
        模型生成结果列表
    """
    from src.tools.model_runner import run_model_batch
    return run_model_batch(
        model_path,
        prompts,
        max_new_tokens=int(EVAL_MAX_NEW_TOKENS),
        disable_thinking=True,
    )


# ── 探针文件加载 ────────────────────────────────────────────

def _load_probe_questions(probe_path: str) -> list[dict]:
    """从探针集文件中加载题目列表，兼容 dict 和 list 两种格式。

    Args:
        probe_path: 探针集 JSON 文件路径

    Returns:
        扁平题目列表
    """
    path = Path(probe_path)
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return _load_probe_questions_jsonl(probe_path)
    with open(path, "r", encoding="utf-8") as f:
        probe_data = json.load(f)
    if isinstance(probe_data, dict):
        questions = []
        for items in probe_data.values():
            if isinstance(items, list):
                questions.extend(q for q in items if isinstance(q, dict))
        return questions
    if isinstance(probe_data, list):
        return [q for q in probe_data if isinstance(q, dict)]
    return []


def _load_probe_questions_jsonl(probe_path: str) -> list[dict]:
    """从 JSONL 文件中逐行加载探针题目。

    Args:
        probe_path: JSONL 文件路径

    Returns:
        题目列表
    """
    if not probe_path:
        return []
    path = Path(probe_path)
    if not path.exists() or path.is_dir():
        return []
    questions = []
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
                questions.append(item)
    return questions


# ── Holdout 注册 ────────────────────────────────────────────

def _load_previous_round_heldouts(session_dir: Path) -> list[dict]:
    """加载历史轮次的 holdout 题目列表。

    从 session 目录下所有 round_*_datasets/round_heldout_questions.json 中提取。

    Args:
        session_dir: 当前 session 目录

    Returns:
        所有历史 holdout 题目列表
    """
    heldouts: list[dict] = []
    for path in sorted(session_dir.glob("round_*_datasets/round_heldout_questions.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, list):
            heldouts.extend(q for q in loaded if isinstance(q, dict))
    return heldouts


# ── 主入口 ──────────────────────────────────────────────────

def bootstrap_node(state: EvoState) -> dict:
    """系统启动 / 恢复的 LangGraph 节点。

    执行流程：
    1. 确定 trace_id、session_dir
    2. 加载 checkpoint / DAG / replay buffer
    3. 推断 round_id、data_window、champion 模型路径
    4. 构建 global probe set → frozen probe set
    5. 将探针题注册到 question_registry（防泄露）
    6. 评估 base model 基线（frozen probe 1-shot + 3-shot）
    7. 组装完整 State 并发送 GOAL_REQUEST → TEACHER

    Args:
        state: LangGraph 输入 State

    Returns:
        初始化后的 State 更新字典
    """
    trace_id = state.get("trace_id") or EVOLVE_TRACE_ID or str(uuid.uuid4())[:8]
    session_dir = get_session_dir(trace_id)

    # 1. 恢复持久化状态
    checkpoint = _load_evolution_checkpoint(trace_id)
    loaded_nodes, loaded_edges, loaded_current_node_id = load_search_dag(session_dir)
    loaded_replay_entries = load_replay_buffer(session_dir)

    # 2. 恢复或创建 DAG
    has_loaded_dag = bool(loaded_nodes)
    if has_loaded_dag:
        search_dag_nodes = loaded_nodes
        search_dag_edges = loaded_edges
        current_search_node_id = loaded_current_node_id or loaded_nodes[-1].get("node_id", "")
    else:
        root_node = create_root_node(trace_id)
        search_dag_nodes = [root_node.model_dump()]
        search_dag_edges = []
        current_search_node_id = root_node.node_id

    # 3. 推断当前轮次
    round_id = state.get("round_id")
    if round_id is None:
        round_id = int(checkpoint.get("next_round_id", _infer_next_round_id(search_dag_edges)) or 0)

    # 4. 确定 champion 模型路径
    champion_model_path = (
        state.get("champion_model_path")
        or checkpoint.get("champion_model_path")
        or _infer_latest_promoted_champion(trace_id, search_dag_edges)
        or CHAMPION_MODEL_PATH
    )
    warmup_model(champion_model_path)

    # 6. 构建探针集
    probe_path = build_global_probe_set(trace_id)
    heldout_registry_path = str(session_dir / "heldout_registry.json")
    frozen_probe_path = build_frozen_probe_set(trace_id, probe_path)

    external_probe_path = EXTERNAL_PROBE_PATH
    holdout_eval_path = checkpoint.get("holdout_eval_path", "")
    if not holdout_eval_path:
        candidate_holdout_path = session_dir / "holdout_eval.json"
        holdout_eval_path = str(candidate_holdout_path) if candidate_holdout_path.exists() else ""

    # 7. 评估 base model 基线
    base_frozen_error_rate = _evaluate_base_model_frozen(
        champion_model_path, frozen_probe_path, trace_id, checkpoint,
    )

    # 8. 注册探针题到 question_registry（防止数据泄露）
    mark_questions_probe_holdout(heldout_registry_path, _load_probe_questions(probe_path))
    external_probe_questions = _load_probe_questions_jsonl(external_probe_path)
    if external_probe_questions:
        mark_questions_external_probe(
            heldout_registry_path,
            external_probe_questions,
            metadata={
                "reason": "external_probe_benchmark",
                "manifest_path": EXTERNAL_PROBE_MANIFEST_PATH,
            },
        )
        print(
            "[bootstrap] Registered external probe holdout: "
            f"path={external_probe_path} count={len(external_probe_questions)}"
        )
    if holdout_eval_path:
        mark_questions_active_holdout(
            heldout_registry_path,
            _load_probe_questions(holdout_eval_path),
            metadata={"reason": "stable_holdout_eval"},
        )
    previous_heldouts = _load_previous_round_heldouts(session_dir)
    if previous_heldouts:
        mark_questions_active_holdout(heldout_registry_path, previous_heldouts)
        print(f"[bootstrap] Restored prior round heldouts into registry: count={len(previous_heldouts)}")

    # 9. 发出首条消息：GOAL_REQUEST → TEACHER
    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.SYSTEM,
            receiver=AgentName.TEACHER,
            message_type=MessageType.GOAL_REQUEST,
        ),
        payload=GoalRequestPayload(goal=str(state.get("user_goal", ""))),
    )

    return {
        "trace_id": trace_id,
        "round_id": round_id,
        "global_probe_set_path": probe_path,
        "probe_frozen_set_path": frozen_probe_path,
        "frozen_probe_eval_method": FROZEN_PROBE_EVAL_METHOD,
        "external_probe_path": external_probe_path,
        "holdout_eval_path": holdout_eval_path,
        "pending_message": msg,
        "champion_model_path": champion_model_path,
        "current_search_node_id": current_search_node_id,
        "search_dag_nodes": search_dag_nodes,
        "search_dag_edges": search_dag_edges,
        "replay_buffer_entries": loaded_replay_entries,
        "replay_sample_ratio_override": checkpoint.get("replay_sample_ratio_override", 0.30),
        "round_data_stats": checkpoint.get("round_data_stats", {}),
        "rollout_difficulty_distribution": checkpoint.get("rollout_difficulty_distribution", {}),
        "rollout_hard_ratio": float(checkpoint.get("rollout_hard_ratio", 0.0) or 0.0),
        "rollout_pass_count_histogram": checkpoint.get("rollout_pass_count_histogram", {}),
        "difficulty_threshold_policy": checkpoint.get("difficulty_threshold_policy", {}),
        "last_inspection_decision": checkpoint.get("last_inspection_decision", ""),
        "kept_branch_decision": checkpoint.get("kept_branch_decision", ""),
        "kept_branch_node_id": checkpoint.get("kept_branch_node_id", ""),
        "rollback_streak": int(checkpoint.get("rollback_streak", 0) or 0),
        "diagnostic_mode": "train",
        "probe_diagnostic_path": checkpoint.get("probe_diagnostic_path", ""),
        "probe_diagnostic_focus_path": checkpoint.get("probe_diagnostic_focus_path", ""),
        "target_bucket": checkpoint.get("target_bucket", ""),
        "last_dataset_bundle": checkpoint.get("last_dataset_bundle", {}),
        "last_dataset_bundle_state": checkpoint.get("last_dataset_bundle_state", {}),
        "last_dataset_bundle_round_id": checkpoint.get("last_dataset_bundle_round_id"),
        "last_attempt_question_ids": checkpoint.get("last_attempt_question_ids", []),
        "last_attempt_reserved_dataset_question_ids": checkpoint.get("last_attempt_reserved_dataset_question_ids", {}),
        "mastered_memory_set_path": checkpoint.get("mastered_memory_set_path", ""),
        "champion_holdout_baseline": checkpoint.get("champion_holdout_baseline"),
        "champion_frozen_probe_error": checkpoint.get("champion_frozen_probe_error"),
        "base_frozen_error_rate": base_frozen_error_rate,
        "heldout_registry_path": heldout_registry_path,
        "current_window_offset": 0,
        "current_window_size": int(DATA_WINDOW_SIZE or 0),
        "windows_loaded_this_round": 0,
        "profile_items_loaded_this_round": 0,
        "max_windows_per_round": max(1, int(MAX_PROFILE_WINDOWS_PER_ROUND or 1)),
        "max_profile_items_per_round": max(1, int(MAX_PROFILE_ITEMS_PER_ROUND or DATA_WINDOW_SIZE or 1)),
        "quota_met": True,
        "quota_shortfall": {},
        "quota_accumulated_questions": [],
        "data_replenishment_needed": False,
        "data_replenishment_exhausted": False,
        "replenishment_cycle_active": False,
        "replenishment_attempted_windows": {},
        "replenishment_attempted_windows_round_id": -1,
        "next_dataset_ref": {},
        "dataset_review_pending_refs": [],
        "dataset_review_job_id": "",
        "dataset_review_active": False,
        "dataset_review_completed": True,
        "dataset_review_drained_count": 0,
        "dataset_states_path": str(session_dir / "dataset_states"),
    }
