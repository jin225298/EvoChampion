"""
数据集题库管理工具（Dataset Bank）

核心职责（自底向上分层接口）：
1. 从 HuggingFace 加载基准数据集，标准化并标注模块，缓存为 bank.json
2. 提供关键词规则推断题目所属模块（ratio_rate / money_cost / geometry_measurement 等），
   作为 GLiNER 模型不可用时的 fallback 分类器
3. 从已分类的题库中按 module 轮询构建全局探针集（global probe set），
   用于评估学生 agent 的能力

数据流：
  HF dataset → build_dataset_bank → bank.json（缓存）
  load_dataset_bank → 分类后的行列表 → build_probe_from_bank → global_probe_set

层次关系：
  本模块是机制层工具，向上层（bootstrap / classifier / screening_entry）提供通用接口。
  不包含任何数据集特化逻辑（如 gsm8k_bank.py），调用方通过参数指定 dataset_id。
"""

import json
import random
import re
from pathlib import Path

from config.settings import (
    BENCHMARK_ANSWER_KEY,
    BENCHMARK_DATASET_ID,
    BENCHMARK_HOLDOUT_OFFSET,
    BENCHMARK_QUESTION_KEY,
    BENCHMARK_SPLIT,
    BENCHMARK_SUBSET,
    GLOBAL_PROBE_SIZE,
)
from src.tools.dataset_adapter import load_hf_dataset_with_fallback

# 模块关键词映射：通过题目文本中出现的关键词推断数学题型
_MODULE_KEYWORDS = {
    "ratio_rate": ["per", "each", "every", "rate", "miles per", "km/h", "hour", "minute", "second"],
    "money_cost": ["$", "dollar", "cost", "price", "rent", "paid", "buy", "sold", "profit", "percent off"],
    "counting_combinatorics": ["how many", "ways", "arrangements", "combinations", "permutations"],
    "geometry_measurement": ["area", "perimeter", "rectangle", "triangle", "circle", "radius", "volume", "length", "width", "height"],
    "fraction_percent": ["fraction", "%", "percent", "half", "third", "quarter", "ratio"],
    "work_time": ["finish", "complete", "together", "alone", "work", "days", "hours"],
    "age_numbers": ["older", "younger", "age", "years old", "sum of", "difference"],
}

# 用于检测题目文本中是否包含数字（兜底判断是否为算术题）
_NUM_PATTERN = re.compile(r"\d")


def infer_module(question_text: str) -> str:
    """根据题目文本中的关键词推断所属数学题型模块。

    使用 _MODULE_KEYWORDS 中预定义的规则集进行关键词匹配，
    得分最高的模块即为推断结果。若无任何关键词命中但包含数字，
    则归类为 arithmetic_misc；否则返回 unknown。

    此函数是 GLiNER 模型不可用时的 fallback 分类器，
    也可作为 screening_entry 中为题目预标注 module 字段的默认方式。

    Args:
        question_text: 题目文本

    Returns:
        模块名（如 "ratio_rate", "geometry_measurement", "unknown" 等）
    """
    text = (question_text or "").lower()
    if not text:
        return "unknown"
    scores: dict[str, int] = {}
    for module, keywords in _MODULE_KEYWORDS.items():
        scores[module] = sum(1 for kw in keywords if kw in text)
    best_module = max(scores, key=lambda k: scores[k]) if scores else "unknown"
    if scores.get(best_module, 0) == 0:
        if _NUM_PATTERN.search(text):
            return "arithmetic_misc"
        return "unknown"
    return best_module


def _normalize_item(item: dict, idx: int, split: str) -> dict:
    """将一条原始数据集记录标准化为题库条目。

    兼容多种数据字段名（question/input/problem, answer/output/target）。

    Args:
        item: 原始数据集中的一条记录
        idx: 该记录在数据集中的索引
        split: 数据集分片名（如 train / test）

    Returns:
        标准化后的题目字典，包含 question_id, question_text, gold_answer, source_dataset_id
    """
    question_text = str(
        item.get(BENCHMARK_QUESTION_KEY)
        or item.get("question")
        or item.get("input")
        or item.get("problem")
        or ""
    )
    gold_answer = str(
        item.get(BENCHMARK_ANSWER_KEY)
        or item.get("answer")
        or item.get("output")
        or item.get("target")
        or ""
    )
    # Preserve code-domain test + entry_point so execution judging can run the
    # candidate against the reference test (bank/probe rows included).
    from src.tools.code_execution import extract_code_test_fields
    _test_code, _entry_point, _ = extract_code_test_fields(item)
    _has_code_test = bool(_test_code and _entry_point)
    return {
        "question_id": f"{BENCHMARK_DATASET_ID.replace('/', '_')}_{split}_{idx}",
        "question_text": question_text,
        "gold_answer": gold_answer,
        "rollout_gold_answer": gold_answer,
        "train_output": "",
        "target_style": "answer",
        "source_dataset_id": f"{BENCHMARK_DATASET_ID}/{split}",
        "source_dataset_row_id": str(idx),
        "test": _test_code,
        "entry_point": _entry_point,
        "evaluation_method": "code_execution" if _has_code_test else "gold",
        "needs_judge": False,
    }


def build_dataset_bank(session_dir: Path, dataset_id: str = None, subset: str = None, split: str = None) -> Path:
    """构建题库文件（如有缓存则直接返回）。

    从 HuggingFace 加载指定数据集，将每条记录标准化后标注：
    - module：题型模块（通过 infer_module 关键词推断）
    - partition：probe_holdout（holdout 偏移量之后）或 train_pool（之前）
    写入 session_dir 下的 bank.json 作为缓存。

    Args:
        session_dir: 当前 session 目录
        dataset_id: HuggingFace 数据集 ID，默认取 BENCHMARK_DATASET_ID
        subset: 数据集子集名，默认取 BENCHMARK_SUBSET
        split: 数据集分片名，默认取 BENCHMARK_SPLIT

    Returns:
        题库 JSON 文件路径 (Path)
    """
    ds_id = dataset_id or BENCHMARK_DATASET_ID
    ds_subset = subset or BENCHMARK_SUBSET
    ds_split = split or BENCHMARK_SPLIT
    bank_path = session_dir / f"{ds_id.replace('/', '_')}_bank.json"
    if bank_path.exists():
        return bank_path

    dataset = load_hf_dataset_with_fallback(ds_id, ds_subset, ds_split)
    rows: list[dict] = []
    for idx, item in enumerate(dataset):
        q = _normalize_item(item, idx, ds_split)
        if not q["question_text"] or not q["gold_answer"]:
            continue
        module = infer_module(q["question_text"])
        holdout_offset = BENCHMARK_HOLDOUT_OFFSET
        partition = "probe_holdout" if idx >= holdout_offset else "train_pool"
        rows.append({**q, "module": module, "partition": partition, "bank_index": idx})

    with open(bank_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return bank_path


def load_dataset_bank(session_dir: Path, dataset_id: str = None, subset: str = None, split: str = None) -> list[dict]:
    """加载题库数据（自动构建或从缓存读取）。

    先调用 build_dataset_bank 确保题库文件存在，再返回所有有效题目。

    Args:
        session_dir: 当前 session 目录
        dataset_id: HuggingFace 数据集 ID
        subset: 数据集子集名
        split: 数据集分片名

    Returns:
        已分类的题库行列表，每行包含 module, partition 等字段
    """
    path = build_dataset_bank(session_dir, dataset_id=dataset_id, subset=subset, split=split)
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return [r for r in loaded if isinstance(r, dict)]


def build_probe_from_bank(rows: list[dict]) -> dict[str, list[dict]]:
    """从已分类的题库行中按 module 轮询构建全局探针集。

    探针集用于每轮训练后评估模型在未见题上的准确率。
    构建策略：在 module 间做 round-robin 采样，保证探针集在题型维度上均衡覆盖。

    Args:
        rows: 已分类的题库行列表，每行需包含:
              - partition: "probe_holdout" 表示属于探针候选池
              - module: 题型模块（如 "geometry_measurement"）
              - question_id / question_text / gold_answer / source_dataset_id

    Returns:
        {module_name: [question_dict, ...]} 格式的探针集字典
    """
    candidates = [r for r in rows if r.get("partition") == "probe_holdout"]
    by_module: dict[str, list[dict]] = {}
    for row in candidates:
        m = row.get("module", "unknown")
        by_module.setdefault(m, []).append(row)

    rng = random.Random(20260509)
    for pool in by_module.values():
        rng.shuffle(pool)

    selected: list[dict] = []
    modules = sorted(by_module.keys()) or ["unknown"]

    idx = 0
    while len(selected) < GLOBAL_PROBE_SIZE and modules:
        m = modules[idx % len(modules)]
        pool = by_module.get(m, [])
        if pool:
            selected.append(pool.pop())
        else:
            modules = [mod for mod in modules if by_module.get(mod)]
            if not modules:
                break
        idx += 1

    grouped: dict[str, list[dict]] = {}
    for row in selected:
        grouped.setdefault(row.get("module", "unknown"), []).append({
            "question_id": row.get("question_id", ""),
            "question_text": row.get("question_text", ""),
            "gold_answer": row.get("gold_answer", ""),
            "rollout_gold_answer": row.get("rollout_gold_answer") or row.get("gold_answer", ""),
            "train_output": row.get("train_output", ""),
            "target_style": row.get("target_style", "answer"),
            "source_dataset_id": row.get("source_dataset_id", f"{BENCHMARK_DATASET_ID}/{BENCHMARK_SPLIT}"),
            "source_dataset_row_id": row.get("source_dataset_row_id"),
            "module": row.get("module", "unknown"),
        })
    return grouped
