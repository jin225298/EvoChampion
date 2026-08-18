"""
经验回放池（Replay Buffer）—— 跨轮次题目复用与灾难性遗忘防护

核心职责：
1. 存储历史上成功训练过的题目（含难度评分、模块标签、使用轮次等元信息）
2. 每轮训练时按动态难度 + 最近最少使用（LRU）策略从中采样，与新题目混合，
   防止模型在专注新能力时遗忘旧能力（catastrophic forgetting）
3. 提供创建、加载、保存、采样、添加的全套 CRUD 接口

采样优先级（sample_from_bucket）：
  1. 从未被采样过的目标难度题目 → 优先探索新回放池内容
  2. 从未被采样过的其他难度题目 → 兜底未使用的回放条目
  3. 使用次数最少 + 最久未用的条目 → 已用过但值得复习的题目

Agent 可调参数：
  - sample_ratio: 通过 state["replay_sample_ratio_override"] 覆盖默认采样比例
  - target_bucket: 通过 state["target_bucket"] 指定优先采样的动态难度桶
  其余逻辑（排序规则、优先级顺序、max_size）均为硬编码机制。

使用场景：
  rollout_aggregator → create_replay_entry → 将全错题目加入回放池
  strategy_inspector → save_replay_buffer → 决策后持久化回放池
  filter → sample_from_bucket → 混合回放题目到训练数据
  bootstrap → load_replay_buffer → 恢复历史回放池
"""

import json
import random
from pathlib import Path
from uuid import uuid4

from src.models.messages import ReplayBufferEntryPayload
from src.tools.question_fields import infer_target_style


def create_replay_entry(
    dataset_signature: str,
    source_round: int,
    success_score: float,
    uniqueness_score: float = 1.0,
    difficulty_score: float = 0.5,
    coverage_score: float = 0.5,
    question_id: str = "",
    question_text: str = "",
    gold_answer: str = "",
    rollout_gold_answer: str = "",
    train_output: str = "",
    target_style: str = "answer",
    evaluation_method: str = "gold",
    needs_judge: bool = False,
    source_dataset_id: str | None = None,
    source_dataset_row_id: str | None = None,
    source_dataset_split: str | None = None,
    source_dataset_subset: str | None = None,
    source_dataset_requested_split: str | None = None,
    source_dataset_split_names: list[str] | None = None,
    source_dataset_columns: list[str] | None = None,
    source_dataset_first_row: dict | None = None,
    source_dataset_schema: dict | None = None,
    dynamic_difficulty: str = "unknown",
    origin_round_id: int | None = None,
    pass_count: int | None = None,
    rollout_count: int | None = None,
    pass_rate: float | None = None,
    bucket: str = "unknown",
    module: str = "unknown",
    used_in_rounds: list[int] | None = None,
) -> ReplayBufferEntryPayload:
    """创建一个回放池条目。

    Args:
        dataset_signature: 数据来源标识（如 "gsm8k/train"）
        source_round: 题目首次出现的轮次
        success_score: 成功率评分（越高越优先保留）
        uniqueness_score: 唯一性评分
        difficulty_score: 难度评分
        coverage_score: 覆盖度评分
        question_id: 题目 ID
        question_text: 题目文本
        gold_answer: 标准答案
        rollout_gold_answer: rollout/eval 判题用最终答案
        train_output: 训练监督输出，可为 CoT/完整解答
        target_style: 训练目标格式，只允许 answer 或 cot
        evaluation_method: gold 表示用标准答案判题，llm_judge 表示用参考解法交给裁判判题
        needs_judge: 是否需要 LLM 裁判判题
        bucket: 动态难度桶（easy/medium/hard）
        module: 题型模块
        used_in_rounds: 已被采样的轮次列表

    Returns:
        ReplayBufferEntryPayload 实例
    """
    rollout_text = rollout_gold_answer or gold_answer
    target = target_style if target_style in {"answer", "cot"} else "answer"
    if target == "answer" and not train_output:
        train_output = rollout_text
    target = infer_target_style(train_output, rollout_text, gold_answer, explicit=target)
    return ReplayBufferEntryPayload(
        entry_id=f"replay_{uuid4().hex[:8]}",
        dataset_signature=dataset_signature,
        source_round=source_round,
        success_score=success_score,
        uniqueness_score=uniqueness_score,
        difficulty_score=difficulty_score,
        coverage_score=coverage_score,
        question_id=question_id,
        question_text=question_text,
        gold_answer=gold_answer,
        rollout_gold_answer=rollout_text,
        train_output=train_output,
        target_style=target,
        evaluation_method=evaluation_method if evaluation_method in {"gold", "llm_judge", "code_execution"} else "gold",
        needs_judge=bool(needs_judge or evaluation_method == "llm_judge"),
        source_dataset_id=source_dataset_id,
        source_dataset_row_id=source_dataset_row_id,
        source_dataset_split=source_dataset_split,
        source_dataset_subset=source_dataset_subset,
        source_dataset_requested_split=source_dataset_requested_split,
        source_dataset_split_names=source_dataset_split_names or [],
        source_dataset_columns=source_dataset_columns or [],
        source_dataset_first_row=source_dataset_first_row or {},
        source_dataset_schema=source_dataset_schema or {},
        dynamic_difficulty=dynamic_difficulty or bucket or "unknown",
        origin_round_id=origin_round_id if origin_round_id is not None else source_round,
        pass_count=pass_count,
        rollout_count=rollout_count,
        pass_rate=pass_rate,
        bucket=bucket,
        module=module,
        used_in_rounds=used_in_rounds if used_in_rounds is not None else [],
    )


def save_replay_buffer(entries: list[dict], session_dir: Path) -> None:
    """持久化回放池到 session 目录。

    Args:
        entries: 回放池条目列表
        session_dir: 当前 session 目录
    """
    buffer_path = session_dir / "replay_buffer.json"
    with open(buffer_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def load_replay_buffer(session_dir: Path) -> list[dict]:
    """从 session 目录加载回放池。

    Args:
        session_dir: 当前 session 目录

    Returns:
        回放池条目列表，不存在时返回空列表
    """
    buffer_path = session_dir / "replay_buffer.json"
    if not buffer_path.exists():
        return []
    with open(buffer_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_from_bucket(
    entries: list[dict],
    n: int,
    current_round: int,
    target_bucket: str = "",
    max_size: int = 1000,
) -> list[dict]:
    """按动态难度和最近使用情况从回放池采样。

    采样优先级（从高到低）：
    1. 目标难度桶中从未使用过的条目
    2. 其他难度桶中从未使用过的条目
    3. 使用次数最少 + 最久未用的剩余条目

    bucket 字段存的是 rollout 后的 dynamic_difficulty（easy/medium/hard）。
    target_bucket 由上层（state["target_bucket"]）控制，
    可由 agent 通过 strategy 动态指定。

    Args:
        entries: 回放池条目列表
        n: 采样数量
        current_round: 当前轮次（用于标记已使用）
        target_bucket: 优先采样的动态难度桶，空字符串表示不区分
        max_size: 回放池最大容量（硬编码，超出截断）

    Returns:
        采样后的条目列表，已标记 used_in_rounds
    """
    if not entries:
        return []

    # 按 question_id 去重，保留最后出现的版本
    seen_ids: set[str] = set()
    unique_entries: list[dict] = []
    for entry in reversed(entries):
        qid = entry.get("question_id", "")
        if qid and qid in seen_ids:
            continue
        seen_ids.add(qid)
        unique_entries.append(entry)

    unique_entries = unique_entries[:max_size]

    # 兼容旧回放池：旧条目可能缺少字段
    for entry in unique_entries:
        entry.setdefault("used_in_rounds", [])
        entry.setdefault("bucket", "unknown")
        entry.setdefault("module", "unknown")

    # 按动态难度分桶：目标桶优先，其他桶作为补充
    if target_bucket and target_bucket in ("easy", "medium", "hard", "unknown"):
        target_pool = [e for e in unique_entries if e.get("bucket") == target_bucket]
        other_pool = [e for e in unique_entries if e.get("bucket") != target_bucket]
    else:
        target_pool = unique_entries
        other_pool = []

    # 排序键：先按使用次数（少→多），再按最近使用轮次（旧→新）
    def _sort_key(entry: dict) -> tuple:
        used_rounds = entry.get("used_in_rounds", [])
        use_count = len(used_rounds)
        last_use = max(used_rounds) if used_rounds else -1
        return (use_count, last_use)

    target_pool.sort(key=_sort_key)
    other_pool.sort(key=_sort_key)

    # 分层采样：优先从未使用过的题目开始
    unused_target = [e for e in target_pool if not e.get("used_in_rounds")]
    unused_other = [e for e in other_pool if not e.get("used_in_rounds")]
    all_sorted = target_pool + other_pool

    selected: list[dict] = []

    # 优先级 1：目标桶未使用
    random.shuffle(unused_target)
    take = min(len(unused_target), n - len(selected))
    selected.extend(unused_target[:take])

    # 优先级 2：其他桶未使用
    if len(selected) < n:
        random.shuffle(unused_other)
        take = min(len(unused_other), n - len(selected))
        selected.extend(unused_other[:take])

    # 优先级 3：剩余条目按 LRU 排序补齐
    if len(selected) < n:
        already_selected_ids = {e.get("entry_id", e.get("question_id", "")) for e in selected}
        for entry in all_sorted:
            eid = entry.get("entry_id", entry.get("question_id", ""))
            if eid in already_selected_ids:
                continue
            selected.append(entry)
            if len(selected) >= n:
                break

    # 标记本轮已使用
    for entry in selected:
        rounds = entry.get("used_in_rounds", [])
        if current_round not in rounds:
            rounds.append(current_round)
            entry["used_in_rounds"] = rounds

    return selected[:n]


def add_to_replay_buffer(
    entries: list[dict],
    new_entry: dict,
    max_size: int = 1000,
) -> list[dict]:
    """添加新条目到回放池，按 success_score 降序保留 top-N。

    回放池容量有限（硬编码 max_size=1000），超出时淘汰低分条目。

    Args:
        entries: 当前回放池
        new_entry: 新条目
        max_size: 最大容量

    Returns:
        截断后的回放池
    """
    entries.append(new_entry)
    entries = sorted(entries, key=lambda e: e.get("success_score", 0), reverse=True)
    return entries[:max_size]
