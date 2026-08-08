"""
Filter Node — 训练题目筛选与分配节点。

职责：
1. Pre-rollout: 从 materialized 数据集中采样 candidate 题目，确定需要 rollout 的题目列表
2. Post-rollout: 根据 cascade rollout 标签筛选/补全题目，分配至 train/cotest/test/probe 等 split
3. 维护 cross-dataset pool，支持跨数据集补采样
4. 管理 replay buffer 与新数据的混合
5. 追踪 mastered 题目，避免重复训练已掌握的题目

核心流程：
  MaterializedDataset → [pre-filter] → RolloutRequest → [rollout] → RolloutResult → [post-filter] → FilterResult
"""

import json
import math
import random
import hashlib
import os
from importlib import import_module
from collections import Counter
from pathlib import Path

from config.settings import (
    COTEST_SPLIT_RATIO,            # cotest 占总 pool 的比例
    FILTER_DEDUPLICATE,             # 是否启用去重
    FILTER_DROP_MASTERED,           # 是否丢弃已掌握的题目
    FILTER_SAMPLING_METHOD,         # 采样方法: "stratified" / "random"
    FILTER_TARGET_QUESTIONS_PER_ROUND,  # 每轮最终目标题目数
    LF_VAL_SPLIT_RATIO,             # 逻辑验证器验证集比例
    MAX_PROFILE_ITEMS_PER_ROUND,    # 每轮最多加载的 profiling 题目数
    MAX_PROFILE_WINDOWS_PER_ROUND,  # 每轮最多加载的 dataset window 数
    MASTERED_CORRECT_RATIO,         # 判定"已掌握"的正确率阈值比例
    MASTERED_CORRECT_THRESHOLD,     # 判定"已掌握"的绝对正确次数阈值
    PROBE_POOL_INTAKE_RATIO,        # probe pool 在 split 时的摄入比例
    REPLAY_BUFFER_SAMPLE_RATIO,     # replay buffer 采样比例（相对于新题目数量）
    ROLLOUT_MAX_NEW_TOKENS,
    ROLLOUT_TEMPERATURE,
    ROLLOUT_TIMES,                  # 旧 mastered 阈值兼容；难度标签不再依赖它
    ROLLOUT_TOP_P,
    TARGET_BUCKET_RATIO,            # target bucket 偏置采样的比例
    TEST_SPLIT_RATIO,               # test pool 占总 pool 的比例
    TRAIN_LF_EVAL_ENABLED,          # 是否启用 LF val 拆分的逻辑验证器
    TRAIN_SPLIT_RATIO,              # train pool 占总 pool 的比例
)
from src.models.messages import (
    AgentName,
    FilterResultPayload,
    FilterStrategyPayload,
    MaterializedDatasetPayload,
    MessageHeader,
    MessageType,
    QuestionScore,
    RolloutRequestPayload,
    RolloutResultPayload,
    RoutedMessage,
)
from src.models.state import EvoState
from src.tools.question_registry import TRAIN_BLOCKING_STATUSES, drop_registered_questions, question_text_hash
from src.tools.replay_buffer import sample_from_bucket
from src.tools.message_artifacts import load_payload_list, write_json_artifact
from src.tools.dataset_state import ITEM_STATE_UNUSED, DatasetStateManager
from src.tools.question_fields import add_processed_question_fields, normalize_target_style

# 一道合法数学题目必须包含的字段
_MATH_QUESTION_REQUIRED_FIELDS = frozenset({"question_id", "question_text", "gold_answer"})
# 动态难度的合法取值集合
_DYNAMIC_DIFFICULTIES = {"easy", "medium", "hard", "unknown"}
_FALLBACK_SCHEMA_REASONS = {
    "fallback schema",
    "fallback flat schema",
    "fallback conversation schema",
}
ROLLOUT_CACHE_JUDGE_VERSION = "cascade-v1"


def _rollout_config_hash() -> str:
    payload = {
        "cascade": "answer_only_then_thinking",
        "max_new_tokens": int(ROLLOUT_MAX_NEW_TOKENS),
        "temperature": float(ROLLOUT_TEMPERATURE),
        "top_p": float(ROLLOUT_TOP_P),
        "instruction_prefix_sha1": hashlib.sha1(
            os.getenv("INSTRUCTION_PREFIX", "").encode("utf-8")
        ).hexdigest(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _has_forbidden_fallback_schema(question: dict) -> bool:
    schema = question.get("source_dataset_schema") if isinstance(question, dict) else None
    if not isinstance(schema, dict):
        return False
    reason = str(schema.get("reason") or "").strip().lower()
    return reason in _FALLBACK_SCHEMA_REASONS or schema.get("usable") is False


def _drop_forbidden_fallback_schema_questions(questions: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    dropped = 0
    for question in questions:
        if _has_forbidden_fallback_schema(question):
            dropped += 1
            continue
        kept.append(question)
    return kept, dropped


def _replenishment_seen_state(state: EvoState) -> dict:
    """追踪当前轮次已 seen/loaded/attempted 的 dataset window，用于 replenishment 状态管理。

    维护三类 window 记录：
    - cached_replenishment_windows_seen: 当前轮已查看过的 window
    - replenishment_loaded_windows: 当前轮已加载的 window
    - replenishment_attempted_windows: 当前轮已尝试加载的 window
    每个记录都关联 round_id，确保跨轮重置。
    """
    round_id = int(state.get("round_id", 0) or 0)

    def _same_round(field_name: str) -> bool:
        """检查 state 中某字段的值是否等于当前 round_id。"""
        try:
            return int(state.get(field_name, -1) or -1) == round_id
        except (TypeError, ValueError):
            return False

    def _parse_window_map(raw: object, round_field: str) -> dict[str, list[int]]:
        """解析原始 window map 为 {dataset_id: [sorted_offsets]} 格式。

        仅当 round_field 指向的轮次与当前轮一致时才返回有效数据，
        否则返回空 dict（跨轮自动重置）。
        """
        if not _same_round(round_field):
            return {}
        if not isinstance(raw, dict):
            return {}
        parsed_map: dict[str, list[int]] = {}
        for dataset_id, offsets in raw.items():
            if not isinstance(offsets, list):
                continue
            parsed: set[int] = set()
            for offset in offsets:
                try:
                    parsed.add(int(offset))
                except (TypeError, ValueError):
                    continue
            if parsed:
                parsed_map[str(dataset_id)] = sorted(parsed)
        return parsed_map

    def _mark_current_window(target: dict[str, list[int]]) -> None:
        """将当前活跃 window 的 offset 标记到指定的 map 中。"""
        current_window = (state.get("round_data_stats") or {}).get("window", {})
        if not isinstance(current_window, dict):
            return
        dataset_id = str(current_window.get("dataset_id") or state.get("active_dataset_id") or "")
        if not dataset_id:
            return
        try:
            offset = int(current_window.get("offset", state.get("current_window_offset", 0)) or 0)
        except (TypeError, ValueError):
            return
        offsets = set(target.get(dataset_id, []))
        offsets.add(max(0, offset))
        target[dataset_id] = sorted(offsets)

    raw_seen = state.get("cached_replenishment_windows_seen") or {}
    seen = _parse_window_map(raw_seen, "cached_replenishment_windows_seen_round_id")
    _mark_current_window(seen)
    update: dict = {"cached_replenishment_windows_seen": seen}
    update["cached_replenishment_windows_seen_round_id"] = round_id
    raw_loaded = state.get("replenishment_loaded_windows") or {}
    loaded = _parse_window_map(raw_loaded, "replenishment_loaded_windows_round_id")
    _mark_current_window(loaded)
    update["replenishment_loaded_windows"] = loaded
    update["replenishment_loaded_windows_round_id"] = round_id
    raw_attempted = state.get("replenishment_attempted_windows") or {}
    attempted = _parse_window_map(raw_attempted, "replenishment_attempted_windows_round_id")
    _mark_current_window(attempted)
    update["replenishment_attempted_windows"] = attempted
    update["replenishment_attempted_windows_round_id"] = round_id
    return update


def _difficulty_key(q: dict) -> str:
    """从题目 dict 中提取 dynamic_difficulty 字段，缺失时返回 "unknown"。"""
    return str(q.get("dynamic_difficulty") or "unknown")


def sample_questions_for_round(
    questions: list[dict],
    strategy: dict,
    mastered_memory_path: str | None = None,
    focus_questions: list[dict] | None = None,
    target_bucket: str = "",
    sampling_plan: dict | None = None,
) -> list[dict]:
    """Pre-rollout 阶段的主采样函数。

    按顺序执行以下步骤：
    1. 去重（可选）—— 基于 question_text 或 dedup_key
    2. 丢弃 mastered 题目（可选）—— 从 mastered_memory_path 文件中读取
    3. 按 sampling_plan 分层采样（primary_axis = dynamic_difficulty）
    4. 按 probe focus 偏置采样（优先选取与失败 probe 同 module 的题目）
    5. 兜底随机/分层采样
    """
    pool = list(questions)

    # Step 1: 基于文本去重
    if strategy.get("deduplicate", True):
        seen_texts: set[str] = set()
        deduped = []
        for q in pool:
            text_key = q.get("dedup_key") or q.get("question_text", "")
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                deduped.append(q)
        pool = deduped

    # Step 2: 从 mastered 记忆文件中读取题目 ID，从候选池中移除
    if strategy.get("drop_mastered", True) and mastered_memory_path:
        mastered_path = Path(mastered_memory_path)
        if mastered_path.exists():
            with open(mastered_path, "r", encoding="utf-8") as f:
                mastered = json.load(f)
            if isinstance(mastered, dict):
                mastered_list: list[dict] = []
                for v in mastered.values():
                    if isinstance(v, list):
                        mastered_list.extend(v)
            else:
                mastered_list = mastered if isinstance(mastered, list) else []
            mastered_ids = {q.get("question_id", "") for q in mastered_list if isinstance(q, dict)}
            pool = [q for q in pool if q.get("question_id", "") not in mastered_ids]

    target_count = int(strategy.get("target_questions_per_round", len(pool)) or len(pool))

    # Step 3: 如果有 sampling_plan，按 difficulty 权重分层采样
    if sampling_plan and sampling_plan.get("primary_axis") == "dynamic_difficulty":
        stratified = _sample_by_plan(pool, sampling_plan, target_count)
        if len(stratified) >= target_count:
            return stratified[:target_count]

    # Step 4: 如果有 probe focus 题目，偏置采样至与失败 probe 同 module 的题目
    if focus_questions:
        focus_modules = set(
            q.get("module")
            for q in focus_questions
            if q.get("module") and q.get("module") != "unknown"
        )
        if focus_modules:
            module_matched = [q for q in pool if q.get("module") in focus_modules]
            other = [q for q in pool if q.get("module") not in focus_modules]
            random.shuffle(module_matched)
            random.shuffle(other)
            prioritized = module_matched + other
            if len(prioritized) >= target_count:
                return prioritized[:target_count]

    # Step 5: 兜底——shuffle 后取前 target_count 个
    method = strategy.get("sampling_method", "stratified")
    if method == "random" or method == "stratified":
        random.shuffle(pool)

    return pool[:target_count]


def _normalized_weights(raw: dict | None, allowed: set[str] | None = None) -> dict[str, float]:
    """将权重字典归一化为总和为 1.0 的概率分布。

    可选参数 allowed 用于过滤仅允许的 key，非正权重会被丢弃。
    """
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, value in raw.items():
        if allowed is not None and key not in allowed:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight > 0:
            cleaned[str(key)] = weight
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in cleaned.items()}


def _allocate_counts(weights: dict[str, float], total: int, available: dict[str, int]) -> dict[str, int]:
    """按权重比例分配配额，同时受 available 上限约束。

    使用 largest-remainder 方法处理余数分配，确保总数恰好等于 total。
    例如：total=10, weights={a: 0.3, b: 0.7}, available={a: 2, b: 100}
    → quotas={a: 2 (受 avail 限制), b: 8}。
    """
    if total <= 0:
        return {key: 0 for key in weights}
    usable = {key: weight for key, weight in weights.items() if available.get(key, 0) > 0}
    if not usable:
        return {}
    normalized = _normalized_weights(usable)
    quotas = {
        key: min(available.get(key, 0), int(total * weight))
        for key, weight in normalized.items()
    }
    remaining = total - sum(quotas.values())
    # 按小数部分降序分配余数（largest-remainder 法）
    order = sorted(
        normalized,
        key=lambda key: (total * normalized[key]) - int(total * normalized[key]),
        reverse=True,
    )
    while remaining > 0 and order:
        progressed = False
        for key in order:
            if remaining <= 0:
                break
            if quotas.get(key, 0) >= available.get(key, 0):
                continue
            quotas[key] = quotas.get(key, 0) + 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return quotas


def _take_unique(source: list[dict], count: int, used_ids: set[tuple[str, object]]) -> list[dict]:
    """从 source 中取最多 count 个不重复的题目，跳过 used_ids 中已有的 key。"""
    if count <= 0:
        return []
    selected = []
    for q in source:
        keys = _question_keys(q)
        if keys & used_ids:
            continue
        selected.append(q)
        used_ids.update(keys)
        if len(selected) >= count:
            break
    return selected


def _source_row_identity(question: dict) -> tuple[str, str, str, str] | None:
    if not isinstance(question, dict):
        return None
    dataset_id = str(question.get("source_dataset_id") or "")
    row_id = str(question.get("source_dataset_row_id") or "")
    if not dataset_id or not row_id:
        return None
    return (
        dataset_id,
        str(question.get("source_dataset_subset") or ""),
        str(question.get("source_dataset_split") or ""),
        row_id,
    )


def _question_key(question: dict) -> tuple[str, object]:
    """生成题目的唯一 key：优先用数据集行身份，否则用 question_id，最后用题面 hash。"""
    if not isinstance(question, dict):
        return ("missing", "")
    row_identity = _source_row_identity(question)
    if row_identity is not None:
        return ("source_row", row_identity)
    if question.get("question_text"):
        return ("text_hash", question_text_hash(question))
    question_id = str(question.get("question_id") or "")
    if question_id:
        return ("question_id", question_id)
    return ("missing", "")


def _question_keys(question: dict) -> set[tuple[str, object]]:
    if not isinstance(question, dict):
        return set()
    row_identity = _source_row_identity(question)
    if row_identity is not None:
        keys: set[tuple[str, object]] = {("source_row", row_identity)}
        question_id = str(question.get("question_id") or "")
        if question_id:
            keys.add(("question_id", question_id))
        return keys
    if question.get("question_text"):
        return {("text_hash", question_text_hash(question))}
    question_id = str(question.get("question_id") or "")
    return {("question_id", question_id)} if question_id else set()


def _remove_questions_from_pools(cross_pool: dict[str, list[dict]], selected: list[dict]) -> None:
    """从 cross_pool 中移除已被选中（selected）的题目，避免重复分配。"""
    selected_keys = {
        key
        for q in selected
        for key in _question_keys(q)
        if key[1]
    }
    if not selected_keys:
        return
    for dataset_id, pool in list(cross_pool.items()):
        cross_pool[dataset_id] = [q for q in pool if not (_question_keys(q) & selected_keys)]


def _fresh_questions(questions: list[dict]) -> list[dict]:
    """过滤出非 replay 且具有 source_dataset_id 的"新鲜"题目。"""
    return [
        q for q in questions
        if q.get("source_role") != "replay" and q.get("source_dataset_id")
    ]


def _dataset_state_manager(state: EvoState) -> DatasetStateManager | None:
    """从 state 中获取 DatasetStateManager 实例。"""
    ds_states_path = state.get("dataset_states_path", "")
    return DatasetStateManager(Path(ds_states_path)) if ds_states_path else None


def _unused_question_ids(ds_mgr: DatasetStateManager | None, dataset_id: str) -> set[str] | None:
    """获取数据集中所有状态为 UNUSED 的题目 ID 集合。"""
    if ds_mgr is None or not dataset_id:
        return None
    ds = ds_mgr.datasets.get(dataset_id) or ds_mgr.load_cached(dataset_id)
    if ds is None:
        return None
    unused_by_difficulty = ds_mgr.get_unused_by_difficulty(dataset_id)
    return {
        qid
        for qids in unused_by_difficulty.values()
        for qid in qids
    }


def _filter_unused_pool(pool: list[dict], ds_mgr: DatasetStateManager | None, dataset_id: str) -> list[dict]:
    """过滤 pool，只保留对应数据集中状态为 UNUSED 的题目。"""
    unused_ids = _unused_question_ids(ds_mgr, dataset_id)
    if unused_ids is None:
        return pool
    return [q for q in pool if str(q.get("question_id") or "") in unused_ids]


def _filter_questions_to_unused_by_dataset(questions: list[dict], ds_mgr: DatasetStateManager | None) -> list[dict]:
    """过滤题目列表，只保留各数据集 source 中状态仍为 UNUSED 的题目。

    对每个 dataset 做缓存，避免重复查找。无对应 dataset_state 的题目默认放行。
    """
    if ds_mgr is None:
        return questions
    allowed_cache: dict[str, set[str] | None] = {}
    filtered: list[dict] = []
    for question in questions:
        dataset_id = str(question.get("source_dataset_id") or "")
        if not dataset_id:
            filtered.append(question)
            continue
        if dataset_id not in allowed_cache:
            ds = ds_mgr.datasets.get(dataset_id) or ds_mgr.load_cached(dataset_id)
            allowed_cache[dataset_id] = None if ds is None else {
                qid
                for qid, item in ds.items.items()
                if item.state == ITEM_STATE_UNUSED
            }
        allowed_ids = allowed_cache[dataset_id]
        if allowed_ids is None or str(question.get("question_id") or "") in allowed_ids:
            filtered.append(question)
    return filtered


def _filter_pools_to_unused(cross_pool: dict[str, list[dict]], ds_mgr: DatasetStateManager | None) -> None:
    """原地过滤 cross_pool 中所有 dataset 的 pool，只保留 UNUSED 状态题目。"""
    if ds_mgr is None:
        return
    for dataset_id, pool in list(cross_pool.items()):
        cross_pool[dataset_id] = _filter_unused_pool(pool, ds_mgr, dataset_id)


def _cached_rollout_score(
    question: dict,
    ds_mgr: DatasetStateManager | None,
    *,
    rollout_model_key: str = "",
    rollout_config_hash: str = "",
    rollout_judge_version: str = ROLLOUT_CACHE_JUDGE_VERSION,
) -> dict | None:
    """从 DatasetState 中读取题目缓存的 rollout 结果（pass_rate, pass_count 等）。

    如果 dataset_state 中有该题目的 rollout 记录且状态为 UNUSED，则直接返回
    构造好的 scored dict，避免重复进行 rollout 计算。
    """
    if ds_mgr is None:
        return None
    dataset_id = str(question.get("source_dataset_id") or "")
    question_id = str(question.get("question_id") or "")
    if not dataset_id or not question_id:
        return None
    ds = ds_mgr.datasets.get(dataset_id) or ds_mgr.load_cached(dataset_id)
    if ds is None:
        return None
    item = ds.items.get(question_id)
    if item is None or item.state != ITEM_STATE_UNUSED:
        return None
    rollout_count = int(item.rollout_count or 0)
    if rollout_count <= 0:
        return None
    if item.rollout_model_key and item.rollout_model_key != rollout_model_key:
        return None
    if item.rollout_config_hash and item.rollout_config_hash != rollout_config_hash:
        return None
    if item.rollout_judge_version and item.rollout_judge_version != rollout_judge_version:
        return None
    pass_rate = max(0.0, min(1.0, float(item.pass_rate or 0.0)))
    pass_count = max(0, min(rollout_count, int(round(pass_rate * rollout_count))))
    scored = add_processed_question_fields({
        "question_id": question_id,
        "question_text": str(question.get("question_text") or ""),
        "gold_answer": str(question.get("gold_answer") or ""),
        "source_dataset_id": dataset_id,
        "source_dataset_row_id": question.get("source_dataset_row_id"),
        "source_dataset_split": question.get("source_dataset_split"),
        "source_dataset_subset": question.get("source_dataset_subset"),
        "source_dataset_requested_split": question.get("source_dataset_requested_split"),
        "source_dataset_split_names": question.get("source_dataset_split_names") or [],
        "source_dataset_columns": question.get("source_dataset_columns") or [],
        "source_dataset_first_row": question.get("source_dataset_first_row") or {},
        "source_dataset_schema": question.get("source_dataset_schema") or {},
        "dataset_window_id": question.get("dataset_window_id"),
        "dataset_window_offset": question.get("dataset_window_offset"),
        "dataset_window_limit": question.get("dataset_window_limit"),
        "source_role": question.get("source_role") or "new",
        "module": question.get("module"),
        "dynamic_difficulty": str(item.difficulty or "unknown"),
        "pass_count": pass_count,
        "rollout_count": rollout_count,
        "pass_rate": pass_rate,
        "rollout_stage": item.rollout_stage or "",
        "rollout_model_key": item.rollout_model_key or rollout_model_key,
        "rollout_config_hash": item.rollout_config_hash or rollout_config_hash,
        "rollout_judge_version": item.rollout_judge_version or rollout_judge_version,
        "replay_use_count": question.get("replay_use_count"),
        "correct_flags": [True] * pass_count + [False] * (rollout_count - pass_count),
    }, question)
    return scored


def _split_cached_rollout_questions(
    questions: list[dict],
    ds_mgr: DatasetStateManager | None,
    *,
    rollout_model_key: str = "",
    rollout_config_hash: str = "",
    rollout_judge_version: str = ROLLOUT_CACHE_JUDGE_VERSION,
) -> tuple[list[dict], list[dict]]:
    """将题目分为两组：无缓存的（需要实际 rollout）和有缓存结果的。"""
    uncached: list[dict] = []
    cached_scored: list[dict] = []
    for question in questions:
        cached = _cached_rollout_score(
            question,
            ds_mgr,
            rollout_model_key=rollout_model_key,
            rollout_config_hash=rollout_config_hash,
            rollout_judge_version=rollout_judge_version,
        )
        if cached is None:
            uncached.append(question)
        else:
            cached_scored.append(cached)
    return uncached, cached_scored


def _limit_cached_scored_for_replenishment(
    scored_questions: list[dict],
    state: EvoState,
) -> list[dict]:
    """限制 replenishment 场景下复用缓存结果的数量。

    考虑已有 quota_accumulated_questions 的已选数量，从缓存中按 difficulty 权重
    补足剩余配额。兜底时取前 N 个。
    """
    if not scored_questions:
        return []
    train_target = FILTER_TARGET_QUESTIONS_PER_ROUND
    target = _post_rollout_pool_target(train_target)
    existing = [
        q for q in list(state.get("quota_accumulated_questions") or [])
        if isinstance(q, dict)
    ]
    weights = (state.get("sampling_plan") or {}).get("difficulty_weights")
    if isinstance(weights, dict) and weights:
        selected, _shortfall = _sample_primary_with_shortfall(
            scored_questions,
            weights,
            target,
            existing_selected=existing,
        )
        if selected:
            return selected[:target]
        return scored_questions[:target]
    remaining = max(0, target - len(existing))
    return scored_questions[:remaining]


def _emit_cached_rollout_result(state: EvoState, scored_questions: list[dict]) -> RoutedMessage:
    """构造一条使用缓存结果替代实际 rollout 的 ROLLOUT_RESULT 消息。

    当所有候选题目都已缓存了 rollout 结果时，不需要实际执行 rollout，
    直接构造一条伪消息进入 post-rollout 流程。
    """
    scored_ref = write_json_artifact(
        trace_id=str(state.get("trace_id", "")),
        round_id=int(state.get("round_id", 0) or 0),
        producer="filter",
        name="cached_scored_questions",
        data=scored_questions,
    )
    return RoutedMessage(
        header=MessageHeader(
            trace_id=str(state.get("trace_id", "")),
            round_id=int(state.get("round_id", 0) or 0),
            sender=AgentName.ROLLOUT_AGGREGATOR,
            receiver=AgentName.FILTER,
            message_type=MessageType.ROLLOUT_RESULT,
        ),
        payload=RolloutResultPayload(
            scored_questions=[],
            scored_questions_ref=scored_ref,
        ),
    )


def _mark_below_threshold_exhausted(
    ds_mgr: DatasetStateManager | None,
    dataset_ids: set[str],
    threshold: int,
) -> None:
    """标记剩余 unused 题目数低于 threshold 的数据集为 exhausted。"""
    if ds_mgr is None:
        return
    for dataset_id in sorted(did for did in dataset_ids if did):
        if ds_mgr.mark_exhausted_if_below(dataset_id, threshold):
            print(
                f"[filter] Dataset exhausted: dataset={dataset_id} "
                f"remaining_unused={ds_mgr.count_unused(dataset_id)} threshold={threshold}"
            )


def _shortfall_from_selected(
    selected: list[dict],
    weights: dict[str, float],
    target_total: int,
) -> dict[str, int]:
    """计算已选题目与目标配额之间的差异（按 difficulty 统计）。"""
    quotas = _difficulty_quotas(weights, target_total)
    if not quotas:
        return {}
    selected_counts = Counter(_difficulty_key(q) for q in selected)
    shortfall: dict[str, int] = {}
    for diff, quota in quotas.items():
        deficit = quota - int(selected_counts.get(diff, 0) or 0)
        if deficit > 0:
            shortfall[diff] = deficit
    unmet_total = target_total - len(selected)
    if unmet_total > 0 and not shortfall:
        shortfall["unknown"] = unmet_total
    return shortfall


def _difficulty_quotas(weights: dict[str, float], target_total: int) -> dict[str, int]:
    """从权重计算每个 difficulty 的绝对配额，使用 largest-remainder 处理余数。"""
    total_weight = sum(float(value) for value in weights.values())
    if target_total <= 0 or total_weight <= 0:
        return {}
    quotas: dict[str, int] = {}
    fractions: dict[str, float] = {}
    allocated_total = 0
    for diff, weight in sorted(weights.items()):
        exact = target_total * float(weight) / total_weight
        quota = int(exact)
        quotas[str(diff)] = quota
        fractions[str(diff)] = exact - quota
        allocated_total += quota
    remainder = target_total - allocated_total
    for diff, _fraction in sorted(fractions.items(), key=lambda item: item[1], reverse=True)[:max(0, remainder)]:
        quotas[diff] = quotas.get(diff, 0) + 1
    return quotas


def _post_rollout_pool_target(
    train_target: int,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    cotest_ratio: float = COTEST_SPLIT_RATIO,
    test_ratio: float = TEST_SPLIT_RATIO,
    probe_ratio: float = PROBE_POOL_INTAKE_RATIO,
    lf_val_ratio: float = LF_VAL_SPLIT_RATIO,
    lf_val_enabled: bool = TRAIN_LF_EVAL_ENABLED,
) -> int:
    """将配置的最终 train 目标数转换为 split 之前的 pool 总量。

    因为题目池会被拆分为 train/cotest/test/probe 等多个 split，
    所以需要按比例放大，确保 train split 能分到 target 个。
    如果启用了 LF val 再扣减其比例。
    """
    target = max(1, int(train_target or 1))
    ratio_sum = (
        float(train_ratio or 0.0)
        + float(cotest_ratio or 0.0)
        + float(test_ratio or 0.0)
        + float(probe_ratio or 0.0)
    )
    if ratio_sum <= 0 or float(train_ratio or 0.0) <= 0:
        return target
    train_share = float(train_ratio or 0.0) / ratio_sum
    if lf_val_enabled:
        lf_val_share = max(0.0, min(0.95, float(lf_val_ratio or 0.0)))
        train_share *= 1.0 - lf_val_share
    train_share = max(train_share, 1e-6)
    return max(target, int(math.ceil(target / train_share)))


def _dominant_target_style(questions: list[dict]) -> str | None:
    """找出题目列表中最常见的 target_style。"""
    counts = Counter(
        style for style in (normalize_target_style(q.get("target_style")) for q in questions)
        if style is not None
    )
    return str(counts.most_common(1)[0][0]) if counts else None


def _is_dataset_exhausted(ds_mgr: DatasetStateManager | None, dataset_id: str) -> bool:
    """检查指定数据集是否已被标记为 exhausted。"""
    return bool(ds_mgr is not None and dataset_id and ds_mgr.all_exhausted(dataset_id))


def _dataset_identity(ref: dict) -> tuple[str, str, str]:
    """从 dataset ref dict 提取身份三元组 (dataset_id, subset, split)。"""
    return (
        str(ref.get("dataset_id", "")),
        str(ref.get("subset") or ""),
        str(ref.get("split") or "train"),
    )


def _ref_window_start(ref: dict) -> int | None:
    """获取 dataset ref 中的 shard_start 偏移量（如果有）。"""
    raw_start = ref.get("shard_start")
    if raw_start is None:
        return None
    try:
        return int(raw_start)
    except (TypeError, ValueError):
        return None


def _unique_replenishment_refs(state: EvoState) -> list[dict]:
    """从 state 中收集所有不重复的 dataset ref，去重后返回。

    数据来源包括：next_dataset_ref, dataset_schema_info, dataset_pool, previous_dataset_refs。
    去重依据：(dataset_id, subset, split, shard_start) 四元组。
    """
    candidates: list[dict] = []
    next_ref = state.get("next_dataset_ref")
    if isinstance(next_ref, dict):
        candidates.append(dict(next_ref))
    schema_info = state.get("dataset_schema_info") or {}
    if isinstance(schema_info, dict):
        schema_ref = schema_info.get("dataset_ref")
        if isinstance(schema_ref, dict):
            candidates.append(dict(schema_ref))
    for ref in state.get("dataset_pool") or []:
        if isinstance(ref, dict):
            candidates.append(dict(ref))
    for ref in state.get("previous_dataset_refs") or []:
        if isinstance(ref, dict):
            candidates.append(dict(ref))

    seen: set[tuple[str, str, str, int | None]] = set()
    unique_refs: list[dict] = []
    for ref in candidates:
        dataset_id, subset, split = _dataset_identity(ref)
        if not dataset_id:
            continue
        key = (dataset_id, subset, split, _ref_window_start(ref))
        if key in seen:
            continue
        seen.add(key)
        unique_refs.append(ref)
    return unique_refs


def _has_replenishment_candidate(state: EvoState, active_dataset_id: str, window_exhausted: bool) -> bool:
    """检查是否有可用的 replenishment 候选（新的 dataset window）。

    逻辑：
    - 如果 window 未耗尽，检查当前 dataset 是否仍有 unused 题目，或其他 ref 可用
    - 如果 window 已耗尽，检查其他 dataset 或同一 dataset 的后续 shard
    """
    refs = _unique_replenishment_refs(state)
    ds_mgr = _dataset_state_manager(state)
    if state.get("dataset_review_active") and state.get("dataset_review_pending_refs"):
        return True
    if not window_exhausted:
        return bool(active_dataset_id and not _is_dataset_exhausted(ds_mgr, active_dataset_id)) or bool(refs)

    current_offset = int(state.get("current_window_offset", 0) or 0)
    for ref in refs:
        dataset_id = str(ref.get("dataset_id", ""))
        if _is_dataset_exhausted(ds_mgr, dataset_id):
            continue
        if dataset_id and dataset_id != active_dataset_id:
            return True
        ref_start = _ref_window_start(ref)
        if dataset_id == active_dataset_id and ref_start is not None and ref_start > current_offset:
            return True
    return False


def _has_cached_replenishment_ref(
    state: EvoState,
    shortfall: dict[str, int],
    max_windows: int,
    max_profile_items: int,
) -> bool:
    """检查是否有之前缓存过的 window 可以复用，而非必须加载新 window。

    通过构造一个 budget 耗尽后的 probe state，询问 screening_entry 是否还有
    之前缓存过的下一次 window ref。
    """
    if not shortfall:
        return False

    screening_entry_module = import_module("src.nodes.screening_entry")
    budget_exhausted_probe = {
        **state,
        "quota_shortfall": dict(shortfall),
        "windows_loaded_this_round": max_windows,
        "profile_items_loaded_this_round": max_profile_items,
        "max_windows_per_round": max_windows,
        "max_profile_items_per_round": max_profile_items,
    }
    return screening_entry_module._ref_for_next_window(budget_exhausted_probe) is not None


def _sample_by_plan(
    questions: list[dict],
    sampling_plan: dict,
    target_count: int,
) -> list[dict]:
    """按 sampling_plan 进行两层分层采样：先按 difficulty 分桶，再按 module 细分。

    使用 _allocate_counts 进行配额分配，确保每个 bucket/module 都按权重取到足量题目。
    如果配额未满，兜底从所有 bucket 中补充。
    """
    bucket_weights = _normalized_weights(
        sampling_plan.get("difficulty_weights"),
        allowed={"easy", "medium", "hard", "unknown"},
    )
    module_weights = _normalized_weights(sampling_plan.get("module_weights"))
    if not bucket_weights:
        return []
    has_dynamic_labels = any(
        str(q.get("dynamic_difficulty") or "") in {"easy", "medium", "hard", "unknown"}
        for q in questions
    )
    if not has_dynamic_labels:
        return []

    by_bucket: dict[str, list[dict]] = {}
    for q in questions:
        bucket = _difficulty_key(q)
        by_bucket.setdefault(bucket, []).append(q)
    for items in by_bucket.values():
        random.shuffle(items)

    bucket_available = {bucket: len(by_bucket.get(bucket, [])) for bucket in bucket_weights}
    bucket_quotas = _allocate_counts(bucket_weights, target_count, bucket_available)
    selected: list[dict] = []
    used_ids: set[tuple[str, object]] = set()

    for bucket, bucket_quota in bucket_quotas.items():
        if bucket_quota <= 0:
            continue
        bucket_pool = by_bucket.get(bucket, [])
        bucket_selected_count = 0
        if module_weights:
            by_module: dict[str, list[dict]] = {}
            for q in bucket_pool:
                by_module.setdefault(q.get("module") or "unknown", []).append(q)
            module_available = {
                module: len(by_module.get(module, []))
                for module in module_weights
            }
            module_quotas = _allocate_counts(module_weights, bucket_quota, module_available)
            for module, module_quota in module_quotas.items():
                picked = _take_unique(by_module.get(module, []), module_quota, used_ids)
                selected.extend(picked)
                bucket_selected_count += len(picked)
        selected.extend(_take_unique(bucket_pool, bucket_quota - bucket_selected_count, used_ids))

    if len(selected) < target_count:
        fallback = []
        for items in by_bucket.values():
            fallback.extend(items)
        random.shuffle(fallback)
        selected.extend(_take_unique(fallback, target_count - len(selected), used_ids))

    selected = selected[:target_count]

    selected_bucket_counts = Counter(
        _difficulty_key(q)
        for q in selected
    )
    selected_module_counts = Counter(q.get("module") or "unknown" for q in selected)
    print(
        "[filter] Plan sampling: "
        f"difficulty_weights={bucket_weights}, module_weights={module_weights}, "
        f"selected_difficulty={dict(selected_bucket_counts)}, selected_modules={dict(selected_module_counts)}"
    )
    return selected


def _load_probe_focus_questions(path: str | None) -> list[dict]:
    """从 JSON 文件加载 probe diagnostic focus 题目列表。

    这些题目来自 probe 阶段失败的样本，用于在后续采样中偏置选择相似题目。
    """
    if not path:
        return []
    focus_path = Path(path)
    if not focus_path.exists():
        return []
    try:
        with open(focus_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception as exc:
        print(f"[filter] Probe focus load warning: {exc}")
        return []
    if not isinstance(loaded, list):
        return []
    return [q for q in loaded if isinstance(q, dict)]


def _sample_by_probe_focus(
    questions: list[dict],
    focus_questions: list[dict],
    target_bucket: str,
    target_count: int,
) -> list[dict]:
    """偏置采样到与 probe 失败题目相似（同 module / 同 difficulty）的候选题目。

    Probe 题目本身不会插入训练，它们只定义 profile 用于选择相似的训练候选。
    优先选择同时 module + difficulty 匹配的，其次 module only，再 difficulty only。
    """
    module_counts = Counter(
        q.get("module")
        for q in focus_questions
        if q.get("module") and q.get("module") != "unknown"
    )
    difficulty_counts = Counter(
        q.get("dynamic_difficulty")
        for q in focus_questions
        if q.get("dynamic_difficulty") and q.get("dynamic_difficulty") != "unknown"
    )
    focus_modules = set(module_counts)
    focus_difficulty = (
        target_bucket
        if target_bucket in _DYNAMIC_DIFFICULTIES
        else (difficulty_counts.most_common(1)[0][0] if difficulty_counts else "")
    )

    if not focus_modules and not focus_difficulty:
        return []

    exact: list[dict] = []
    module_only: list[dict] = []
    bucket_only: list[dict] = []
    other: list[dict] = []

    for q in questions:
        module_match = bool(q.get("module") in focus_modules)
        difficulty_match = bool(focus_difficulty and q.get("dynamic_difficulty") == focus_difficulty)
        if module_match and difficulty_match:
            exact.append(q)
        elif module_match:
            module_only.append(q)
        elif difficulty_match:
            bucket_only.append(q)
        else:
            other.append(q)

    for group in (exact, module_only, bucket_only, other):
        random.shuffle(group)

    focus_quota = int(target_count * TARGET_BUCKET_RATIO)
    result = []
    result.extend(exact[:focus_quota])
    if len(result) < focus_quota:
        result.extend(module_only[: focus_quota - len(result)])
    if len(result) < focus_quota:
        result.extend(bucket_only[: focus_quota - len(result)])

    remaining = target_count - len(result)
    if remaining > 0:
        used_ids = {key for q in result for key in _question_keys(q)}
        fallback = [
            q for q in exact + module_only + bucket_only + other
            if not (_question_keys(q) & used_ids)
        ]
        result.extend(fallback[:remaining])

    print(
        "[filter] Probe focus sampling: "
        f"focus_modules={dict(module_counts)}, focus_difficulty={focus_difficulty}, "
        f"selected_focus={min(len(result), focus_quota)}/{target_count}"
    )
    return result


def _oversample_by_dynamic_difficulty(
    questions: list[dict],
    target_difficulty: str,
    target_count: int,
) -> list[dict]:
    """只按 rollout 后的动态难度做偏置采样，不再使用 step 作为策略参数。"""
    in_bucket: list[dict] = []
    out_bucket: list[dict] = []

    for q in questions:
        difficulty = str(q.get("dynamic_difficulty") or "unknown")
        if difficulty == target_difficulty:
            in_bucket.append(q)
        else:
            out_bucket.append(q)

    random.shuffle(in_bucket)
    random.shuffle(out_bucket)

    target_from_bucket = min(len(in_bucket), int(target_count * TARGET_BUCKET_RATIO))
    remainder = target_count - target_from_bucket

    result = in_bucket[:target_from_bucket]
    result.extend(out_bucket[:remainder])
    return result


def mix_replay_and_new_data(
    new_questions: list[dict],
    replay_buffer: list[dict],
    sample_ratio: float,
    current_round: int = 0,
    target_bucket: str = "",
) -> tuple[list[dict], list[int]]:
    """将新题目与 replay buffer 中的旧题目混合，按比例采样。

    Replay buffer 条目按 bucket（dynamic_difficulty）感知的方式采样，
    确保各难度级别的题目都有机会被回放。

    Returns:
        mixed_questions: 混合后的题目列表，replay 题目标记 source_role="replay"
        selected_replay_indices: 选中 replay 条目在原 replay_buffer 中的下标，
                                  用于回写 used_in_rounds。
    """
    if not replay_buffer:
        return new_questions, []

    valid_replay = [
        entry for entry in replay_buffer
        if _MATH_QUESTION_REQUIRED_FIELDS.issubset(entry.keys())
    ]

    if not new_questions:
        sample_size = 0
    else:
        sample_size = min(len(valid_replay), max(0, int(len(new_questions) * sample_ratio)))

    selected = sample_from_bucket(
        entries=valid_replay,
        n=sample_size,
        current_round=current_round,
        target_bucket=target_bucket,
    )

    # 将选中的 replay 条目映射回原始 replay_buffer 下标，用于回写 used_in_rounds
    selected_entry_ids = {
        (e.get("entry_id", ""), e.get("question_id", "")) for e in selected
    }
    selected_indices = []
    for i, entry in enumerate(replay_buffer):
        key = (entry.get("entry_id", ""), entry.get("question_id", ""))
        if key in selected_entry_ids:
            selected_indices.append(i)

    # 从选中的 replay 条目中提取 question 级别的信息
    replay_questions = []
    for e in selected:
        if not _MATH_QUESTION_REQUIRED_FIELDS.issubset(e.keys()):
            continue
        replay_questions.append(
            add_processed_question_fields(
                {
                    "question_id": e.get("question_id", ""),
                    "question_text": e.get("question_text", ""),
                    "gold_answer": e.get("gold_answer", ""),
                    "evaluation_method": e.get("evaluation_method", "gold"),
                    "needs_judge": bool(e.get("needs_judge", False)),
                    "source_dataset_id": e.get("source_dataset_id"),
                    "source_dataset_row_id": e.get("source_dataset_row_id"),
                    "source_dataset_split": e.get("source_dataset_split"),
                    "source_dataset_subset": e.get("source_dataset_subset"),
                    "source_dataset_requested_split": e.get("source_dataset_requested_split"),
                    "source_dataset_split_names": e.get("source_dataset_split_names") or [],
                    "source_dataset_columns": e.get("source_dataset_columns") or [],
                    "source_dataset_first_row": e.get("source_dataset_first_row") or {},
                    "source_dataset_schema": e.get("source_dataset_schema") or {},
                    "module": e.get("module"),
                    "dynamic_difficulty": e.get("dynamic_difficulty") or e.get("bucket") or e.get("difficulty", "unknown"),
                    "pass_count": e.get("pass_count"),
                    "rollout_count": e.get("rollout_count"),
                    "pass_rate": e.get("pass_rate"),
                    "source_role": "replay",
                    "replay_use_count": len(e.get("used_in_rounds", [])),
                },
                e,
            )
        )

    tagged_new_questions = []
    for question in new_questions:
        tagged = dict(question)
        tagged.setdefault("source_role", "new")
        tagged_new_questions.append(tagged)

    return tagged_new_questions + replay_questions, selected_indices


def drop_mastered_questions(scored_questions: list[QuestionScore]) -> list[dict]:
    """丢弃所有 correct_flags 全部为 True 的"已掌握"题目。

    仅用于 post-rollout 阶段严格筛选——只要有一道题做错（一轮 rollout 中有任一失败）就保留。
    """
    kept = []
    for item in scored_questions:
        if all(item.correct_flags):
            continue
        kept.append(
            add_processed_question_fields({
                "question_id": item.question_id,
                "question_text": item.question_text,
                "gold_answer": item.gold_answer,
                "evaluation_method": item.evaluation_method,
                "needs_judge": item.needs_judge,
                "source_dataset_id": item.source_dataset_id,
                "source_dataset_row_id": item.source_dataset_row_id,
                "source_dataset_split": item.source_dataset_split,
                "source_dataset_subset": item.source_dataset_subset,
                "source_dataset_requested_split": item.source_dataset_requested_split,
                "source_dataset_split_names": item.source_dataset_split_names,
                "source_dataset_columns": item.source_dataset_columns,
                "source_dataset_first_row": item.source_dataset_first_row,
                "source_dataset_schema": item.source_dataset_schema,
                "source_role": item.source_role,
                "module": item.module,
                "dynamic_difficulty": item.dynamic_difficulty,
                "pass_count": item.pass_count,
                "rollout_count": item.rollout_count,
                "pass_rate": item.pass_rate,
                "replay_use_count": item.replay_use_count,
            }, item)
        )
    return kept


def collect_mastered_questions(scored_questions: list[QuestionScore]) -> list[dict]:
    """收集已达到"掌握"阈值的题目。

    判断标准：
    - 当 rollout_count == ROLLOUT_TIMES（完整 rollout）：使用 MASTERED_CORRECT_THRESHOLD
    - 其他情况：correct_count >= rollout_count * MASTERED_CORRECT_RATIO（向上取整）
    """
    mastered = []
    for item in scored_questions:
        rollout_count = item.rollout_count or len(item.correct_flags)
        correct_count = sum(1 for flag in item.correct_flags if flag)
        if rollout_count == ROLLOUT_TIMES:
            threshold = MASTERED_CORRECT_THRESHOLD
        else:
            threshold = int(rollout_count * MASTERED_CORRECT_RATIO + 0.999999)
        threshold = min(max(1, threshold), max(1, rollout_count))
        if correct_count < threshold:
            continue
        mastered.append(
            add_processed_question_fields({
                "question_id": item.question_id,
                "question_text": item.question_text,
                "gold_answer": item.gold_answer,
                "evaluation_method": item.evaluation_method,
                "needs_judge": item.needs_judge,
                "source_dataset_id": item.source_dataset_id,
                "source_dataset_row_id": item.source_dataset_row_id,
                "source_dataset_split": item.source_dataset_split,
                "source_dataset_subset": item.source_dataset_subset,
                "source_dataset_requested_split": item.source_dataset_requested_split,
                "source_dataset_split_names": item.source_dataset_split_names,
                "source_dataset_columns": item.source_dataset_columns,
                "source_dataset_first_row": item.source_dataset_first_row,
                "source_dataset_schema": item.source_dataset_schema,
                "source_role": item.source_role,
                "module": item.module,
                "dynamic_difficulty": item.dynamic_difficulty,
                "pass_count": item.pass_count,
                "rollout_count": item.rollout_count,
                "pass_rate": item.pass_rate,
                "replay_use_count": item.replay_use_count,
            }, item)
        )
    return mastered


def filter_pre_rollout_node(state: EvoState) -> dict:
    """Pre-rollout 筛选节点——在模型执行 rollout 之前确定哪些题目需要评估。

    完整流程：
    1. 从 state 获取 materialized 题目
    2. 过滤掉已使用/已废弃的题目（基于 dataset_state）
    3. 过滤掉 heldout registry 中的题目
    4. 将题目分为"有缓存 rollout 结果"和"需要实际 rollout"两组
    5. 如果有缓存结果且 window 非新探索，跳过 rollout（避免重复计算）
    6. 从 uncached 题目中采样本轮需要 rollout 的候选
    7. 混合 replay buffer 中的旧题目
    8. 如果既无缓存也无新题目 → 发空结果；如果全是缓存 → 发伪 rollout 结果
    9. 否则 → 发送 RolloutRequest 给 ROLLOUT_DISPATCHER
    """
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    materialized = MaterializedDatasetPayload.model_validate(
        pending_message.payload
    )

    strategy = FilterStrategyPayload(
        target_questions_per_round=FILTER_TARGET_QUESTIONS_PER_ROUND,
        deduplicate=FILTER_DEDUPLICATE,
        drop_mastered=FILTER_DROP_MASTERED,
        sampling_method=FILTER_SAMPLING_METHOD,
    )

    target_bucket = state.get("target_bucket", "")
    focus_questions = _load_probe_focus_questions(state.get("probe_diagnostic_focus_path"))
    if focus_questions:
        focus_modules = Counter(
            q.get("module")
            for q in focus_questions
            if q.get("module") and q.get("module") != "unknown"
        )
        focus_buckets = Counter(
            q.get("dynamic_difficulty")
            for q in focus_questions
            if q.get("dynamic_difficulty") and q.get("dynamic_difficulty") != "unknown"
        )
        print(
            "[filter] Loaded probe diagnostic focus: "
            f"count={len(focus_questions)}, modules={dict(focus_modules)}, difficulties={dict(focus_buckets)}"
        )

    # Router 只传 artifact 地址；这里按需把题目列表读回机制层 state。
    raw_questions = load_payload_list(materialized.questions, materialized.questions_ref)
    raw_questions, fallback_schema_dropped = _drop_forbidden_fallback_schema_questions(raw_questions)
    if fallback_schema_dropped:
        print(
            "[filter] Dropped "
            f"{fallback_schema_dropped} fallback-schema questions before rollout"
        )
    ds_mgr = _dataset_state_manager(state)
    raw_count = len(raw_questions)
    raw_questions = _filter_questions_to_unused_by_dataset(raw_questions, ds_mgr)
    state_dropped = raw_count - len(raw_questions)
    if state_dropped:
        print(f"[filter] Dropped {state_dropped} used/defeated dataset-state questions before rollout")
    eligible_questions, heldout_dropped = drop_registered_questions(
        raw_questions,
        state.get("heldout_registry_path"),
        block_statuses=TRAIN_BLOCKING_STATUSES,
    )
    if heldout_dropped:
        print(f"[filter] Dropped {heldout_dropped} heldout questions from candidate pool")

    # 分组：有缓存结果的直接复用，无缓存的才需要 rollout
    rollout_model_key = str(state.get("champion_model_path", "") or "")
    rollout_config_hash = _rollout_config_hash()
    rollout_judge_version = ROLLOUT_CACHE_JUDGE_VERSION
    uncached_candidates, cached_scored_candidates = _split_cached_rollout_questions(
        eligible_questions,
        ds_mgr,
        rollout_model_key=rollout_model_key,
        rollout_config_hash=rollout_config_hash,
        rollout_judge_version=rollout_judge_version,
    )
    window_meta = (state.get("round_data_stats") or {}).get("window", {})
    reused_scored_window = (
        isinstance(window_meta, dict)
        and window_meta.get("new_exploration") is False
        and bool(cached_scored_candidates)
    )
    if reused_scored_window:
        uncached_candidates = []
    cached_scored_questions = _limit_cached_scored_for_replenishment(
        cached_scored_candidates,
        state,
    )
    sampled_questions = sample_questions_for_round(
        questions=uncached_candidates,
        strategy={
            "target_questions_per_round": len(uncached_candidates),
            "deduplicate": False,
            "drop_mastered": FILTER_DROP_MASTERED,
            "sampling_method": FILTER_SAMPLING_METHOD,
        },
        mastered_memory_path=state.get("mastered_memory_set_path"),
        focus_questions=focus_questions,
        sampling_plan=state.get("sampling_plan") or {},
    )

    # 混合 replay buffer 中的旧题目
    replay_buffer = list(state.get("replay_buffer_entries", []))
    replay_buffer, replay_heldout_dropped = drop_registered_questions(
        replay_buffer,
        state.get("heldout_registry_path"),
        block_statuses=TRAIN_BLOCKING_STATUSES,
    )
    if replay_heldout_dropped:
        print(f"[filter] Dropped {replay_heldout_dropped} heldout questions from replay buffer")
    round_id = state.get("round_id", 0)
    sample_ratio = float(state.get("replay_sample_ratio_override", REPLAY_BUFFER_SAMPLE_RATIO))
    if state.get("replenishment_cycle_active"):
        sample_ratio = 0.0
    mixed_questions, selected_indices = mix_replay_and_new_data(
        new_questions=sampled_questions,
        replay_buffer=replay_buffer,
        sample_ratio=sample_ratio,
        current_round=round_id,
        target_bucket=target_bucket,
    )

    # 回写 used_in_rounds 到 replay buffer 条目
    # sample_from_bucket 会原地修改 entry（dict 引用共享），
    # 所以 replay_buffer 条目实际已被更新。此循环作为安全网确保万无一失。
    for idx in selected_indices:
        if 0 <= idx < len(replay_buffer):
            entry = replay_buffer[idx]
            used_rounds = entry.setdefault("used_in_rounds", [])
            if round_id not in used_rounds:
                used_rounds.append(round_id)

    # 确保所有混合后的题目包含必要字段
    valid_mixed = [
        q for q in mixed_questions
        if _MATH_QUESTION_REQUIRED_FIELDS.issubset(q.keys())
    ]
    # 既无缓存也无新题目 → 直接发空结果
    if not cached_scored_questions and not valid_mixed:
        msg = _emit_cached_rollout_result(state, [])
        return {
            "candidate_questions": [],
            "cached_rollout_scored_questions": [],
            "rollout_config_hash": rollout_config_hash,
            "rollout_judge_version": rollout_judge_version,
            "pending_message": msg,
            **_replenishment_seen_state(state),
            "replay_buffer_entries": replay_buffer,
            "replay_buffer_used_count": len(selected_indices),
            "probe_focus_used_count": len(focus_questions),
        }

    if cached_scored_questions:
        print(
            "[filter] Reused cached rollout labels before rollout: "
            f"cached={len(cached_scored_questions)}, uncached={len(valid_mixed)}"
        )
    # 全缓存无新题 → 用伪 rollout 结果直接进入 post-rollout 流程
    if cached_scored_questions and not valid_mixed:
        msg = _emit_cached_rollout_result(state, cached_scored_questions)
        return {
            "candidate_questions": [],
            "cached_rollout_scored_questions": [],
            "rollout_config_hash": rollout_config_hash,
            "rollout_judge_version": rollout_judge_version,
            "pending_message": msg,
            **_replenishment_seen_state(state),
            "replay_buffer_entries": replay_buffer,
            "replay_buffer_used_count": len(selected_indices),
            "probe_focus_used_count": len(focus_questions),
        }

    # 需要实际 rollout → 发送 RolloutRequest
    questions_ref = write_json_artifact(
        trace_id=str(state.get("trace_id", "")),
        round_id=int(state.get("round_id", 0) or 0),
        producer="filter",
        name="rollout_questions",
        data=valid_mixed,
    )
    payload = RolloutRequestPayload(
        questions=[],
        questions_ref=questions_ref,
        rollout_times=1,
        model_path=rollout_model_key,
        rollout_config_hash=rollout_config_hash,
        rollout_judge_version=rollout_judge_version,
    )

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=str(state.get("trace_id", "")),
            round_id=int(state.get("round_id", 0) or 0),
            sender=AgentName.FILTER,
            receiver=AgentName.ROLLOUT_DISPATCHER,
            message_type=MessageType.ROLLOUT_REQUEST,
        ),
        payload=payload,
    )

    return {
        "candidate_questions": valid_mixed,
        "cached_rollout_scored_questions": cached_scored_questions,
        "rollout_config_hash": rollout_config_hash,
        "rollout_judge_version": rollout_judge_version,
        "pending_message": msg,
        **_replenishment_seen_state(state),
        "replay_buffer_entries": replay_buffer,
        "replay_buffer_used_count": len(selected_indices),
        "probe_focus_used_count": len(focus_questions),
    }


def _collect_kept_and_mastered(scored_questions: list[QuestionScore]) -> tuple[list[dict], list[dict]]:
    """保留所有题目在训练池中，同时收集已掌握题目。

    已掌握的题目仍然保留在训练数据中，用于强化旧知识。
    它们不会进入 test pool（test pool 使用上一轮 test split 中高准确率的题目）。
    dynamic_difficulty 由 rollout cascade 的成功阶段决定，这里不再覆盖。

    Returns:
        (kept_questions, mastered_questions): 全部保留的题目 + mastered 子集
    """
    kept: list[dict] = []
    mastered: list[dict] = []
    for item in scored_questions:
        rollout_count = item.rollout_count or len(item.correct_flags)
        correct_count = sum(1 for flag in item.correct_flags if flag)
        threshold = MASTERED_CORRECT_THRESHOLD if rollout_count == ROLLOUT_TIMES else int(
            rollout_count * MASTERED_CORRECT_RATIO + 0.999999
        )
        threshold = min(max(1, threshold), max(1, rollout_count))
        is_mastered = correct_count >= threshold
        entry = add_processed_question_fields({
            "question_id": item.question_id,
            "question_text": item.question_text,
            "gold_answer": item.gold_answer,
            "evaluation_method": item.evaluation_method,
            "needs_judge": item.needs_judge,
            "source_dataset_id": item.source_dataset_id,
            "source_dataset_row_id": item.source_dataset_row_id,
            "source_dataset_split": item.source_dataset_split,
            "source_dataset_subset": item.source_dataset_subset,
            "source_dataset_requested_split": item.source_dataset_requested_split,
            "source_dataset_split_names": item.source_dataset_split_names,
            "source_dataset_columns": item.source_dataset_columns,
            "source_dataset_first_row": item.source_dataset_first_row,
            "source_dataset_schema": item.source_dataset_schema,
            "dataset_window_id": item.dataset_window_id,
            "dataset_window_offset": item.dataset_window_offset,
            "dataset_window_limit": item.dataset_window_limit,
            "source_role": item.source_role,
            "module": item.module,
            "dynamic_difficulty": item.dynamic_difficulty,
            "pass_count": item.pass_count,
            "rollout_count": item.rollout_count,
            "pass_rate": item.pass_rate,
            "replay_use_count": item.replay_use_count,
        }, item)
        kept.append(entry)
        if is_mastered:
            mastered.append(entry)
    return kept, mastered


def filter_post_rollout_node(state: EvoState) -> dict:
    """Post-rollout 筛选节点——根据 rollout 结果确定最终训练题目分配。

    完整流程：
    1. 合并缓存的和实时 rollout 的结果
    2. 收集 mastered 题目（保留在训练池，不覆盖 rollout 难度）
    3. 过滤出非 replay 的 fresh 题目，补充 window meta 信息
    4. 更新 dataset_state 中的 pass_rate
    5. 将 fresh 题目追加到 cross_dataset_pool
    6. 按 teacher_weights（difficulty）从 primary pool 采样
       - 如果配额不足，从其他 dataset pool 补充（_supplement_from_pools）
       - 如果还不够，触发 replenishment 循环（加载新 window）
    7. 如果 replenishment budget 耗尽，以部分配额继续
    8. 最终混合 replay buffer
    9. 发送 FilterResult 给 CLASSIFIER
    """
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    rollout_result = RolloutResultPayload.model_validate(pending_message.payload)
    # 合并之前缓存的 scored 结果和本轮实时 rollout 的结果
    scored_questions = [
        QuestionScore.model_validate(q)
        for q in (
            list(state.get("cached_rollout_scored_questions") or [])
            + load_payload_list(
            rollout_result.scored_questions,
            rollout_result.scored_questions_ref,
        )
        )
    ]

    # Step 1: 收集 mastered，保留在训练池中且不覆盖 cascade 难度标签
    kept_questions, mastered_questions = _collect_kept_and_mastered(scored_questions)
    fresh_kept_questions = _fresh_questions(kept_questions)
    # Step 2: 补充 window 级别的元数据到题目 dict
    current_window = (state.get("round_data_stats") or {}).get("window", {})
    if isinstance(current_window, dict):
        window_offset = current_window.get("offset")
        window_limit = current_window.get("limit")
        window_id = current_window.get("window_id")
        for question in fresh_kept_questions:
            if question.get("dataset_window_offset") is None and window_offset is not None:
                question["dataset_window_offset"] = window_offset
            if question.get("dataset_window_limit") is None and window_limit is not None:
                question["dataset_window_limit"] = window_limit
            if question.get("dataset_window_id") is None and window_id is not None:
                question["dataset_window_id"] = window_id

    # Step 3: 确定主数据集（题目出现最多的 source_dataset）
    ds_counts = Counter(q.get("source_dataset_id") for q in fresh_kept_questions)
    active_dataset_id = str(ds_counts.most_common(1)[0][0] or "") if ds_counts else ""

    # Step 4: 更新 dataset_state 中的 pass_rate 信息
    ds_mgr = _dataset_state_manager(state)
    if ds_mgr is not None:
        by_dataset: dict[str, list[dict]] = {}
        for question in fresh_kept_questions:
            dataset_id = str(question.get("source_dataset_id") or "")
            if dataset_id:
                by_dataset.setdefault(dataset_id, []).append(question)
        for dataset_id, scored_items in by_dataset.items():
            ds_mgr.update_pass_rates(
                dataset_id=dataset_id,
                scored_items=scored_items,
                round_id=int(state.get("round_id", 0) or 0),
                probe_acc=float(state.get("champion_frozen_probe_error", 0.0) or 0.0),
            )

    # Step 5: 维护 cross-dataset pool
    cross_pool_raw = state.get("cross_dataset_pool") or {}
    if isinstance(cross_pool_raw, list):
        cross_pool_raw = {}
    cross_pool: dict[str, list[dict]] = dict(cross_pool_raw)

    # 处理 quota 累积的题目（来自 replenishment 的多轮累积）
    quota_accumulated = [
        q for q in list(state.get("quota_accumulated_questions") or [])
        if isinstance(q, dict)
    ]
    accumulated_keys = {key for q in quota_accumulated for key in _question_keys(q) if key[1]}

    # 将 fresh 题目追加到 active dataset 的 pool
    if active_dataset_id:
        cross_pool.setdefault(active_dataset_id, [])
        cross_pool[active_dataset_id].extend(
            q for q in fresh_kept_questions if not (_question_keys(q) & accumulated_keys)
        )
    _filter_pools_to_unused(cross_pool, ds_mgr)

    # Step 6: 按 difficulty 权重采样
    sampling_plan = state.get("sampling_plan") or {}
    teacher_weights = sampling_plan.get("difficulty_weights")
    train_target = FILTER_TARGET_QUESTIONS_PER_ROUND
    target = _post_rollout_pool_target(train_target)

    shortfall: dict[str, int] = {}
    selected: list[dict] = []
    if isinstance(teacher_weights, dict) and teacher_weights and active_dataset_id:
        # 有 teacher weights → 按 difficulty 分层采样，优先填满所有难度
        primary_pool = _filter_unused_pool(cross_pool.get(active_dataset_id, []), ds_mgr, active_dataset_id)

        # 按难度分组
        by_difficulty = {"easy": [], "medium": [], "hard": []}
        for q in primary_pool:
            diff = q.get("dynamic_difficulty", "unknown")
            if diff in by_difficulty:
                by_difficulty[diff].append(q)

        # 计算每个难度的目标数量
        difficulty_targets = {k: int(target * v) for k, v in teacher_weights.items()}

        # 优先填满当前窗口的所有难度
        selected_from_pool = []
        for diff in ["easy", "medium", "hard"]:
            needed = difficulty_targets.get(diff, 0)
            available = by_difficulty.get(diff, [])
            take = min(needed, len(available))
            if take > 0:
                import random
                selected_from_pool.extend(random.sample(available, take))

        selected = quota_accumulated + selected_from_pool
        selected, shortfall = selected[:target], _shortfall_from_selected(selected, teacher_weights, target)
        selected = quota_accumulated + [
            q for q in selected if not (_question_keys(q) & accumulated_keys)
        ]
        shortfall = _shortfall_from_selected(selected, teacher_weights, target)

        # 如果配额不足，从其他 dataset pool 补充
        if shortfall:
            supplement_target_style = _dominant_target_style(selected) or _dominant_target_style(primary_pool)
            supplemented = _supplement_from_pools(
                cross_pool,
                active_dataset_id,
                shortfall,
                ds_mgr,
                target_style=supplement_target_style,
            )
            selected_keys = {key for q in selected for key in _question_keys(q) if key[1]}
            selected.extend(q for q in supplemented if not (_question_keys(q) & selected_keys))
            shortfall = _shortfall_from_selected(selected, teacher_weights, target)
        if not shortfall:
            _remove_questions_from_pools(cross_pool, selected)
    else:
        # 无 teacher weights → 直接取前 N 个
        primary_pool = cross_pool.get(active_dataset_id, []) if active_dataset_id else []
        if not primary_pool:
            for ds_pool in cross_pool.values():
                primary_pool.extend(ds_pool)
        primary_pool = _filter_unused_pool(primary_pool, ds_mgr, active_dataset_id)
        selected = quota_accumulated + [
            q for q in primary_pool if not (_question_keys(q) & accumulated_keys)
        ]
        selected = selected[:target]
        if active_dataset_id:
            selected_ids = {q.get("question_id", "") for q in selected}
            cross_pool[active_dataset_id] = [
                q for q in cross_pool.get(active_dataset_id, [])
                if q.get("question_id", "") not in selected_ids
            ]
        shortfall = {} if len(selected) >= target else {"unknown": target - len(selected)}

    kept_questions = selected
    # Step 7: 检查是否可以触发 replenishment（加载新 window）
    windows_loaded = int(state.get("windows_loaded_this_round", 0) or 0)
    max_windows = max(1, int(state.get("max_windows_per_round", MAX_PROFILE_WINDOWS_PER_ROUND) or MAX_PROFILE_WINDOWS_PER_ROUND))
    profile_items_loaded = int(state.get("profile_items_loaded_this_round", 0) or 0)
    max_profile_items = max(1, int(state.get("max_profile_items_per_round", MAX_PROFILE_ITEMS_PER_ROUND) or MAX_PROFILE_ITEMS_PER_ROUND))
    window_exhausted = bool(current_window.get("exhausted")) if isinstance(current_window, dict) else False
    loaded_no_questions = not scored_questions
    prior_replenishment_exhausted = bool(state.get("data_replenishment_exhausted"))
    replenishment_state = {
        **state,
        "cross_dataset_pool": cross_pool,
        "quota_accumulated_questions": kept_questions,
        "quota_shortfall": dict(shortfall),
    }
    has_new_window_budget = (
        not prior_replenishment_exhausted
        and windows_loaded < max_windows
        and profile_items_loaded < max_profile_items
    )
    can_reuse_cached_window = (
        not prior_replenishment_exhausted
        and _has_cached_replenishment_ref(
            replenishment_state,
            shortfall,
            max_windows,
            max_profile_items,
        )
    )
    can_load_new_window = (
        not prior_replenishment_exhausted
        and has_new_window_budget
        and _has_replenishment_candidate(
            replenishment_state,
            active_dataset_id,
            window_exhausted,
        )
    )
    can_replenish = (
        bool(shortfall)
        and (can_reuse_cached_window or can_load_new_window)
    )
    # 如果配额不足且还有 budget，请求下一个 dataset window
    if can_replenish:
        next_dataset_ref = state.get("next_dataset_ref") or state.get("dataset_schema_info", {}).get("dataset_ref", {})
        print(
            "[filter] Quota shortfall remains; requesting next dataset window: "
            f"shortfall={shortfall}, selected_so_far={len(kept_questions)}, "
            f"pool_target={target}, train_target={train_target}, "
            f"windows={windows_loaded}/{max_windows}, items={profile_items_loaded}/{max_profile_items}, "
            f"cached_reuse={can_reuse_cached_window}"
        )
        return {
            "pending_message": pending_message,
            "cross_dataset_pool": cross_pool,
            "cached_rollout_scored_questions": [],
            **_replenishment_seen_state(state),
            "quota_met": False,
            "quota_shortfall": dict(shortfall),
            "quota_accumulated_questions": kept_questions,
            "data_replenishment_needed": True,
            "data_replenishment_exhausted": False,
            "replenishment_cycle_active": True,
            "next_dataset_ref": next_dataset_ref if isinstance(next_dataset_ref, dict) else {},
            "filtered_questions": [],
            "mastered_questions": mastered_questions,
            "round_data_stats": {
                **(state.get("round_data_stats") or {}),
                "quota_shortfall": dict(shortfall),
                "quota_selected_so_far": len(kept_questions),
                "quota_train_target": train_target,
                "quota_pool_target": target,
                "quota_accumulated_by_difficulty": dict(
                    Counter(_difficulty_key(q) for q in kept_questions)
                ),
                "needs_more_data": True,
            },
        }

    # Step 8: replenishment budget 耗尽，尝试从已有easy/hard按2:1替换补充
    quota_met = not shortfall
    data_replenishment_exhausted = bool(shortfall)
    if data_replenishment_exhausted and shortfall:
        reason_parts = []
        if prior_replenishment_exhausted:
            reason_parts.append("replenishment_exhausted")
        if window_exhausted:
            reason_parts.append("window_exhausted")
        if loaded_no_questions:
            reason_parts.append("no_questions_loaded")
        if windows_loaded >= max_windows:
            reason_parts.append("max_windows_reached")
        if profile_items_loaded >= max_profile_items:
            reason_parts.append("max_profile_items_reached")

        # Fallback: 从已加载窗口的easy/hard池补充medium缺口
        total_shortfall = sum(shortfall.values())
        # 从primary_pool中收集所有未选中的easy/hard题目
        fallback_pool = {"easy": [], "hard": []}
        kept_ids = {q.get("question_id") for q in kept_questions}

        for q in cross_pool.get(active_dataset_id, []):
            if q.get("question_id") not in kept_ids:
                diff = q.get("dynamic_difficulty")
                if diff in fallback_pool:
                    fallback_pool[diff].append(q)

        easy_available = len(fallback_pool["easy"])
        hard_available = len(fallback_pool["hard"])

        if easy_available + hard_available > 0 and total_shortfall > 0:
            import random
            # 按2:1从easy和hard抽取
            easy_take = min(easy_available, int(total_shortfall * 2 / 3))
            hard_take = min(hard_available, total_shortfall - easy_take)
            if easy_take < int(total_shortfall * 2 / 3):
                hard_take = min(hard_available, total_shortfall - easy_take)

            fallback_questions = []
            if easy_take > 0:
                fallback_questions.extend(random.sample(fallback_pool["easy"], easy_take))
            if hard_take > 0:
                fallback_questions.extend(random.sample(fallback_pool["hard"], hard_take))

            kept_questions.extend(fallback_questions)
            print(f"[filter] Fallback: supplemented {len(fallback_questions)} from window pool (easy={easy_take}, hard={hard_take}) to fill shortfall={total_shortfall}")

        print(
            "[filter] Proceeding with partial quota after replenishment budget exhausted: "
            f"shortfall={shortfall}, selected={len(kept_questions)}, "
            f"reason={'+'.join(reason_parts) or 'no_replenishment_available'}"
        )

    # Step 9: 最终裁剪并清理 pool
    kept_questions = kept_questions[:target]
    touched_dataset_ids = {
        str(q.get("source_dataset_id") or "")
        for q in kept_questions
        if q.get("source_dataset_id")
    }
    _remove_questions_from_pools(cross_pool, kept_questions)
    if data_replenishment_exhausted:
        _mark_below_threshold_exhausted(ds_mgr, touched_dataset_ids or {active_dataset_id}, target)

    source_ids = {str(q.get("source_dataset_id")) for q in kept_questions if q.get("source_dataset_id")}
    if len(source_ids) > 1:
        print(f"[filter] Cross-dataset supplement: sampled from {len(source_ids)} datasets ({sorted(source_ids)})")

    # Step 10: 混合 replay buffer
    replay_buffer = list(state.get("replay_buffer_entries", []))
    round_id = int(state.get("round_id", 0) or 0)
    sample_ratio = float(state.get("replay_sample_ratio_override", REPLAY_BUFFER_SAMPLE_RATIO))
    mixed_questions, selected_indices = mix_replay_and_new_data(
        new_questions=kept_questions,
        replay_buffer=replay_buffer,
        sample_ratio=sample_ratio,
        current_round=round_id,
        target_bucket=state.get("target_bucket", ""),
    )
    for idx in selected_indices:
        if 0 <= idx < len(replay_buffer):
            entry = replay_buffer[idx]
            used_rounds = entry.setdefault("used_in_rounds", [])
            if round_id not in used_rounds:
                used_rounds.append(round_id)

    # Step 11: 写出最终 filtered 结果
    questions_ref = write_json_artifact(
        trace_id=str(state.get("trace_id", "")),
        round_id=int(state.get("round_id", 0) or 0),
        producer="filter",
        name="filtered_questions",
        data=mixed_questions,
    )

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=str(state.get("trace_id", "")),
            round_id=int(state.get("round_id", 0) or 0),
            sender=AgentName.FILTER,
            receiver=AgentName.CLASSIFIER,
            message_type=MessageType.FILTER_RESULT,
        ),
        payload=FilterResultPayload(
            questions=[],
            questions_ref=questions_ref,
        ),
    )

    total_in_pool = sum(len(v) for v in cross_pool.values())
    print(f"[filter] Post-rollout: sampled {len(kept_questions)} from dataset={active_dataset_id} -> "
          f"mixed {len(mixed_questions)} with replay "
          f"(mastered={len(mastered_questions)}, pool_total={total_in_pool}, pool_datasets={len(cross_pool)})")

    return {
        "filtered_questions": mixed_questions,
        "mastered_questions": mastered_questions,
        "pending_message": msg,
        "cross_dataset_pool": cross_pool,
        "cached_rollout_scored_questions": [],
        **_replenishment_seen_state(state),
        "replay_buffer_entries": replay_buffer,
        "replay_buffer_used_count": len(selected_indices),
        "quota_met": quota_met,
        "quota_shortfall": dict(shortfall),
        "quota_accumulated_questions": [],
        "data_replenishment_needed": False,
        "data_replenishment_exhausted": data_replenishment_exhausted,
        "replenishment_cycle_active": False,
        "round_data_stats": {
            **(state.get("round_data_stats") or {}),
            "quota_shortfall": dict(shortfall),
            "quota_selected_so_far": len(kept_questions),
            "quota_train_target": train_target,
            "quota_pool_target": target,
            "quota_accumulated_by_difficulty": dict(
                Counter(_difficulty_key(q) for q in kept_questions)
            ),
            "needs_more_data": data_replenishment_exhausted,
        },
    }


def _sample_primary_with_shortfall(
    pool: list[dict],
    weights: dict[str, float],
    target_total: int,
    existing_selected: list[dict] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """按 difficulty 比例从单个数据集的 pool 中采样，同时计算短缺量。

    考虑已存在的 selected（quota_accumulated），每个 difficulty 桶的配额
    减去已有数后，再从 pool 中取不足的部分。
    如果 pool 中某个 difficulty 的可用题量不足，记录 shortfall。

    Returns:
        selected_questions: 本次新选中的题目
        shortfall_by_bucket: 每个 difficulty 还差多少题
    """
    by_difficulty: dict[str, list[dict]] = {}
    for q in pool:
        diff = str(q.get("dynamic_difficulty") or "unknown")
        by_difficulty.setdefault(diff, []).append(q)
    for items in by_difficulty.values():
        random.shuffle(items)

    quotas = _difficulty_quotas(weights, target_total)
    if not quotas:
        return pool[:target_total], {}

    selected: list[dict] = []
    shortfall: dict[str, int] = {}
    existing_selected = existing_selected or []
    used_ids = {key for q in existing_selected for key in _question_keys(q) if key[1]}
    existing_counts = Counter(_difficulty_key(q) for q in existing_selected)

    for diff, quota in sorted(quotas.items()):
        remaining_quota = max(0, quota - int(existing_counts.get(diff, 0) or 0))
        available = by_difficulty.get(diff, [])
        taken = _take_unique(available, remaining_quota, used_ids)
        selected.extend(taken)
        deficit = remaining_quota - len(taken)
        if deficit > 0:
            shortfall[diff] = deficit

    return selected, shortfall


def _supplement_from_pools(
    cross_pool: dict[str, list[dict]],
    exclude_dataset_id: str,
    shortfall: dict[str, int],
    ds_state_mgr,
    target_style: str | None = None,
) -> list[dict]:
    """从其他数据集的 pool 中按 difficulty 补充题目。

    仅从 dataset_state 标记为 'in_use' 的数据集中借用。
    选中的题目会从源 pool 中移除（避免重复选择）。
    可选参数 target_style 限制补充题目的 target_style 必须匹配主数据集风格。
    """
    if not shortfall:
        return []

    supplemented: list[dict] = []
    used_ids: set[tuple[str, object]] = set()
    active_dataset_ids: list[str] = []

    if ds_state_mgr is not None:
        for ds_id in cross_pool:
            if ds_id == exclude_dataset_id:
                continue
            ds = ds_state_mgr.datasets.get(ds_id) or ds_state_mgr.load_cached(ds_id)
            if ds is not None and ds.state == "in_use":
                active_dataset_ids.append(ds_id)
    else:
        active_dataset_ids = [did for did in cross_pool if did != exclude_dataset_id]

    required_style = normalize_target_style(target_style)
    print(
        f"[filter] Cross-dataset supplement needed: {shortfall}, "
        f"available datasets={active_dataset_ids}, target_style={required_style or 'any'}"
    )

    for difficulty, needed in list(shortfall.items()):
        remaining = needed
        for ds_id in active_dataset_ids:
            if remaining <= 0:
                break
            pool = cross_pool.get(ds_id, [])
            matching = [
                q for q in _filter_unused_pool(pool, ds_state_mgr, ds_id)
                if str(q.get("dynamic_difficulty") or "unknown") == difficulty
                and (required_style is None or normalize_target_style(q.get("target_style")) == required_style)
            ]
            if not matching:
                continue
            taken = _take_unique(matching, remaining, used_ids)
            supplemented.extend(taken)
            remaining -= len(taken)
            taken_ids = {key for q in taken for key in _question_keys(q)}
            cross_pool[ds_id] = [
                q for q in pool if not (_question_keys(q) & taken_ids)
            ]
            print(
                f"[filter] Supplemented {len(taken)} {difficulty} from {ds_id} "
                f"(remaining_need={remaining}, pool_left={len(cross_pool[ds_id])})"
            )
        if remaining > 0:
            print(f"[filter] Warning: could not fill {difficulty} shortfall of {remaining}")

    return supplemented
