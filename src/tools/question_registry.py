"""
题目注册表（Question Registry）—— 跨轮次题目去重与状态追踪

核心职责：
1. 记录每道题目的使用状态（train_seen / active_holdout / probe_holdout / retired_holdout / external_probe），
   防止同一道题在训练、验证、测试之间泄露（data leakage）
2. 支持按 question_id（精确匹配）和 text_hash（内容哈希匹配）双重去重
3. 提供 drop_registered_questions 接口，从候选池中过滤掉已注册的题目，
   确保每轮训练/验证/测试的题目集合互不重叠

状态生命周期：
  新题目 → active_holdout（预留验证/测试池）
       ↓
  train_seen（进入训练集后标记，禁止再进入测试集）
       ↓
  retired_holdout（已退役，可重新进入训练池但禁止回测试集）

  probe_holdout（探针集专用，永久禁止训练和测试）
  external_probe（外部预置探针，永久禁止训练和测试）

使用场景：
  bootstrap → mark_questions_probe_holdout / mark_questions_external_probe → 标记探针题
  data_builder → drop_registered_questions → 过滤训练/测试候选池
  rollout → mark_questions_train_seen → 记录已训练的题目
  strategy_inspector → retire_holdout_questions → 退役老题目
"""

import hashlib
import json
from pathlib import Path

# 训练池屏蔽状态：处于这些状态的题目不得进入训练集
TRAIN_BLOCKING_STATUSES = frozenset({"active_holdout", "probe_holdout", "external_probe"})

# 测试池屏蔽状态：处于这些状态的题目不得进入验证/测试集（比训练更严格）
TEST_BLOCKING_STATUSES = frozenset({
    "active_holdout",
    "probe_holdout",
    "retired_holdout",
    "train_seen",
    "external_probe",
})

# 默认状态：新增题目首次注册时使用
LEGACY_STATUS = "active_holdout"


def question_text_hash(question: dict) -> str:
    """对题目文本做 SHA256 哈希，生成内容指纹。

    当 question_id 缺失或不同数据源使用不同 ID 体系时，
    通过文本哈希判断题目是否相同，作为 question_id 匹配的补充。

    Args:
        question: 包含 question_text 字段的题目字典

    Returns:
        64 位十六进制哈希字符串
    """
    text = question.get("question_text", "")
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_question_registry(path: str | None) -> dict:
    """加载题目注册表文件，不存在或格式异常时返回空注册表。

    兼容 v1 格式（单独的 question_ids / text_hashes 列表）
    和 v2 格式（统一的 entries 列表，每项含 question_id + text_hash + status）。

    Args:
        path: 注册表 JSON 文件路径

    Returns:
        注册表字典，结构: {"version": 2, "entries": [...], "question_ids": [...], "text_hashes": [...]}
    """
    if not path or not Path(path).exists():
        return {"version": 2, "entries": [], "question_ids": [], "text_hashes": []}
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        return {"version": 2, "entries": [], "question_ids": [], "text_hashes": []}
    if isinstance(loaded.get("entries"), list):
        entries = [e for e in loaded.get("entries", []) if isinstance(e, dict)]
        return {
            "version": int(loaded.get("version", 2) or 2),
            "entries": entries,
            "question_ids": list(loaded.get("question_ids", [])),
            "text_hashes": list(loaded.get("text_hashes", [])),
        }
    entries = []
    for qid in loaded.get("question_ids", []):
        entries.append({"question_id": qid, "text_hash": "", "status": LEGACY_STATUS})
    for text_hash in loaded.get("text_hashes", []):
        entries.append({"question_id": "", "text_hash": text_hash, "status": LEGACY_STATUS})
    return {
        "version": 2,
        "entries": entries,
        "question_ids": list(loaded.get("question_ids", [])),
        "text_hashes": list(loaded.get("text_hashes", [])),
    }


def _entry_key(entry: dict) -> tuple[str, str, str, str, str, str]:
    """提取 entry 的唯一标识，优先保留数据集逐行身份。"""
    return (
        str(entry.get("source_dataset_id", "") or ""),
        str(entry.get("source_dataset_subset", "") or ""),
        str(entry.get("source_dataset_split", "") or ""),
        str(entry.get("source_dataset_row_id", "") or ""),
        str(entry.get("question_id", "") or ""),
        str(entry.get("text_hash", "") or ""),
    )


def _question_key(question: dict) -> tuple[str, str, str, str, str, str]:
    """提取题目的唯一标识，优先保留数据集逐行身份。"""
    if not isinstance(question, dict):
        return "", "", "", "", "", ""
    return (
        str(question.get("source_dataset_id", "") or ""),
        str(question.get("source_dataset_subset", "") or ""),
        str(question.get("source_dataset_split", "") or ""),
        str(question.get("source_dataset_row_id", "") or ""),
        str(question.get("question_id", "") or ""),
        question_text_hash(question),
    )


def _normalize_registry(registry: dict) -> dict:
    """对注册表做去重和归一化：同一道题有多条记录时，保留状态优先级最高的一条。

    状态优先级（从高到低）：
      external_probe > probe_holdout > active_holdout > retired_holdout > train_seen

    去重键 = (question_id, text_hash)，优先使用 question_id。

    Args:
        registry: 可能含重复条目的注册表

    Returns:
        去重归一化后的注册表
    """
    by_key: dict[tuple[str, str], dict] = {}
    status_rank = {
        "external_probe": 5,
        "probe_holdout": 4,
        "active_holdout": 3,
        "retired_holdout": 2,
        "train_seen": 1,
    }
    for entry in registry.get("entries", []):
        if not isinstance(entry, dict):
            continue
        dataset_id, subset, split, row_id, qid, text_hash = _entry_key(entry)
        if not qid and not text_hash and not (dataset_id and row_id):
            continue
        status = str(entry.get("status") or LEGACY_STATUS)
        key = (dataset_id, subset, split, row_id, qid, text_hash)
        previous = by_key.get(key)
        if previous is None or status_rank.get(status, 0) >= status_rank.get(str(previous.get("status", "")), 0):
            normalized = dict(entry)
            normalized["source_dataset_id"] = dataset_id or normalized.get("source_dataset_id")
            normalized["source_dataset_subset"] = subset or normalized.get("source_dataset_subset")
            normalized["source_dataset_split"] = split or normalized.get("source_dataset_split")
            normalized["source_dataset_row_id"] = row_id or normalized.get("source_dataset_row_id")
            normalized["question_id"] = qid
            normalized["text_hash"] = text_hash
            normalized["status"] = status
            by_key[key] = normalized

    entries = list(by_key.values())
    question_ids = sorted({e["question_id"] for e in entries if e.get("question_id")})
    text_hashes = sorted({e["text_hash"] for e in entries if e.get("text_hash")})
    return {
        "version": 2,
        "entries": sorted(entries, key=lambda e: (e.get("status", ""), e.get("question_id", ""), e.get("text_hash", ""))),
        "question_ids": question_ids,
        "text_hashes": text_hashes,
    }


def save_question_registry(path: str, registry: dict) -> None:
    """归一化并持久化注册表到文件。

    Args:
        path: 注册表 JSON 文件路径
        registry: 待持久化的注册表字典
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_registry(registry)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


def add_questions_to_registry(
    path: str,
    questions: list[dict],
    status: str = LEGACY_STATUS,
    metadata: dict | None = None,
) -> dict:
    """将题目批量加入注册表，标记为指定状态。

    每道题目记录 question_id、text_hash、source_dataset_id、module 等元信息。

    Args:
        path: 注册表文件路径
        questions: 待注册的题目列表
        status: 状态标记（如 "train_seen", "probe_holdout"）
        metadata: 可选的附加元数据

    Returns:
        更新后的注册表
    """
    registry = load_question_registry(path)
    entries = list(registry.get("entries", []))

    for q in questions:
        qid = q.get("question_id", "") if isinstance(q, dict) else ""
        text_hash = question_text_hash(q) if isinstance(q, dict) else ""
        row_id = str(q.get("source_dataset_row_id") or "") if isinstance(q, dict) else ""
        dataset_id = str(q.get("source_dataset_id") or "") if isinstance(q, dict) else ""
        if not qid and not text_hash and not (dataset_id and row_id):
            continue
        entry = {
            "question_id": qid,
            "text_hash": text_hash,
            "status": status,
            "source_dataset_id": q.get("source_dataset_id") if isinstance(q, dict) else None,
            "source_dataset_row_id": q.get("source_dataset_row_id") if isinstance(q, dict) else None,
            "source_dataset_split": q.get("source_dataset_split") if isinstance(q, dict) else None,
            "source_dataset_subset": q.get("source_dataset_subset") if isinstance(q, dict) else None,
            "module": (q.get("module") or q.get("category")) if isinstance(q, dict) else None,
            "dynamic_difficulty": q.get("dynamic_difficulty") if isinstance(q, dict) else None,
        }
        if metadata:
            entry["metadata"] = metadata
        entries.append(entry)

    updated = {"version": 2, "entries": entries}
    save_question_registry(path, updated)
    return load_question_registry(path)


def _registered_key_sets(
    registry: dict,
    blocking_statuses: frozenset[str],
) -> tuple[set[str], set[str], set[tuple[str, str, str, str]]]:
    """从注册表中提取所有处于屏蔽状态的 question_id 和 text_hash 集合。

    Args:
        registry: 注册表
        blocking_statuses: 屏蔽状态集合（TRAIN_BLOCKING_STATUSES 或 TEST_BLOCKING_STATUSES）

    Returns:
        (registered_ids, registered_hashes) 两个集合
    """
    registered_ids: set[str] = set()
    registered_hashes: set[str] = set()
    registered_rows: set[tuple[str, str, str, str]] = set()
    for entry in registry.get("entries", []):
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or LEGACY_STATUS)
        if status not in blocking_statuses:
            continue
        qid = str(entry.get("question_id", "") or "")
        text_hash = str(entry.get("text_hash", "") or "")
        dataset_id = str(entry.get("source_dataset_id", "") or "")
        row_id = str(entry.get("source_dataset_row_id", "") or "")
        if qid:
            registered_ids.add(qid)
        if text_hash and not (dataset_id and row_id):
            registered_hashes.add(text_hash)
        if dataset_id and row_id:
            registered_rows.add((
                dataset_id,
                str(entry.get("source_dataset_subset", "") or ""),
                str(entry.get("source_dataset_split", "") or ""),
                row_id,
            ))
    return registered_ids, registered_hashes, registered_rows


def drop_registered_questions(
    questions: list[dict],
    registry_path: str | None,
    block_statuses: frozenset[str] | None = None,
) -> tuple[list[dict], int]:
    """从候选题目池中过滤掉已注册的题目。

    这是防止数据泄露的核心函数：每轮训练/测试前，用此函数确保
    候选池中的题目未在注册表中出现（以屏蔽状态出现）。

    Args:
        questions: 候选题目列表
        registry_path: 注册表文件路径
        block_statuses: 屏蔽状态集合，默认 TRAIN_BLOCKING_STATUSES

    Returns:
        (保留的题目列表, 被过滤掉的题目数量)
    """
    registry = load_question_registry(registry_path)
    registered_ids, registered_hashes, registered_rows = _registered_key_sets(
        registry,
        block_statuses or TRAIN_BLOCKING_STATUSES,
    )
    if not registered_ids and not registered_hashes and not registered_rows:
        return questions, 0

    kept = []
    dropped = 0
    for q in questions:
        qid = q.get("question_id", "") if isinstance(q, dict) else ""
        text_hash = question_text_hash(q) if isinstance(q, dict) else ""
        row_key = (
            str(q.get("source_dataset_id", "") or ""),
            str(q.get("source_dataset_subset", "") or ""),
            str(q.get("source_dataset_split", "") or ""),
            str(q.get("source_dataset_row_id", "") or ""),
        ) if isinstance(q, dict) else ("", "", "", "")
        row_registered = bool(row_key[0] and row_key[3] and row_key in registered_rows)
        text_registered = bool(text_hash and not (row_key[0] and row_key[3]) and text_hash in registered_hashes)
        if row_registered or (qid and qid in registered_ids) or text_registered:
            dropped += 1
            continue
        kept.append(q)

    return kept, dropped


# ── 便捷标记函数：将题目批量标记为特定状态 ─────────────────────

def mark_questions_train_seen(path: str, questions: list[dict], metadata: dict | None = None) -> dict:
    """标记题目已进入训练集（禁止重新进入测试集）。"""
    return add_questions_to_registry(path, questions, status="train_seen", metadata=metadata)


def mark_questions_active_holdout(path: str, questions: list[dict], metadata: dict | None = None) -> dict:
    """标记题目为活跃留出（禁止训练，允许测试）。"""
    return add_questions_to_registry(path, questions, status="active_holdout", metadata=metadata)


def mark_questions_probe_holdout(path: str, questions: list[dict], metadata: dict | None = None) -> dict:
    """标记题目为探针留出（永久禁止训练和测试，仅用于探针集评估）。"""
    return add_questions_to_registry(path, questions, status="probe_holdout", metadata=metadata)


def mark_questions_external_probe(path: str, questions: list[dict], metadata: dict | None = None) -> dict:
    """标记题目为外部探针（预置题目，永久禁止训练和测试）。"""
    return add_questions_to_registry(path, questions, status="external_probe", metadata=metadata)


def retire_holdout_questions(path: str, questions: list[dict], metadata: dict | None = None) -> dict:
    """退役留出题目：从 holdout 转为 retired，允许重新进入训练池（但仍禁止测试）。"""
    return add_questions_to_registry(path, questions, status="retired_holdout", metadata=metadata)
