"""Dataset reviewer node."""

import importlib
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, cast

from src.models.messages import (
    AgentName,
    DatasetRef,
    MessageHeader,
    MessageType,
    RoutedMessage,
    SearchResultPayload,
)
from src.models.state import EvoState
from config.settings import (
    DATA_CLEANER_MAX_REPAIR_ATTEMPTS,
    DATASET_REVIEW_FAILURE_BACKOFF_SECONDS,
    DATASET_REVIEW_FAILURES_BEFORE_BACKOFF,
    DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS,
    DATASET_REVIEW_TIMEOUT_KILL_GRACE_SECONDS,
    get_session_dir,
)
from src.tools.agent_prompts import DATASET_REVIEWER_PROMPT
from src.tools.dataset_cleaner_codegen import (
    CleanerCodegenRequest,
    build_cleaner_provider,
    ensure_cleaner_for_ref,
)
from src.tools.dataset_adapter import (
    detect_schema_from_item,
    get_existing_hfd_dataset_source,
    load_hf_dataset_with_fallback,
)
from src.tools.dataset_state import DatasetStateManager
from src.tools.hf_dataset_card import load_dataset_card_summary, patch_hub_timeout
from src.tools.llm_decision import decide_json, prompt_for_agent

_REVIEW_SAMPLE_SIZE = int(os.getenv("DATASET_REVIEW_SAMPLE_ROWS", "3"))
_MAX_REVIEW_CONFIG_CANDIDATES = 12
_MAX_REVIEW_SPLIT_CANDIDATES = 10
_DATASET_CARD_TEXT_CHAR_LIMIT = int(os.getenv("DATASET_REVIEW_CARD_TEXT_CHAR_LIMIT", "12000"))
_DATASET_REVIEW_WORKER_COUNT = max(1, int(os.getenv("DATASET_REVIEW_WORKER_COUNT", "2")))
_DATASETS_SERVER_BASE_URL = os.getenv(
    "DATASETS_SERVER_BASE_URL",
    "https://datasets-server.huggingface.co",
).rstrip("/")
_DATASETS_SERVER_ROWS_ENABLED = os.getenv(
    "DATASET_REVIEW_USE_DATASETS_SERVER_ROWS",
    "1",
).strip().lower() in ("1", "true", "yes", "on")
_DATASETS_SERVER_ROWS_TIMEOUT = float(os.getenv("DATASET_REVIEW_ROWS_TIMEOUT_SECONDS", "60"))
_REVIEW_STREAMING_TIMEOUT = float(os.getenv("DATASET_REVIEW_STREAMING_TIMEOUT_SECONDS", "60"))
_REVIEW_ALLOW_HFD = os.getenv(
    "DATASET_REVIEW_ALLOW_HFD_DURING_REVIEW",
    "0",
).strip().lower() in ("1", "true", "yes", "on")

_FIRST_ACCEPT_WAIT_SECONDS = float(os.getenv("DATASET_REVIEW_FIRST_ACCEPT_TIMEOUT_SECONDS", "120"))
_REPLENISHMENT_ACCEPT_WAIT_SECONDS = float(os.getenv("DATASET_REVIEW_REPLENISHMENT_WAIT_SECONDS", "120"))

_GENERIC_CONFIG_CANDIDATES: tuple[str | None, ...] = (
    None,
    "main",
)
_GENERIC_SPLIT_CANDIDATES = (
    "train",
    "validation",
    "test",
)


def _review_hfd_cache_only_enabled() -> bool:
    cache_mode = os.getenv("DATASET_CACHE_MODE", "").strip().lower()
    return (
        os.getenv("HFD_DATASET_CACHE_ONLY", "").strip().lower() in ("1", "true", "yes", "on")
        or os.getenv("DATASET_OFFSET_CACHE_MODE", "").strip().lower() in ("1", "true", "yes", "on")
        or cache_mode in {"offset", "offline", "cache_only", "cache-only"}
    )


def _allow_hfd_for_review_load() -> bool:
    return bool(_REVIEW_ALLOW_HFD or _review_hfd_cache_only_enabled())

_TRUE_PROBLEM_FIELD_NAMES = {
    "question",
    "problem",
    "prompt",
    "input",
    "instruction",
    "informal_statement",
    "formal_statement",
    "theorem_statement",
    "statement",
}

_OPAQUE_ID_FIELD_NAMES = {
    "id",
    "uid",
    "uuid",
    "name",
    "title",
    "slug",
    "filename",
    "file_name",
    "path",
}

_FORMAL_PROOF_FIELD_NAMES = {
    "formal_proof",
    "proof",
    "lean_proof",
    "coq_proof",
    "isabelle_proof",
    "formalization",
}

_FORMAL_CODE_MARKERS = (
    "import ",
    "theorem ",
    "lemma ",
    "example ",
    ":=",
    "begin",
    "end",
    "by ",
    "norm_num",
    "rw ",
    "refl",
    "qed",
)

_POLLUTED_TRANSCRIPT_MARKERS = (
    "### human",
    "### user",
    "### assistant",
    "### assistance",
    "assistant:**",
    "assistant: **",
    "human:",
    "user:",
)

_GENERATED_REFERENCE_TEMPLATE_MARKERS = (
    "decode the problem",
    "information retrieval and association",
    "information analysis and integration",
    "response development",
    "response and answer formulation",
    "post-response reflection",
    "knowledge graph",
)

_GENERIC_SINGLE_TEXT_FIELDS = {
    "text",
    "prompt",
    "instruction",
    "input",
    "query",
    "context",
}


def _normalised_column_names(values: list[Any]) -> set[str]:
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _looks_like_opaque_identifier(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lower = text.lower()
    if " " in text or "\n" in text:
        return False
    if any(marker in lower for marker in ("__", "round", "grade", "word_problem", "theorem_proving")):
        return True
    alpha_num = sum(ch.isalnum() for ch in text)
    separators = sum(ch in "_-./:" for ch in text)
    return len(text) >= 16 and separators >= 2 and alpha_num >= 8


def _looks_like_formal_proof_code(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    marker_hits = sum(1 for marker in _FORMAL_CODE_MARKERS if marker in text)
    return marker_hits >= 2 or ("import " in text and ("begin" in text or ":=" in text))


def _formal_proof_only_reject_reason(resolved_ref: dict) -> str | None:
    columns = _normalised_column_names(resolved_ref.get("source_dataset_columns") or [])
    if not columns:
        return None
    if columns & _TRUE_PROBLEM_FIELD_NAMES:
        return None
    if not (columns & _OPAQUE_ID_FIELD_NAMES and columns & _FORMAL_PROOF_FIELD_NAMES):
        return None

    rows = [
        row for row in resolved_ref.get("source_dataset_raw_rows", [])
        if isinstance(row, dict)
    ]
    first_row = resolved_ref.get("source_dataset_first_row")
    if isinstance(first_row, dict):
        rows = [first_row, *rows]
    if not rows:
        return None

    id_fields = [field for field in _OPAQUE_ID_FIELD_NAMES if field in columns]
    proof_fields = [field for field in _FORMAL_PROOF_FIELD_NAMES if field in columns]
    evidence_rows = 0
    for row in rows[:3]:
        lower_row = {str(key).strip().lower(): value for key, value in row.items()}
        id_value = next((lower_row.get(field) for field in id_fields if field in lower_row), "")
        proof_value = next((lower_row.get(field) for field in proof_fields if field in lower_row), "")
        if _looks_like_opaque_identifier(id_value) and _looks_like_formal_proof_code(proof_value):
            evidence_rows += 1

    if evidence_rows:
        return "reject: opaque id/name plus formal proof code without problem statement"
    return None


def _has_independent_gold_or_reference_schema(schema: dict[str, Any]) -> bool:
    for field in (
        schema.get("answer_field"),
        schema.get("rollout_gold_field"),
        schema.get("train_output_field"),
    ):
        if isinstance(field, str) and field.strip():
            return True
    return False


def _looks_like_polluted_single_text_reference(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    transcript_hit = any(marker in text for marker in _POLLUTED_TRANSCRIPT_MARKERS)
    template_hits = sum(1 for marker in _GENERATED_REFERENCE_TEMPLATE_MARKERS if marker in text)
    generated_preamble = any(
        marker in text
        for marker in (
            "novel creative puzzle",
            "ai companion",
            "fictional example",
            "based on the question and answer you provided",
        )
    )
    if transcript_hit and template_hits >= 2:
        return True
    if generated_preamble and template_hits >= 2:
        return True
    return template_hits >= 4 and ("solution:" in text or "final response" in text)


def _polluted_single_text_reject_reason(resolved_ref: dict) -> str | None:
    schema = resolved_ref.get("source_dataset_schema")
    if not isinstance(schema, dict):
        schema = {}
    if _has_independent_gold_or_reference_schema(schema):
        return None

    columns = _normalised_column_names(resolved_ref.get("source_dataset_columns") or [])
    question_field = str(schema.get("question_field") or "").strip().lower()
    if not (
        columns <= _GENERIC_SINGLE_TEXT_FIELDS
        or question_field in _GENERIC_SINGLE_TEXT_FIELDS
        or str(schema.get("reason") or "").strip().lower() == "fallback schema"
    ):
        return None

    rows = [
        row for row in resolved_ref.get("source_dataset_raw_rows", [])
        if isinstance(row, dict)
    ]
    first_row = resolved_ref.get("source_dataset_first_row")
    if isinstance(first_row, dict):
        rows = [first_row, *rows]
    if not rows:
        return None

    evidence_rows = 0
    for row in rows[:3]:
        values = [row.get(question_field)] if question_field and question_field in row else list(row.values())
        if any(_looks_like_polluted_single_text_reference(value) for value in values):
            evidence_rows += 1
    if evidence_rows:
        return "reject: polluted transcript/reference solution in single text without independent answer"
    return None


class _ReviewJob:
    def __init__(self, job_id: str, refs: list[dict], state: dict[str, Any]) -> None:
        self.job_id = job_id
        self.refs = refs
        self.state = state
        self.results: list[dict[str, Any]] = []
        self.completed = False
        self.error = ""
        self.condition = threading.Condition()
        self.thread: threading.Thread | None = None


_REVIEW_JOBS: dict[str, _ReviewJob] = {}
_REVIEW_JOBS_LOCK = threading.Lock()


def _transient_review_reject_reason(reason: str) -> bool:
    text = str(reason or "").lower()
    transient_markers = (
        "no usable review samples",
        "review timed out",
        "review worker",
        "blacklisted",
        "without result",
        "aborted",
        "process exited",
        "datasets_server_rows",
        "streaming",
        "network",
        "timeout",
    )
    return any(marker in text for marker in transient_markers)


def _reject_verdict_for_ref(raw_ref: dict, reason: str, *, failure_stage: str = "") -> dict:
    transient = _transient_review_reject_reason(reason)
    return {
        "dataset_id": str(raw_ref.get("dataset_id", "unknown")),
        "verdict": "reject",
        "reason": reason,
        "suitability_score": 0.0,
        "sample_count": 0,
        "transient": transient,
        "cache_policy": "backoff" if transient else "stable",
        "failure_stage": str(failure_stage or ""),
        "review_ref": _review_ref_for_cache(raw_ref),
        "resolved_ref": _resolved_ref(
            raw_ref,
            raw_ref.get("subset"),
            raw_ref.get("split") or "train",
        ),
    }


# ═══════════════════════════════════════════════════════════
#  名称标准化（config/subset/split 的统一处理）
# ═══════════════════════════════════════════════════════════

def _normalise_config_name(value: Any) -> str | None:
    """标准化 config/subset 名称。None/"default" → None（即 HF 默认）。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "default":
        return None
    return text


def _normalise_split_name(value: Any) -> str | None:
    """标准化 split 名称。空字符串 → None。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique_config_candidates(values: list[Any]) -> list[str | None]:
    """去重后的 config 候选列表，保持首次出现顺序。"""
    seen: set[str] = set()
    candidates: list[str | None] = []
    for value in values:
        candidate = _normalise_config_name(value)
        key = candidate if candidate is not None else ""
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _unique_split_candidates(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for value in values:
        candidate = _normalise_split_name(value)
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def _datasets_module() -> Any:
    return importlib.import_module("datasets")


def _short_exception(exc: Exception, max_len: int = 240) -> str:
    """将异常转换为短字符串，过长时截断。"""
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


_HF_METADATA_NETWORK_TIMEOUT = float(os.getenv("HF_METADATA_NETWORK_TIMEOUT_SECONDS", "180"))


def _hf_metadata_call(call: Callable[[], Any]) -> list[str]:
    restore = patch_hub_timeout(_HF_METADATA_NETWORK_TIMEOUT)
    try:
        names = call()
        return [str(name).strip() for name in names if str(name).strip()]
    finally:
        restore()


def _available_config_names_info(dataset_id: str) -> tuple[list[str], str]:
    """查询数据集的所有可用 config 名称。返回 (名称列表, 错误信息)。"""
    get_config_names = getattr(_datasets_module(), "get_dataset_config_names")
    try:
        return _hf_metadata_call(lambda: get_config_names(dataset_id)), ""
    except Exception as exc:
        reason = _short_exception(exc)
        print(
            f"[dataset_reviewer] {dataset_id}: stage=config_metadata "
            f"failed={reason}"
        )
        return [], reason


def _available_split_names_info(dataset_id: str, config_name: str | None) -> tuple[list[str], str]:
    """查询指定 config 的所有可用 split 名称。返回 (名称列表, 错误信息)。

    兼容新版（keyword arg）和旧版（positional arg）的 get_dataset_split_names API。
    """
    try:
        get_split_names = getattr(_datasets_module(), "get_dataset_split_names")
    except Exception:
        return [], "datasets module has no get_dataset_split_names"

    def _call_get_split_names():
        try:
            return get_split_names(dataset_id, config_name=config_name)
        except TypeError:
            if config_name is None:
                return get_split_names(dataset_id)
            else:
                return get_split_names(dataset_id, config_name)

    try:
        return _hf_metadata_call(_call_get_split_names), ""
    except Exception as exc:
        reason = _short_exception(exc)
        print(
            f"[dataset_reviewer] {dataset_id}: stage=split_metadata "
            f"subset={config_name!r} failed={reason}"
        )
        return [], reason


def _metadata_auth_headers() -> dict[str, str]:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
        or ""
    ).strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _datasets_server_splits_info(dataset_id: str) -> tuple[list[dict[str, str | None]], str]:
    """Query datasets-server /splits for config/split metadata.

    This is a fallback for cases where datasets.get_dataset_* metadata
    resolution fails, but the dataset viewer has already indexed the dataset.
    """
    query = urllib.parse.urlencode({"dataset": dataset_id})
    url = f"{_DATASETS_SERVER_BASE_URL}/splits?{query}"
    try:
        request = urllib.request.Request(url, headers=_metadata_auth_headers())
        with urllib.request.urlopen(request, timeout=_HF_METADATA_NETWORK_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        reason = _short_exception(exc)
        print(
            f"[dataset_reviewer] {dataset_id}: stage=datasets_server_splits "
            f"failed={reason}"
        )
        return [], reason

    raw_splits = payload.get("splits") if isinstance(payload, dict) else None
    if not isinstance(raw_splits, list):
        return [], "datasets-server response has no splits list"

    entries: list[dict[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_splits:
        if not isinstance(item, dict):
            continue
        split = _normalise_split_name(item.get("split"))
        if split is None:
            continue
        config = _normalise_config_name(item.get("config"))
        key = (config or "", split)
        if key in seen:
            continue
        seen.add(key)
        entries.append({"config": config, "split": split})
    return entries, ""


def _datasets_server_rows_info(
    dataset_id: str,
    subset: str | None,
    split: str,
) -> tuple[list[dict[str, Any]], str]:
    """Query datasets-server /rows for lightweight review samples."""
    if not _DATASETS_SERVER_ROWS_ENABLED:
        return [], "datasets-server rows disabled"
    query = {
        "dataset": dataset_id,
        "split": split,
        "offset": "0",
        "length": str(max(1, int(_REVIEW_SAMPLE_SIZE))),
    }
    normalized_subset = _normalise_config_name(subset)
    if normalized_subset is not None:
        query["config"] = normalized_subset
    url = f"{_DATASETS_SERVER_BASE_URL}/rows?{urllib.parse.urlencode(query)}"
    try:
        request = urllib.request.Request(url, headers=_metadata_auth_headers())
        with urllib.request.urlopen(request, timeout=_DATASETS_SERVER_ROWS_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        reason = _short_exception(exc)
        print(
            f"[dataset_reviewer] {dataset_id}: stage=datasets_server_rows "
            f"subset={normalized_subset!r} split={split!r} failed={reason}"
        )
        return [], reason

    raw_rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(raw_rows, list):
        return [], "datasets-server response has no rows list"

    rows: list[dict[str, Any]] = []
    for item in raw_rows[:_REVIEW_SAMPLE_SIZE]:
        row = item.get("row") if isinstance(item, dict) else item
        if isinstance(row, dict):
            rows.append(_json_safe_sample_row(row))
    if not rows:
        return [], "datasets-server rows response had no usable row payloads"
    return rows, ""


def _hfd_local_split_entries(dataset_id: str) -> tuple[list[dict[str, str | None]], str]:
    """Use hfd/local metadata as a late split discovery fallback."""
    if os.getenv("USE_HFD_DATASET_DOWNLOAD", "").strip().lower() not in ("1", "true", "yes", "on"):
        return [], "hfd disabled"
    dataset_source = get_existing_hfd_dataset_source(dataset_id)
    if not dataset_source:
        return [], "hfd local source unavailable"

    try:
        get_config_names = getattr(_datasets_module(), "get_dataset_config_names")
        get_split_names = getattr(_datasets_module(), "get_dataset_split_names")
    except Exception as exc:
        return [], _short_exception(exc)

    try:
        raw_configs = get_config_names(dataset_source)
    except Exception:
        raw_configs = []
    config_candidates = _unique_config_candidates(list(raw_configs)) or [None]

    entries: list[dict[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for config in config_candidates[:_MAX_REVIEW_CONFIG_CANDIDATES]:
        try:
            raw_splits = get_split_names(dataset_source, config_name=config)
        except TypeError:
            try:
                raw_splits = get_split_names(dataset_source) if config is None else get_split_names(dataset_source, config)
            except Exception:
                raw_splits = []
        except Exception:
            raw_splits = []
        for split in _unique_split_candidates(list(raw_splits))[:_MAX_REVIEW_SPLIT_CANDIDATES]:
            key = (config or "", split)
            if key in seen:
                continue
            seen.add(key)
            entries.append({"config": config, "split": split})
    return entries, "" if entries else "hfd local metadata has no splits"


def _config_order_from_split_entries(
    entries: list[dict[str, str | None]],
    requested_subset: Any,
) -> list[str | None]:
    configs = _unique_config_candidates([entry.get("config") for entry in entries])
    requested = _normalise_config_name(requested_subset)
    if requested in configs:
        return [requested, *[config for config in configs if config != requested]][:_MAX_REVIEW_CONFIG_CANDIDATES]
    return configs[:_MAX_REVIEW_CONFIG_CANDIDATES]


def _split_names_for_entry_config(
    entries: list[dict[str, str | None]],
    config: str | None,
) -> list[str]:
    normalized = _normalise_config_name(config)
    return _unique_split_candidates([
        entry.get("split")
        for entry in entries
        if _normalise_config_name(entry.get("config")) == normalized
    ])


def discover_config_splits(
    dataset_id: str,
    requested_subset: Any = None,
    requested_split: Any = None,
) -> tuple[list[tuple[str | None, list[str]]], list[dict[str, str]]]:
    """Discover config/split candidates, keeping datasets.get_* as first pass.

    Returns (candidates, attempts). Fallbacks are intentionally
    ordered: datasets metadata, datasets-server /splits, hfd/local metadata.
    """
    def attempt(stage: str, subset: str | None, split: str, detail: str) -> dict[str, str]:
        return {
            "stage": stage,
            "subset": repr(_normalise_config_name(subset)),
            "split": str(split),
            "detail": detail[:240],
        }

    config_candidates = _config_candidates(dataset_id, requested_subset)
    candidates: list[tuple[str | None, list[str]]] = []
    attempts: list[dict[str, str]] = []
    metadata_error_seen = False
    for subset in config_candidates:
        split_names, split_metadata_error = _available_split_names_info(dataset_id, subset)
        if not split_names:
            attempts.append(
                attempt(
                    "split_metadata",
                    subset,
                    str(requested_split or "train"),
                    split_metadata_error or "no split metadata",
                )
            )
            if split_metadata_error:
                metadata_error_seen = True
                break
            continue
        candidates.append((subset, split_names))
    if candidates or not metadata_error_seen:
        return candidates, attempts

    server_entries, _server_error = _datasets_server_splits_info(dataset_id)
    server_candidates = [
        (subset, _split_names_for_entry_config(server_entries, subset))
        for subset in _config_order_from_split_entries(server_entries, requested_subset)
    ]
    server_candidates = [(subset, splits) for subset, splits in server_candidates if splits]
    if server_candidates:
        return server_candidates, attempts

    hfd_entries, _hfd_error = _hfd_local_split_entries(dataset_id)
    hfd_candidates = [
        (subset, _split_names_for_entry_config(hfd_entries, subset))
        for subset in _config_order_from_split_entries(hfd_entries, requested_subset)
    ]
    return [(subset, splits) for subset, splits in hfd_candidates if splits], attempts


# ═══════════════════════════════════════════════════════════
#  Config/Split 候选生成
# ═══════════════════════════════════════════════════════════

def _config_candidates(dataset_id: str, requested_subset: Any) -> list[str | None]:
    metadata_names, _metadata_error = _available_config_names_info(dataset_id)
    metadata_configs = _unique_config_candidates(metadata_names)
    if metadata_configs:
        requested = _normalise_config_name(requested_subset)
        if requested in metadata_configs:
            return [requested, *[config for config in metadata_configs if config != requested]][:_MAX_REVIEW_CONFIG_CANDIDATES]
        return metadata_configs[:_MAX_REVIEW_CONFIG_CANDIDATES]

    return _unique_config_candidates([
        requested_subset,
        *_GENERIC_CONFIG_CANDIDATES,
    ])[:_MAX_REVIEW_CONFIG_CANDIDATES]


def _ranked_split_candidates(available_splits: list[str], requested_split: Any) -> list[str]:
    metadata_splits = _unique_split_candidates(available_splits)
    if metadata_splits:
        requested = _normalise_split_name(requested_split)
        if requested in metadata_splits:
            return [requested, *[split for split in metadata_splits if split != requested]][:_MAX_REVIEW_SPLIT_CANDIDATES]
        return metadata_splits[:_MAX_REVIEW_SPLIT_CANDIDATES]

    return _unique_split_candidates([
        requested_split,
        *_GENERIC_SPLIT_CANDIDATES,
    ])[:_MAX_REVIEW_SPLIT_CANDIDATES]


def _load_hf_dataset(dataset_id: str, subset: str | None, split: str, *, streaming: bool):
    """加载 HF 数据集，兼容 subset=None 时省略 name 参数的情形。"""
    normalized_subset = _normalise_config_name(subset)
    if os.getenv("USE_HFD_DATASET_DOWNLOAD", "").strip().lower() in ("1", "true", "yes", "on"):
        return load_hf_dataset_with_fallback(
            dataset_id,
            normalized_subset,
            split,
            streaming=streaming,
            allow_hfd=_allow_hfd_for_review_load(),
        )
    load_dataset = getattr(_datasets_module(), "load_dataset")
    if normalized_subset is None:
        return load_dataset(dataset_id, split=split, streaming=streaming)
    return load_dataset(
        dataset_id,
        name=normalized_subset,
        split=split,
        streaming=streaming,
    )


def _resolved_ref(ref: dict, subset: str | None, split: str) -> dict:
    """生成"已解析"的 ref（包含确定下来的 subset 和 split）。"""
    resolved = dict(ref)
    resolved["subset"] = _normalise_config_name(subset)
    resolved["split"] = split
    return resolved


def _json_safe_sample_row(row: Any) -> dict[str, Any]:
    """将数据集的一行转为 JSON-safe 的字典（非基本类型截断为字符串）。"""
    sample: dict[str, Any] = {}
    if not isinstance(row, dict):
        try:
            row = dict(row)
        except Exception:
            return sample
    for key, value in row.items():
        name = str(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            sample[name] = value
        elif isinstance(value, (list, dict)):
            sample[name] = value
        else:
            sample[name] = str(value)[:500]
    return sample


def _take_json_safe_rows(iterator: Any, limit: int) -> tuple[list[dict[str, Any]], Any]:
    rows: list[dict[str, Any]] = []
    for _ in range(max(0, int(limit))):
        try:
            rows.append(_json_safe_sample_row(next(iterator)))
        except StopIteration:
            break
    return rows, iterator


def _review_metadata_from_first_row(
    first_row: dict[str, Any],
    split_names: list[str],
    raw_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """从数据集首行提取元数据（列名、schema 等）。"""
    schema = detect_schema_from_item(first_row) if first_row else {}
    return {
        "source_dataset_split_names": list(split_names),
        "source_dataset_columns": list(first_row.keys()),
        "source_dataset_first_row": first_row,
        "source_dataset_raw_rows": list(raw_rows or ([first_row] if first_row else [])),
        "source_dataset_schema": schema,
    }


def _raw_rows_to_review_samples(raw_rows: list[dict[str, Any]], split_names: list[str]) -> list[dict[str, Any]]:
    """Build lightweight prompt previews from raw rows without adapter gating."""
    samples: list[dict[str, Any]] = []
    for row in raw_rows[:_REVIEW_SAMPLE_SIZE]:
        if not isinstance(row, dict):
            continue
        schema = detect_schema_from_item(row)
        question_field = schema.get("question_field")
        answer_field = schema.get("answer_field")
        rollout_field = schema.get("rollout_gold_field") or answer_field
        train_field = schema.get("train_output_field") or answer_field
        sample: dict[str, Any] = {
            "source_dataset_split_names": list(split_names),
            "source_dataset_columns": list(row.keys()),
            "source_dataset_first_row": row,
            "source_dataset_schema": schema,
        }
        if question_field in row:
            sample["question_text"] = str(row.get(question_field) or "")
        if answer_field in row:
            sample["gold_answer"] = str(row.get(answer_field) or "")
        if rollout_field in row:
            sample["rollout_gold_answer"] = str(row.get(rollout_field) or "")
        if train_field in row:
            sample["train_output"] = str(row.get(train_field) or "")
        if schema.get("target_style"):
            sample["target_style"] = schema.get("target_style")
        samples.append(sample)
    return samples


def _attach_review_metadata(ref: dict, metadata: dict[str, Any]) -> dict:
    """将审查元数据（列名、首行、schema）附加到 ref。"""
    if not metadata:
        return ref
    enriched = dict(ref)
    for key in (
        "source_dataset_split_names",
        "source_dataset_columns",
        "source_dataset_first_row",
        "source_dataset_raw_rows",
        "source_dataset_schema",
    ):
        value = metadata.get(key)
        if value not in (None, [], {}):
            enriched[key] = value
    return enriched


def _adapt_review_stream(
    dataset: Any,
    _dataset_id: str,
    _subset: str | None,
    _split: str,
    _requested_split: str,
    split_names: list[str],
) -> tuple[list[dict], dict]:
    """从数据集流中采样 raw rows 并提取审查元数据。"""
    iterator = iter(dataset)
    raw_rows, _iterator = _take_json_safe_rows(iterator, _REVIEW_SAMPLE_SIZE)
    if not raw_rows:
        raise StopIteration
    first_row = raw_rows[0]
    metadata = _review_metadata_from_first_row(first_row, split_names, raw_rows)
    samples = _raw_rows_to_review_samples(raw_rows, split_names)
    return samples, metadata


def _load_review_sample(ref: dict) -> tuple[list[dict], dict]:
    """加载审查样本。

    返回 (样本列表, 已解析的 ref)。样本为空表示加载失败。
    """
    dataset_id = ref.get("dataset_id", "unknown")
    requested_split = ref.get("split") or "train"
    attempts: list[dict[str, str]] = []
    best_raw_resolved_ref: dict | None = None

    def record_attempt(stage: str, subset: str | None, split: str, detail: str) -> None:
        attempts.append({
            "stage": stage,
            "subset": repr(_normalise_config_name(subset)),
            "split": str(split),
            "detail": detail[:240],
        })

    def try_load_from_candidates(
        candidates: list[tuple[str | None, list[str]]],
    ) -> tuple[list[dict], dict] | None:
        nonlocal best_raw_resolved_ref
        for subset, split_names in candidates:
            split_candidates = _ranked_split_candidates(split_names, requested_split)
            if split_names:
                split_candidates = [split for split in split_candidates if split in split_names]
            if not split_candidates:
                record_attempt("split", subset, str(requested_split), "no split candidates")
                continue
            effective_split_names = split_names or split_candidates
            for split in split_candidates:
                rows, rows_error = _datasets_server_rows_info(str(dataset_id), subset, split)
                if rows:
                    metadata = _review_metadata_from_first_row(rows[0], effective_split_names, rows)
                    items = _raw_rows_to_review_samples(rows, effective_split_names)
                    if items:
                        if attempts:
                            print(
                                f"[dataset_reviewer] {dataset_id}: review sample resolved via "
                                f"datasets-server rows after {len(attempts)} failed attempt(s); "
                                f"subset={subset!r} split={split!r}"
                            )
                        return items, _attach_review_metadata(_resolved_ref(ref, subset, split), metadata)
                    if best_raw_resolved_ref is None:
                        fallback_subset = ref.get("subset") if ref.get("subset") is not None else subset
                        best_raw_resolved_ref = _attach_review_metadata(
                            _resolved_ref(ref, fallback_subset, split),
                            metadata,
                        )
                    record_attempt("datasets_server_rows", subset, split, "raw rows produced no review preview samples")
                elif rows_error:
                    record_attempt("datasets_server_rows", subset, split, rows_error)
                try:
                    restore = patch_hub_timeout(_REVIEW_STREAMING_TIMEOUT)
                    try:
                        dataset = _load_hf_dataset(
                            str(dataset_id),
                            subset,
                            split,
                            streaming=True,
                        )
                        items, metadata = _adapt_review_stream(
                            dataset,
                            str(dataset_id),
                            subset,
                            split,
                            str(requested_split),
                            effective_split_names,
                        )
                    finally:
                        restore()
                except StopIteration:
                    record_attempt("sample", subset, split, "dataset split is empty")
                    continue
                except Exception as exc:
                    record_attempt("load", subset, split, _short_exception(exc))
                    continue
                if items:
                    if attempts:
                        print(
                            f"[dataset_reviewer] {dataset_id}: review sample resolved after "
                            f"{len(attempts)} failed attempt(s); subset={subset!r} split={split!r}"
                        )
                    return items, _attach_review_metadata(_resolved_ref(ref, subset, split), metadata)
                if best_raw_resolved_ref is None:
                    fallback_subset = ref.get("subset") if ref.get("subset") is not None else subset
                    best_raw_resolved_ref = _attach_review_metadata(_resolved_ref(ref, fallback_subset, split), metadata)
                record_attempt("sample", subset, split, "raw rows produced no review preview samples")
        return None

    discovered_candidates, discovery_attempts = discover_config_splits(
        str(dataset_id),
        ref.get("subset"),
        requested_split,
    )
    attempts.extend(discovery_attempts)

    result = try_load_from_candidates(discovered_candidates)
    if result is not None:
        return result

    generic_candidates = [
        (subset, [])
        for subset in _unique_config_candidates([ref.get("subset"), *_GENERIC_CONFIG_CANDIDATES])
    ]
    result = try_load_from_candidates(generic_candidates)
    if result is not None:
        return result
    if attempts:
        summary = "; ".join(
            f"{item['stage']} subset={item['subset']} split={item['split']} detail={item['detail']}"
            for item in attempts[:8]
        )
        if len(attempts) > 8:
            summary += f"; ... {len(attempts) - 8} more"
        print(
            f"[dataset_reviewer] {dataset_id}: no usable review samples after "
            f"{len(attempts)} attempt(s): {summary}"
        )
    return [], best_raw_resolved_ref or _resolved_ref(ref, ref.get("subset"), requested_split)


# ═══════════════════════════════════════════════════════════
#  审查状态管理（黑名单、失败记录、缓存）
# ═══════════════════════════════════════════════════════════

def _dataset_state_manager(state: EvoState | dict[str, Any]) -> DatasetStateManager | None:
    """从 state 中获取 DatasetStateManager 实例。"""
    cache_path = state.get("dataset_states_path", "")
    if not cache_path:
        return None
    return DatasetStateManager(Path(str(cache_path)))


def _review_blacklist_reason(ref: DatasetRef | dict, state: EvoState | dict[str, Any]) -> str | None:
    """查询数据集是否被加入审查黑名单。返回黑名单原因或 None。"""
    data = ref.model_dump(mode="json") if isinstance(ref, DatasetRef) else ref
    dataset_id = str(data.get("dataset_id", "") or "")
    if not dataset_id:
        return None
    mgr = _dataset_state_manager(state)
    if mgr is None:
        return None
    return mgr.review_blacklist_reason(dataset_id)


def _mark_review_failure(
    ref: DatasetRef | dict,
    state: EvoState | dict[str, Any],
    reason: str,
    *,
    stage: str = "",
) -> None:
    """标记一次审查失败（用于黑名单背压）。"""
    data = ref.model_dump(mode="json") if isinstance(ref, DatasetRef) else ref
    dataset_id = str(data.get("dataset_id", "") or "")
    if not dataset_id:
        return
    mgr = _dataset_state_manager(state)
    if mgr is None:
        return
    mgr.mark_review_failure(
        dataset_id,
        reason,
        backoff_seconds=DATASET_REVIEW_FAILURE_BACKOFF_SECONDS,
        failures_before_backoff=DATASET_REVIEW_FAILURES_BEFORE_BACKOFF,
        stage=stage,
        status="backoff" if _transient_review_reject_reason(reason) else "failed",
        ref_key="/".join(_ref_cache_key(data)),
    )


def _clear_review_failure(ref: DatasetRef | dict, state: EvoState | dict[str, Any]) -> None:
    """清除审查失败记录（审查成功后调用）。"""
    data = ref.model_dump(mode="json") if isinstance(ref, DatasetRef) else ref
    dataset_id = str(data.get("dataset_id", "") or "")
    if not dataset_id:
        return
    mgr = _dataset_state_manager(state)
    if mgr is None:
        return
    mgr.clear_review_failure(dataset_id)


def _ref_cache_key(ref: DatasetRef | dict) -> tuple[str, str, str]:
    """为 ref 生成缓存键：(dataset_id, normalized_subset, normalized_split)。"""
    data = ref.model_dump(mode="json") if isinstance(ref, DatasetRef) else ref
    return (
        str(data.get("dataset_id", "")),
        _normalise_config_name(data.get("subset")) or "",
        _normalise_split_name(data.get("split")) or "train",
    )


def _verdict_cache_key(verdict: dict) -> tuple[str, str, str]:
    """为判决生成缓存键。"""
    review_ref = verdict.get("review_ref")
    if not isinstance(review_ref, dict):
        return ("", "", "")
    return _ref_cache_key(review_ref)


def _review_ref_for_cache(requested_ref: dict) -> dict:
    """生成用于缓存的 review_ref（标准化 subset/split）。"""
    review_ref = dict(requested_ref)
    review_ref["subset"] = _normalise_config_name(requested_ref.get("subset"))
    review_ref["split"] = _normalise_split_name(requested_ref.get("split")) or "train"
    return review_ref


def _decorate_verdict(verdict: dict, requested_ref: dict, resolved_ref: dict) -> dict:
    """为判决附加 review_ref 和 resolved_ref 信息。"""
    decorated = dict(verdict)
    decorated["review_ref"] = _review_ref_for_cache(requested_ref)
    decorated["resolved_ref"] = dict(resolved_ref)
    decorated["subset"] = resolved_ref.get("subset")
    decorated["split"] = resolved_ref.get("split")
    return decorated


def _accepted_ref_from_verdict(ref: DatasetRef, verdict: dict) -> DatasetRef:
    """从判决中提取 accepted ref（优先使用 resolved_ref）。"""
    resolved_ref = verdict.get("resolved_ref")
    if isinstance(resolved_ref, dict):
        try:
            return DatasetRef.model_validate(resolved_ref)
        except Exception as exc:
            print(
                f"[dataset_reviewer] Failed to validate resolved ref for "
                f"{ref.dataset_id}: {type(exc).__name__}: {exc}"
            )
    return ref


def _ref_identity_key(ref: DatasetRef | dict) -> tuple[str, str, str]:
    """ref 的身份标识键（同 _ref_cache_key，用于去重）。"""
    return _ref_cache_key(ref)


def _unique_ref_dicts(refs: list[dict]) -> list[dict]:
    """对 ref 字典列表去重。"""
    unique: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_ref in refs:
        if not isinstance(raw_ref, dict):
            continue
        key = _ref_identity_key(raw_ref)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        unique.append(dict(raw_ref))
    return unique


# ═══════════════════════════════════════════════════════════
#  池（pool）和待处理列表的更新操作
# ═══════════════════════════════════════════════════════════

def _append_accepted_refs_to_pool(state: EvoState, accepted_refs: list[DatasetRef]) -> list[dict]:
    """将接受的 ref 追加到 dataset_pool 末尾。"""
    updated_pool: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for raw_ref in state.get("dataset_pool") or []:
        if not isinstance(raw_ref, dict):
            continue
        key = _ref_identity_key(raw_ref)
        if not key[0] or key in seen_keys:
            continue
        seen_keys.add(key)
        updated_pool.append(dict(raw_ref))

    for ref in accepted_refs:
        raw_ref = ref.model_dump(mode="json")
        key = _ref_identity_key(raw_ref)
        if not key[0] or key in seen_keys:
            continue
        seen_keys.add(key)
        updated_pool.append(raw_ref)

    return updated_pool


def _remove_reviewed_from_pending(pending_refs: list[dict], reviewed_refs: list[DatasetRef]) -> list[dict]:
    """从待处理列表中移除已审查的 ref。"""
    reviewed_keys = {_ref_identity_key(ref) for ref in reviewed_refs}
    remaining: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_ref in pending_refs:
        if not isinstance(raw_ref, dict):
            continue
        key = _ref_identity_key(raw_ref)
        if not key[0] or key in reviewed_keys or key in seen:
            continue
        seen.add(key)
        remaining.append(dict(raw_ref))
    return remaining


def _merge_pending_with_payload_refs(state: EvoState, payload_refs: list[dict]) -> list[dict]:
    """合并 state 中持久化的待处理列表与消息负载中的新 ref。

    若有异步任务进行中（drained_count > 0），已判决的 ref 从 payload 中过滤掉。
    """
    state_pending_refs = list(state.get("dataset_review_pending_refs") or [])
    if not payload_refs:
        return _unique_ref_dicts(state_pending_refs)

    # 异步任务已经消耗了部分结果后，持久化的 pending 列表是权威来源。
    # 过时的 SEARCH_RESULT 消息可能携带原始批次的全部 ref（包括已被超时/失败拒绝的），
    # 重放这些 ref 会导致飞轮卡住。
    drained_count = int(state.get("dataset_review_drained_count", 0) or 0)
    if state.get("dataset_review_job_id") and drained_count > 0:
        reviewed_keys = {
            _verdict_cache_key(verdict)
            for verdict in state.get("dataset_review_verdicts", [])
            if isinstance(verdict, dict)
        }
        payload_refs = [
            ref for ref in payload_refs
            if _ref_identity_key(ref) not in reviewed_keys
        ]

    return _unique_ref_dicts(state_pending_refs + payload_refs)


def _merge_verdicts(cached_verdicts: list[dict], new_verdicts: list[dict]) -> list[dict]:
    """合并缓存判决和新判决（新判决覆盖旧判决）。"""
    new_verdict_keys = {_verdict_cache_key(nv) for nv in new_verdicts if isinstance(nv, dict)}
    return [
        v for v in cached_verdicts
        if not isinstance(v, dict) or _verdict_cache_key(v) not in new_verdict_keys
    ] + new_verdicts


# ═══════════════════════════════════════════════════════════
#  异步审查任务管理
# ═══════════════════════════════════════════════════════════

def _review_job_state_snapshot(state: EvoState) -> dict[str, Any]:
    """为异步审查任务创建 state 的快照（列表元素深拷贝，避免共享引用）。"""
    snapshot = dict(state)
    for key in (
        "dataset_pool",
        "dataset_review_pending_refs",
        "dataset_review_verdicts",
        "candidate_dataset_refs",
        "previous_dataset_refs",
    ):
        value = snapshot.get(key)
        if isinstance(value, list):
            snapshot[key] = [dict(item) if isinstance(item, dict) else item for item in value]
    return snapshot


def _review_job_id(state: EvoState, refs: list[dict]) -> str:
    """为审查任务生成唯一 ID（基于 trace_id + round_id + refs 指纹）。"""
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    fingerprint = "|".join("/".join(_ref_identity_key(ref)) for ref in refs)
    return f"{trace_id}:{round_id}:{abs(hash(fingerprint))}"


def _review_dataset_process_entry(ref_payload: dict, state_payload: dict[str, Any], result_queue: Any) -> None:
    """子进程入口：执行单条 ref 的审查并将结果放入队列。"""
    restore_inference_env = _apply_review_worker_inference_env()
    try:
        ref = DatasetRef.model_validate(ref_payload)
        verdict = _review_dataset(ref, cast(EvoState, state_payload))
        result_queue.put({"ok": True, "verdict": verdict})
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        restore_inference_env()


def _current_bool_setting(env_key: str, module_attr: str, default: bool) -> bool:
    raw = os.environ.get(env_key)
    if raw is not None:
        return raw.lower() in ("1", "true", "yes")
    try:
        module = importlib.import_module("config.settings")
        return bool(getattr(module, module_attr))
    except Exception:
        return default


def _apply_review_worker_inference_env() -> Callable[[], None]:
    """设置审查工人子进程的推理环境。

    worker 负责 DatasetCard/HF sample 加载，也会调用 reviewer agent。
    保留 USE_LLM_AGENTS/USE_VLLM，让 decide_json 走现有 model_runner，
    多个 worker 可复用同一个 named Ray vLLM actor；继承外层的 vLLM
    内存错误处理策略，避免 actor OOM 后在同一张卡上再尝试 HF fallback。
    """
    use_llm_agents = _current_bool_setting("USE_LLM_AGENTS", "USE_LLM_AGENTS", True)
    use_vllm = _current_bool_setting("USE_VLLM", "USE_VLLM", True)
    disable_hf_fallback = _current_bool_setting(
        "DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR",
        "DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR",
        True,
    )
    ray_address = os.environ.get("RAY_ADDRESS", "").strip() or "local"
    env_updates = {
        "DATASET_REVIEW_WORKER_INFERENCE": "1",
        "USE_LLM_AGENTS": "1" if use_llm_agents else "0",
        "USE_VLLM": "1" if use_vllm else "0",
        "DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR": "1" if disable_hf_fallback else "0",
        "RAY_ADDRESS": ray_address,
    }
    previous_env = {key: os.environ.get(key) for key in env_updates}
    os.environ.update(env_updates)
    previous_disable_ray = os.environ.pop("MODEL_RUNNER_DISABLE_RAY_VLLM", None)

    previous_attrs: list[tuple[Any, str, Any]] = []

    def set_module_attr(module_name: str, attr: str, value: Any) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return
        previous_attrs.append((module, attr, getattr(module, attr, None)))
        setattr(module, attr, value)

    # 同步已导入模块中的开关，保持 reviewer agent 和 Ray vLLM 可用。
    set_module_attr("config.settings", "USE_LLM_AGENTS", use_llm_agents)
    set_module_attr("config.settings", "USE_VLLM", use_vllm)
    set_module_attr("config.settings", "DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR", disable_hf_fallback)
    set_module_attr("config.settings", "RAY_ADDRESS", ray_address)
    set_module_attr("src.tools.llm_decision", "USE_LLM_AGENTS", use_llm_agents)

    def restore() -> None:
        for module, attr, value in reversed(previous_attrs):
            setattr(module, attr, value)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if previous_disable_ray is None:
            os.environ.pop("MODEL_RUNNER_DISABLE_RAY_VLLM", None)
        else:
            os.environ["MODEL_RUNNER_DISABLE_RAY_VLLM"] = previous_disable_ray

    return restore


def _multiprocessing_context() -> Any:
    """获取多进程上下文（优先使用 spawn 模式）。"""
    try:
        return mp.get_context("spawn")
    except ValueError:
        return mp.get_context()


def _record_completed_review_health(ref: DatasetRef, state: EvoState, verdict: dict) -> None:
    """记录审查完成后的健康状态：成功则清除失败记录，否则标记失败。"""
    if verdict.get("verdict") == "accept":
        _clear_review_failure(ref, state)
        return
    reason = str(verdict.get("reason", "") or "")
    if "no usable review samples" in reason.lower():
        _mark_review_failure(ref, state, reason, stage="no_usable_review_samples")


def _is_transient_review_reject(verdict: dict) -> bool:
    """判断判决是否为暂时性拒绝（可复现，不应缓存）。"""
    if verdict.get("verdict") == "accept":
        return False
    if "transient" in verdict:
        return bool(verdict.get("transient"))
    if str(verdict.get("cache_policy") or "").strip().lower() == "backoff":
        return True
    reason = str(verdict.get("reason", "") or "").lower()
    transient_markers = (
        "no usable review samples",
        "review timed out",
        "review worker",
        "blacklisted",
        "without result",
        "aborted",
    )
    return any(marker in reason for marker in transient_markers)


def _should_use_cached_review_verdict(ref: DatasetRef, state: EvoState, verdict: dict) -> bool:
    """判断是否应使用缓存的判决（非暂时性拒绝，或在黑名单中）。"""
    if not _is_transient_review_reject(verdict):
        return True
    return _review_blacklist_reason(ref, state) is not None


def _review_dataset_with_timeout(ref: DatasetRef, state: EvoState) -> dict:
    """在子进程中执行带超时的数据集审查。

    将 _review_dataset 调用隔离到独立进程，超时后强制终止。
    """
    blacklist_reason = _review_blacklist_reason(ref, state)
    if blacklist_reason:
        return _reject_verdict_for_ref(
            ref.model_dump(mode="json"),
            f"skipped blacklisted dataset: {blacklist_reason}",
            failure_stage="review_blacklist",
        )

    timeout_seconds = float(DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS or 0.0)
    if timeout_seconds <= 0:
        verdict = _review_dataset(ref, state)
        _record_completed_review_health(ref, state, verdict)
        return verdict

    ctx = _multiprocessing_context()
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_review_dataset_process_entry,
        args=(ref.model_dump(mode="json"), dict(state), result_queue),
    )
    process.start()
    process.join(timeout_seconds)

    result = _REVIEW_QUEUE_EMPTY
    if process.is_alive():
        result = _read_review_result_queue(result_queue, timeout_seconds=0.25)
        if result is _REVIEW_QUEUE_EMPTY:
            reason = f"review timed out after {timeout_seconds:.0f}s"
            print(f"[dataset_reviewer] {ref.dataset_id}: {reason}; terminating worker")
            _terminate_review_process(
                process,
                float(DATASET_REVIEW_TIMEOUT_KILL_GRACE_SECONDS or 0.0),
            )
            _mark_review_failure(ref, state, reason, stage="review_worker_timeout")
            return _reject_verdict_for_ref(
                ref.model_dump(mode="json"),
                reason,
                failure_stage="review_worker_timeout",
            )
        _terminate_review_process(
            process,
            float(DATASET_REVIEW_TIMEOUT_KILL_GRACE_SECONDS or 0.0),
        )

    if result is _REVIEW_QUEUE_EMPTY:
        result = _read_review_result_queue(result_queue, timeout_seconds=1.0)
    if result is _REVIEW_QUEUE_EMPTY:
        reason = f"review worker exited without result exitcode={process.exitcode}"
        _mark_review_failure(ref, state, reason, stage="review_worker_empty_result")
        return _reject_verdict_for_ref(
            ref.model_dump(mode="json"),
            reason,
            failure_stage="review_worker_empty_result",
        )

    if not isinstance(result, dict) or not result.get("ok"):
        reason = str(result.get("error", "review worker error") if isinstance(result, dict) else "review worker error")
        _mark_review_failure(ref, state, reason, stage="review_worker_error")
        return _reject_verdict_for_ref(
            ref.model_dump(mode="json"),
            reason,
            failure_stage="review_worker_error",
        )

    verdict = result.get("verdict")
    if not isinstance(verdict, dict):
        reason = "review worker returned invalid verdict"
        _mark_review_failure(ref, state, reason, stage="review_worker_invalid_verdict")
        return _reject_verdict_for_ref(
            ref.model_dump(mode="json"),
            reason,
            failure_stage="review_worker_invalid_verdict",
        )
    _record_completed_review_health(ref, state, verdict)
    return verdict


# 队列空哨兵值
_REVIEW_QUEUE_EMPTY = object()


def _read_review_result_queue(result_queue: Any, timeout_seconds: float) -> Any:
    """从结果队列中读取一条消息，超时返回哨兵。"""
    get_result = getattr(result_queue, "get", None)
    if get_result is None:
        return _REVIEW_QUEUE_EMPTY
    try:
        return get_result(timeout=max(0.0, timeout_seconds))
    except queue.Empty:
        return _REVIEW_QUEUE_EMPTY
    except TypeError:
        try:
            return get_result()
        except queue.Empty:
            return _REVIEW_QUEUE_EMPTY


def _terminate_review_process(process: Any, grace_seconds: float) -> None:
    """终止审查子进程：先 terminate，优雅期过后 kill。"""
    if not process.is_alive():
        return
    process.terminate()
    process.join(max(0.0, grace_seconds))
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join()


def _review_job_runner(job: _ReviewJob) -> None:
    """审查工作者线程：并行处理 job 中的 ref。

    对每条 ref：
      1. 检查缓存判决（可重用则跳过审查）
      2. 调用 _review_dataset_with_timeout（子进程隔离）
      3. 将结果放入 job.results

    异常时以拒绝判决填充未完成的 ref。
    """
    def review_one(raw_ref: dict, cached_by_key: dict[tuple[str, str, str], dict], state: dict[str, Any]) -> dict:
        try:
            ref = DatasetRef.model_validate(raw_ref)
            cache_key = _ref_cache_key(ref)
            cached_verdict = cached_by_key.get(cache_key)
            if cached_verdict is not None and _should_use_cached_review_verdict(
                ref,
                cast(EvoState, state),
                cached_verdict,
            ):
                verdict = cached_verdict
            else:
                verdict = _review_dataset_with_timeout(ref, cast(EvoState, state))
            return {
                "review_ref": ref.model_dump(mode="json"),
                "verdict": verdict,
            }
        except Exception as exc:
            fallback_ref = raw_ref if isinstance(raw_ref, dict) else {}
            return {
                "review_ref": dict(fallback_ref),
                "verdict": _reject_verdict_for_ref(
                    fallback_ref,
                    f"review worker error: {type(exc).__name__}: {exc}",
                    failure_stage="review_worker_error",
                ),
            }

    try:
        state = job.state
        cached_verdicts = [v for v in state.get("dataset_review_verdicts", []) if isinstance(v, dict)]
        cached_by_key: dict[tuple[str, str, str], dict] = {}
        for verdict in cached_verdicts:
            if not isinstance(verdict.get("resolved_ref"), dict):
                continue
            key = _verdict_cache_key(verdict)
            if key[0]:
                cached_by_key[key] = verdict
        max_workers = max(1, int(_DATASET_REVIEW_WORKER_COUNT or 1))
        next_index = 0
        running: dict[Future, dict] = {}

        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="dataset-review")
        try:
            while next_index < len(job.refs) and len(running) < max_workers:
                raw_ref = job.refs[next_index]
                running[executor.submit(review_one, raw_ref, cached_by_key, state)] = raw_ref
                next_index += 1

            while running:
                done, _pending = wait(running.keys(), return_when=FIRST_COMPLETED, timeout=0.1)
                if not done:
                    continue
                for future in done:
                    raw_ref = running.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        fallback_ref = raw_ref if isinstance(raw_ref, dict) else {}
                        result = {
                            "review_ref": dict(fallback_ref),
                            "verdict": _reject_verdict_for_ref(
                                fallback_ref,
                                f"review worker error: {type(exc).__name__}: {exc}",
                                failure_stage="review_worker_error",
                            ),
                        }
                    with job.condition:
                        job.results.append(result)
                        job.condition.notify_all()

                while next_index < len(job.refs) and len(running) < max_workers:
                    raw_ref = job.refs[next_index]
                    running[executor.submit(review_one, raw_ref, cached_by_key, state)] = raw_ref
                    next_index += 1
        finally:
            executor.shutdown(wait=True)
    except Exception as exc:
        # 顶层异常：拒绝所有尚未完成的 ref
        with job.condition:
            job.error = f"{type(exc).__name__}: {exc}"
            completed_keys: set[tuple[str, str, str]] = set()
            for result in job.results:
                if not isinstance(result, dict):
                    continue
                review_ref = result.get("review_ref")
                if isinstance(review_ref, dict):
                    completed_keys.add(_ref_identity_key(review_ref))
            for raw_ref in job.refs:
                key = _ref_identity_key(raw_ref)
                if key in completed_keys:
                    continue
                job.results.append({
                    "review_ref": dict(raw_ref),
                    "verdict": _reject_verdict_for_ref(
                        raw_ref,
                        f"review job aborted: {job.error}",
                    ),
                })
            job.completed = True
            job.condition.notify_all()
        return
    with job.condition:
        job.completed = True
        job.condition.notify_all()


def _get_or_start_review_job(state: EvoState, pending_refs: list[dict]) -> _ReviewJob | None:
    """获取或启动一个异步审查任务。

    如果已有匹配的 job（pending refs 是子集）则复用，否则新建线程。
    """
    if not pending_refs:
        return None
    existing_job_id = str(state.get("dataset_review_job_id") or "")
    with _REVIEW_JOBS_LOCK:
        job = _REVIEW_JOBS.get(existing_job_id) if existing_job_id else None
        if job is not None:
            pending_keys = {_ref_identity_key(ref) for ref in pending_refs}
            job_keys = {_ref_identity_key(ref) for ref in job.refs}
            if pending_keys.issubset(job_keys):
                return job
            _REVIEW_JOBS.pop(existing_job_id, None)
        if existing_job_id:
            # 过期的 state 可能在 round 转换后遗留旧的 job_id，不要附加新 ref
            existing_job_id = ""
        job_id = _review_job_id(state, pending_refs)
        job = _REVIEW_JOBS.get(job_id)
        if job is not None:
            return job
        job = _ReviewJob(job_id, pending_refs, _review_job_state_snapshot(state))
        _REVIEW_JOBS[job_id] = job
        thread = threading.Thread(
            target=_review_job_runner,
            args=(job,),
            name=f"dataset-reviewer-{job_id[-12:]}",
            daemon=True,
        )
        job.thread = thread
        thread.start()
        print(f"[dataset_reviewer] Started async review job refs={len(pending_refs)} job_id={job_id}")
        return job


def _cleanup_review_job(job: _ReviewJob) -> None:
    """清理已完成的审查任务（从全局注册表中移除）。"""
    with _REVIEW_JOBS_LOCK:
        _REVIEW_JOBS.pop(job.job_id, None)
    thread = job.thread
    if thread is not None and thread is not threading.current_thread() and not thread.is_alive():
        thread.join(timeout=0.1)


# ═══════════════════════════════════════════════════════════
#  结果提取与等待
# ═══════════════════════════════════════════════════════════

def _review_job_progress_snapshot(job: _ReviewJob, drained_count: int) -> tuple[bool, bool]:
    """获取审查任务的进度快照：(是否完成, 是否有未消费的结果)。"""
    with job.condition:
        completed = job.completed
        has_undrained_results = len(job.results) > drained_count
    return completed, has_undrained_results


def _ready_results_contain_accept(job: _ReviewJob, drained_count: int) -> bool:
    """检查已准备好的结果中是否包含 accept 判决。"""
    for result in job.results[drained_count:]:
        if not isinstance(result, dict):
            continue
        verdict = result.get("verdict")
        if isinstance(verdict, dict) and verdict.get("verdict") == "accept":
            return True
    return False


def _wait_for_review_accept_or_completion(job: _ReviewJob, drained_count: int, timeout_seconds: float) -> None:
    """等待审查任务产生第一个 accept 或完成（带超时）。"""
    if timeout_seconds <= 0:
        return
    deadline = time.time() + timeout_seconds
    with job.condition:
        while not job.completed and not _ready_results_contain_accept(job, drained_count):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            job.condition.wait(timeout=remaining)


def _build_waiting_response(state: EvoState, summary: str) -> RoutedMessage:
    """构建"等待中"的回环消息（自循环直到获得 accept）。"""
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    return RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.DATASET_REVIEWER,
            receiver=AgentName.DATASET_REVIEWER,
            message_type=MessageType.SEARCH_RESULT,
        ),
        payload=SearchResultPayload(
            datasets=[],
            search_summary=summary,
        ),
    )


def _drain_review_job_results(
    job: _ReviewJob,
    state: EvoState,
) -> tuple[list[DatasetRef], list[DatasetRef], list[str], list[dict], int, bool]:
    """从审查任务中提取已准备好的结果。

    返回：
        (accepted, reviewed_refs, rejected_ids, new_verdicts, new_drained_count, completed)
    """
    drained_count = int(state.get("dataset_review_drained_count", 0) or 0)
    with job.condition:
        ready_results = list(job.results[drained_count:])
        completed = job.completed

    accepted: list[DatasetRef] = []
    reviewed_refs: list[DatasetRef] = []
    rejected_ids: list[str] = []
    new_verdicts: list[dict] = []
    consumed = 0

    for result in ready_results:
        review_ref_raw = result.get("review_ref") if isinstance(result, dict) else {}
        verdict = result.get("verdict") if isinstance(result, dict) else {}
        if not isinstance(review_ref_raw, dict) or not isinstance(verdict, dict):
            consumed += 1
            continue
        try:
            review_ref = DatasetRef.model_validate(review_ref_raw)
        except Exception:
            consumed += 1
            continue
        reviewed_refs.append(review_ref)
        new_verdicts.append(verdict)
        if verdict.get("verdict") == "accept":
            accepted.append(_accepted_ref_from_verdict(review_ref, verdict))
            consumed += 1
        else:
            rejected_ids.append(review_ref.dataset_id)
            consumed += 1

    return accepted, reviewed_refs, rejected_ids, new_verdicts, drained_count + consumed, completed


# ═══════════════════════════════════════════════════════════
#  Reviewer agent 上下文
# ═══════════════════════════════════════════════════════════

def _format_samples_for_prompt(samples: list[dict]) -> str:
    """将样本格式化为 LLM prompt 可读的字符串。"""
    lines = []
    for i, q in enumerate(samples[:3], 1):
        text = str(q.get("question_text") or q.get("input", ""))[:200]
        answer = str(q.get("gold_answer") or q.get("output", ""))[:200]
        lines.append(f"[题目{i}] {text}\n[答案{i}] {answer}")
    return "\n\n".join(lines) if lines else "(no samples available)"


def _review_agent_context(
    ref: DatasetRef,
    state: EvoState,
    samples: list[dict],
    resolved_ref: dict,
    card_summary: dict[str, Any],
) -> dict[str, Any]:
    """组装 reviewer agent 所需的紧凑上下文。"""
    return {
        "goal": state.get("user_goal", ""),
        "dataset_id": ref.dataset_id,
        "source": ref.source,
        "subset": resolved_ref.get("subset"),
        "split": resolved_ref.get("split"),
        "sample_count": len(samples),
        "dataset_card": card_summary,
        "source_dataset_split_names": resolved_ref.get("source_dataset_split_names") or [],
        "source_dataset_columns": resolved_ref.get("source_dataset_columns") or [],
        "source_dataset_first_row": resolved_ref.get("source_dataset_first_row") or {},
        "source_dataset_raw_rows": resolved_ref.get("source_dataset_raw_rows") or [],
        "source_dataset_schema": resolved_ref.get("source_dataset_schema") or {},
        "samples": _format_samples_for_prompt(samples),
    }


def _schema_with_review_row_id(schema: dict[str, Any], decision: dict[str, Any], columns: list[Any]) -> dict[str, Any]:
    enriched = dict(schema)
    row_id_source = decision.get("row_id_source")
    column_names = {str(column) for column in columns if str(column)}
    if isinstance(row_id_source, str) and row_id_source in column_names:
        enriched["row_id_field"] = row_id_source
    return enriched


def _cleaner_request_for_review(
    ref: DatasetRef,
    state: EvoState,
    samples: list[dict],
    resolved_ref: dict,
    card_summary: dict[str, Any],
) -> CleanerCodegenRequest:
    trace_id = str(state.get("trace_id", "") or "")
    return CleanerCodegenRequest(
        dataset_id=ref.dataset_id,
        source=ref.source,
        subset=resolved_ref.get("subset"),
        split=str(resolved_ref.get("split") or ref.split or "train"),
        user_goal=str(state.get("user_goal", "") or ""),
        dataset_card=card_summary,
        source_dataset_columns=[str(item) for item in resolved_ref.get("source_dataset_columns", []) if str(item)],
        source_dataset_first_row=(
            resolved_ref.get("source_dataset_first_row")
            if isinstance(resolved_ref.get("source_dataset_first_row"), dict)
            else {}
        ),
        source_dataset_raw_rows=[
            row for row in resolved_ref.get("source_dataset_raw_rows", [])
            if isinstance(row, dict)
        ],
        normalized_samples=[sample for sample in samples if isinstance(sample, dict)],
        cache_root=get_session_dir(trace_id),
    )


def _attach_cleaner_result(resolved_ref: dict, result: Any) -> dict:
    enriched = dict(resolved_ref)
    cache_ref = getattr(result, "cleaner_cache_ref", {}) or {}
    if isinstance(cache_ref, dict) and cache_ref:
        enriched["cleaner_cache_ref"] = dict(cache_ref)
    schema = dict(enriched.get("source_dataset_schema") or {})
    schema_override = getattr(result, "schema_override", {}) or {}
    if isinstance(schema_override, dict):
        schema.update(schema_override)
    if isinstance(cache_ref, dict) and cache_ref:
        schema["cleaner_cache_ref"] = dict(cache_ref)
    enriched["source_dataset_schema"] = schema
    return enriched


def _has_ready_cleaner_ref(ref: dict) -> bool:
    cache_ref = ref.get("cleaner_cache_ref")
    return isinstance(cache_ref, dict) and cache_ref.get("status") == "ready"


# ═══════════════════════════════════════════════════════════
#  主入口：dataset_reviewer_node（LangGraph 节点函数）
# ═══════════════════════════════════════════════════════════

def dataset_reviewer_node(state: EvoState) -> dict:
    """数据集审查节点的 LangGraph 入口。

    输入：
        - pending_message: SEARCH_RESULT 消息（携带候选数据集 refs）
        - dataset_review_pending_refs: 之前未处理完的 refs

    逻辑：
        1. 合并消息中的 refs 与持久化的 pending_refs
        2. 获取/启动异步审查任务
        3. 等待第一个 accept（或超时）
        4. 提取结果，更新 state
        5. 若已得到 accept 则转发到下一节点，
           否则自循环等待（_build_waiting_response）
    """
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    payload_refs: list[dict] = []
    if pending_message.header.message_type == MessageType.SEARCH_RESULT:
        search_result = SearchResultPayload.model_validate(pending_message.payload)
        payload_refs = [ref.model_dump(mode="json") for ref in search_result.datasets]
    pending_refs = _merge_pending_with_payload_refs(state, payload_refs)
    if not pending_refs:
        return _build_empty_response(state)

    job = _get_or_start_review_job(state, pending_refs)
    if job is None:
        return _build_empty_response(state)

    drained_before = int(state.get("dataset_review_drained_count", 0) or 0)
    wait_seconds = (
        _REPLENISHMENT_ACCEPT_WAIT_SECONDS
        if state.get("data_replenishment_needed")
        else _FIRST_ACCEPT_WAIT_SECONDS
    )
    _wait_for_review_accept_or_completion(job, drained_before, wait_seconds)

    accepted, reviewed_refs, rejected, new_verdicts, drained_count, job_completed = _drain_review_job_results(
        job,
        state,
    )
    all_verdicts = _merge_verdicts(
        [v for v in state.get("dataset_review_verdicts", []) if isinstance(v, dict)],
        new_verdicts,
    )
    reviewed_ids = set(state.get("dataset_reviewed_ids", []))
    reviewed_ids.update(ref.dataset_id for ref in reviewed_refs)
    remaining_pending_refs = _remove_reviewed_from_pending(pending_refs, reviewed_refs)

    if accepted and remaining_pending_refs:
        print(
            "[dataset_reviewer] Accepted dataset ready; keeping async review queue active "
            f"pending={len(remaining_pending_refs)} job_id={job.job_id}"
        )

    if job.error:
        print(f"[dataset_reviewer] Async review job error: {job.error}")

    if accepted:
        has_undrained_results = False
    else:
        job_completed, has_undrained_results = _review_job_progress_snapshot(job, drained_count)

    if job_completed and remaining_pending_refs and not has_undrained_results:
        print(
            "[dataset_reviewer] Review job completed with undrained refs; "
            f"marking as rejected count={len(remaining_pending_refs)}"
        )
        stuck_reviewed: list[DatasetRef] = []
        stuck_verdicts: list[dict] = []
        invalid_stuck_count = 0
        for raw_ref in remaining_pending_refs:
            try:
                stuck_ref = DatasetRef.model_validate(raw_ref)
            except Exception as exc:
                invalid_stuck_count += 1
                print(
                    "[dataset_reviewer] Failed to dead-letter invalid pending ref: "
                    f"{type(exc).__name__}: {exc} ref={raw_ref}"
                )
                continue
            stuck_reviewed.append(stuck_ref)
            stuck_verdicts.append(
                _reject_verdict_for_ref(
                    stuck_ref.model_dump(mode="json"),
                    "review job completed without result",
                    failure_stage="review_job_undrained",
                )
            )
        if stuck_verdicts:
            reviewed_refs.extend(stuck_reviewed)
            reviewed_ids.update(ref.dataset_id for ref in stuck_reviewed)
            new_verdicts.extend(stuck_verdicts)
            all_verdicts = _merge_verdicts(
                [v for v in state.get("dataset_review_verdicts", []) if isinstance(v, dict)],
                new_verdicts,
            )
        if invalid_stuck_count:
            print(
                "[dataset_reviewer] Dropped invalid pending refs during dead-letter: "
                f"count={invalid_stuck_count}"
            )
        remaining_pending_refs = []

    if not accepted and job_completed and not remaining_pending_refs:
        print(
            f"[dataset_reviewer] All {len(pending_refs)} pending datasets reviewed; "
            f"no ready accepted ref in this drain, rejected={rejected}"
        )

    if not accepted and (not job_completed or remaining_pending_refs):
        review_active = (not job_completed) or bool(remaining_pending_refs)
        summary = (
            f"async reviewer waiting for first accepted ref; "
            f"ready_reviewed={len(reviewed_refs)} rejected={len(rejected)} "
            f"pending={len(remaining_pending_refs)} completed={job_completed}"
        )
        return {
            "dataset_reviewed_ids": list(reviewed_ids),
            "dataset_review_verdicts": all_verdicts,
            "dataset_review_pending_refs": remaining_pending_refs,
            "dataset_review_job_id": job.job_id,
            "dataset_review_active": review_active,
            "dataset_review_completed": False,
            "dataset_review_drained_count": drained_count,
            "dataset_pool": _append_accepted_refs_to_pool(state, []),
            "pending_message": _build_waiting_response(state, summary),
        }

    review_summary = (
        f"async reviewed_ready={len(reviewed_refs)} accepted={len(accepted)} "
        f"rejected={len(rejected)} pending={len(remaining_pending_refs)} "
        f"completed={job_completed}"
    )
    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.DATASET_REVIEWER,
            receiver=AgentName.SCREENING_ENTRY,
            message_type=MessageType.SEARCH_RESULT,
        ),
        payload=SearchResultPayload(
            datasets=accepted,
            search_summary=review_summary,
        ),
    )

    review_active = (not job_completed) or bool(remaining_pending_refs)
    update = {
        "dataset_reviewed_ids": list(reviewed_ids),
        "dataset_review_verdicts": all_verdicts,
        "dataset_review_pending_refs": remaining_pending_refs,
        "dataset_review_job_id": "" if job_completed and not remaining_pending_refs else job.job_id,
        "dataset_review_active": review_active,
        "dataset_review_completed": job_completed and not remaining_pending_refs,
        "dataset_review_drained_count": drained_count,
        "dataset_pool": _append_accepted_refs_to_pool(state, accepted),
        "pending_message": msg,
    }
    if update["dataset_review_completed"]:
        _cleanup_review_job(job)
    return update


# ═══════════════════════════════════════════════════════════
#  单条数据集的审查逻辑（Dataset Card + sample → agent）
# ═══════════════════════════════════════════════════════════

def _review_dataset(ref: DatasetRef, state: EvoState) -> dict:
    """审查一条数据集引用。

    Dataset Card 是主要语义输入；真实样本用于验证可加载性和补全 schema。
    """
    requested_ref = ref.model_dump(mode="json")
    samples, resolved_ref = _load_review_sample(requested_ref)
    resolved_dataset_ref = DatasetRef.model_validate(resolved_ref)
    raw_rows = [
        row for row in resolved_ref.get("source_dataset_raw_rows", [])
        if isinstance(row, dict)
    ]
    formal_proof_only_reason = _formal_proof_only_reject_reason(resolved_ref)
    if formal_proof_only_reason:
        verdict = {
            "dataset_id": resolved_dataset_ref.dataset_id,
            "verdict": "reject",
            "reason": formal_proof_only_reason,
            "suitability_score": 0.0,
            "sample_count": len(samples),
        }
        print(
            f"[dataset_reviewer] {ref.dataset_id}: "
            f"verdict={verdict['verdict']} score={verdict['suitability_score']} "
            f"reason={verdict['reason']} samples={verdict['sample_count']}"
        )
        return _decorate_verdict(verdict, requested_ref, resolved_ref)

    polluted_single_text_reason = _polluted_single_text_reject_reason(resolved_ref)
    if polluted_single_text_reason:
        verdict = {
            "dataset_id": resolved_dataset_ref.dataset_id,
            "verdict": "reject",
            "reason": polluted_single_text_reason,
            "suitability_score": 0.0,
            "sample_count": len(samples),
        }
        print(
            f"[dataset_reviewer] {ref.dataset_id}: "
            f"verdict={verdict['verdict']} score={verdict['suitability_score']} "
            f"reason={verdict['reason']} samples={verdict['sample_count']}"
        )
        return _decorate_verdict(verdict, requested_ref, resolved_ref)

    if not samples and not raw_rows:
        verdict = {
            "dataset_id": resolved_dataset_ref.dataset_id,
            "verdict": "reject",
            "reason": "no usable review samples",
            "suitability_score": 0.0,
            "sample_count": 0,
        }
        print(
            f"[dataset_reviewer] {ref.dataset_id}: "
            f"verdict={verdict['verdict']} score={verdict['suitability_score']} "
            f"reason={verdict['reason']} samples={verdict['sample_count']}"
        )
        return _decorate_verdict(verdict, requested_ref, resolved_ref)

    card_summary = load_dataset_card_summary(
        ref.dataset_id,
        text_char_limit=_DATASET_CARD_TEXT_CHAR_LIMIT,
        timeout=_HF_METADATA_NETWORK_TIMEOUT,
    )

    decision = decide_json(
        agent_name="dataset_reviewer",
        prompt=prompt_for_agent(dict(state), "dataset_reviewer", DATASET_REVIEWER_PROMPT),
        context=_review_agent_context(ref, state, samples, resolved_ref, card_summary),
        fallback={
            "verdict": "reject",
            "reason": "fallback: LLM decision could not be parsed",
            "suitability_score": 0.0,
        },
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
        max_new_tokens=512,
        disable_thinking=True,
        stop_after_json=False,
    )

    print(
        f"[dataset_reviewer] {ref.dataset_id}: "
        f"verdict={decision.get('verdict', '?')} "
        f"score={decision.get('suitability_score', '?')} "
        f"reason={decision.get('reason', '?')}"
    )
    if decision.get("verdict", "reject") == "accept" and isinstance(resolved_ref.get("source_dataset_schema"), dict):
        resolved_ref["source_dataset_schema"] = _schema_with_review_row_id(
            resolved_ref["source_dataset_schema"],
            decision,
            resolved_ref.get("source_dataset_columns") or [],
        )

    verdict_value = decision.get("verdict", "reject")
    if verdict_value == "accept" and not _has_ready_cleaner_ref(resolved_ref):
        provider = build_cleaner_provider()
        if provider is not None and raw_rows:
            cleaner_result = ensure_cleaner_for_ref(
                _cleaner_request_for_review(ref, state, samples, resolved_ref, card_summary),
                provider=provider,
                max_repair_attempts=DATA_CLEANER_MAX_REPAIR_ATTEMPTS,
            )
            if getattr(cleaner_result, "status", "") == "ready":
                resolved_ref = _attach_cleaner_result(resolved_ref, cleaner_result)
            else:
                verdict_value = "reject"
                decision["reason"] = (
                    "cleaner validation failed: "
                    + str(getattr(cleaner_result, "failure_reason", "") or "unknown")
                )
                decision["suitability_score"] = 0.0
        else:
            verdict_value = "reject"
            decision["reason"] = "cleaner provider unavailable for dataset"
            decision["suitability_score"] = 0.0

    return _decorate_verdict({
        "dataset_id": ref.dataset_id,
        "verdict": verdict_value,
        "reason": decision.get("reason", ""),
        "suitability_score": decision.get("suitability_score", 0.5),
        "sample_count": len(samples),
    }, requested_ref, resolved_ref)


def _build_empty_response(state: EvoState) -> dict:
    """当无可审查的数据集时返回空响应。"""
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.DATASET_REVIEWER,
            receiver=AgentName.SCREENING_ENTRY,
            message_type=MessageType.SEARCH_RESULT,
        ),
        payload=SearchResultPayload(
            datasets=[],
            search_summary="no datasets to review",
        ),
    )
    return {"pending_message": msg}
