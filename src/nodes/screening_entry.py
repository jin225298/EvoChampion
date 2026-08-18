import os
import time
from collections import Counter

from src.models.messages import (
    AgentName,
    DatasetRef,
    DatasetSchemaRequestPayload,
    DatasetSchemaResultPayload,
    FormatErrorPayload,
    MaterializedDatasetPayload,
    MessageHeader,
    MessageType,
    RoutedMessage,
    SearchRequestPayload,
    SearchResultPayload,
)
from src.models.state import EvoState
from config.settings import (
    DATA_WINDOW_SIZE,
    DATA_WINDOW_RETRY_LIMIT,
    MAX_PROFILE_ITEMS_PER_ROUND,
    MAX_PROFILE_WINDOWS_PER_ROUND,
    SCREENING_ENTRY_MAX_QUESTIONS,
    get_classifier_labels,
)
from src.tools.dataset_adapter import load_hf_dataset_with_fallback, normalize_item
from src.tools.dataset_bank import infer_module
from src.tools.dataset_cleaner_codegen import apply_cleaner_to_rows
from src.tools.dataset_state import DATASET_STATE_IN_USE
from src.tools.dataset_state import ITEM_STATE_UNUSED
from src.tools.dataset_state import DatasetStateManager
from src.tools.message_artifacts import write_json_artifact
from src.tools.question_fields import normalize_target_style
from src.tools.search_query import fallback_query_from_goal, normalize_content_search_query


def _window_bounds_from_ref(ref: dict, state: EvoState | dict | None = None) -> tuple[int, int]:
    """Return (offset, limit) for a dataset ref window.

    Ref shard fields win because HF search already emits shard_start/shard_size.
    State fields are fallback hooks for future shortfall loops.
    """
    state = state or {}
    ref_offset = ref.get("shard_start")
    ref_limit = ref.get("shard_size")
    active_dataset_id = str(state.get("active_dataset_id", "") or "")
    dataset_id = str(ref.get("dataset_id", "") or "")
    fallback_offset = state.get("current_window_offset", 0) if dataset_id == active_dataset_id else 0
    offset = int(ref_offset if ref_offset is not None else fallback_offset or 0)
    default_limit = int(DATA_WINDOW_SIZE or SCREENING_ENTRY_MAX_QUESTIONS or 500)
    limit = int(ref_limit if ref_limit is not None else state.get("current_window_size", default_limit) or default_limit)
    return max(0, offset), max(1, limit)


def _window_metadata(ref: dict, offset: int, limit: int, loaded_count: int) -> dict:
    dataset_id = str(ref.get("dataset_id", "unknown"))
    split = ref.get("split") or "train"
    subset = ref.get("subset")
    shard_id = ref.get("shard_id")
    if shard_id is not None:
        window_id = f"{dataset_id}:{split}:shard:{shard_id}:{offset}:{limit}"
    else:
        window_id = f"{dataset_id}:{split}:window:{offset}:{limit}"
    return {
        "dataset_id": dataset_id,
        "split": split,
        "subset": subset,
        "requested_split": ref.get("requested_split") or ref.get("source_dataset_requested_split"),
        "offset": offset,
        "limit": limit,
        "window_id": window_id,
        "loaded_count": loaded_count,
        "exhausted": loaded_count < limit,
    }


def _cleaner_cache_ref_from_ref(ref: dict) -> tuple[dict, str]:
    cache_ref = ref.get("cleaner_cache_ref")
    if isinstance(cache_ref, dict) and cache_ref:
        return dict(cache_ref), "cleaner_cache_ref"
    return {}, ""


_FALLBACK_SCHEMA_REASONS = {
    "fallback schema",
    "fallback flat schema",
    "fallback conversation schema",
}


def _schema_for_ready_cleaner(source_dataset_schema: dict) -> dict:
    reason = str(source_dataset_schema.get("reason") or "").strip().lower()
    if reason in _FALLBACK_SCHEMA_REASONS:
        return {}
    return dict(source_dataset_schema)


def _cleaner_failure_metadata(
    ref: dict,
    offset: int,
    limit: int,
    *,
    reason: str,
    cleaner_status: str | None = None,
    cleaner_source: str = "",
) -> dict:
    metadata = _window_metadata(ref, offset, limit, 0)
    metadata["failure_reason"] = reason
    metadata["cleaner_required"] = True
    if cleaner_status is not None:
        metadata["cleaner_status"] = cleaner_status
    if cleaner_source:
        metadata["cleaner_source"] = cleaner_source
    return metadata


def _load_local_dataset_rows(
    ref: dict,
    offset: int,
    limit: int,
    state: EvoState | dict | None,
) -> tuple[list[dict], dict]:
    """Load a LOCAL dataset directory (code-domain smoke data) without a cleaner.

    Local datasets ship question/answer/test/entry_point already in the right
    shape, so the codegen cleaner (which assumes math rows and drops the test
    fields) is bypassed. Rows are normalized directly and test/entry_point are
    preserved so the test-execution judge can drive candidates.
    """
    dataset_id = ref.get("dataset_id", "unknown")
    subset = ref.get("subset")
    split = ref.get("split") or "train"
    requested_split = ref.get("requested_split") or ref.get("source_dataset_requested_split")
    schema = {
        "usable": True,
        "question_field": "question",
        "answer_field": "answer",
        "rollout_gold_field": "answer",
        "train_output_field": "answer",
        "target_style": "answer",
        "dedup_key_field": "question",
        "schema_type": "flat",
    }
    columns = ["question", "answer", "test", "entry_point"]
    try:
        dataset = load_hf_dataset_with_fallback(dataset_id, subset, split)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[screening_entry] Failed to load local dataset {dataset_id}: {error}")
        metadata = _window_metadata(ref, offset, limit, 0)
        metadata["load_error"] = error
        metadata["failure_reason"] = error
        return [], metadata

    rows = list(dataset)
    window = rows[offset:offset + limit] if limit else rows[offset:]
    questions: list[dict] = []
    round_id = int((state or {}).get("round_id", 0) or 0)
    for i, row in enumerate(window):
        idx = offset + i
        norm = normalize_item(
            row,
            schema,
            idx,
            dataset_id,
            source_dataset_split=str(split) if split is not None else None,
            source_dataset_subset=str(subset) if subset is not None else None,
            source_dataset_requested_split=str(requested_split) if requested_split is not None else None,
            source_dataset_split_names=[str(split)] if split else [],
            source_dataset_columns=columns,
            source_dataset_first_row=rows[0] if rows else {},
            source_dataset_schema=schema,
        )
        if norm is None:
            continue
        test_code = str(row.get("test") or row.get("tests") or row.get("test_code") or "").strip()
        entry_point = str(row.get("entry_point") or row.get("function_name") or "").strip()
        norm["test"] = test_code
        norm["entry_point"] = entry_point
        norm["evaluation_method"] = "code_execution"
        norm["needs_judge"] = False
        norm["added_round"] = round_id
        norm["module"] = "code"
        norm["dataset_window_id"] = ref.get("dataset_window_id")
        norm["dataset_window_offset"] = offset
        norm["dataset_window_limit"] = limit
        questions.append(norm)

    metadata = _window_metadata(ref, offset, limit, len(questions))
    metadata["local_dataset"] = True
    if not questions:
        metadata["failure_reason"] = "no_questions_loaded"
    print(
        f"[screening_entry] Loaded local dataset {offset}:{offset + len(questions)} "
        f"({len(questions)} questions) from {dataset_id} (bypassed cleaner)"
    )
    return questions, metadata


def _load_dataset_from_ref(ref: dict, schema_override: dict | None = None, state: EvoState | dict | None = None) -> tuple[list[dict], dict]:
    """Download and clean a single dataset reference with a validated cleaner."""
    dataset_id = ref.get("dataset_id", "unknown")
    subset = ref.get("subset")
    split = ref.get("split") or "train"
    offset, limit = _window_bounds_from_ref(ref, state)
    # Local dataset directories (code-domain smoke data) bypass the cleaner:
    # they already carry question/answer/test/entry_point and the cleaner would
    # drop the test fields.
    if isinstance(dataset_id, str) and os.path.isdir(dataset_id):
        return _load_local_dataset_rows(ref, offset, limit, state)
    cleaner_cache_ref, cleaner_source = _cleaner_cache_ref_from_ref(ref)
    if not cleaner_cache_ref:
        return [], _cleaner_failure_metadata(
            ref,
            offset,
            limit,
            reason="cleaner_cache_ref_missing",
        )
    cleaner_status = str(cleaner_cache_ref.get("status") or "unknown")
    if cleaner_status != "ready":
        return [], _cleaner_failure_metadata(
            ref,
            offset,
            limit,
            reason=f"cleaner_cache_ref_not_ready:{cleaner_status}",
            cleaner_status=cleaner_status,
            cleaner_source=cleaner_source,
        )

    try:
        dataset = load_hf_dataset_with_fallback(
            dataset_id,
            subset,
            split,
            streaming=True,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[screening_entry] Failed to load {dataset_id}: {error}")
        metadata = _window_metadata(ref, offset, limit, 0)
        metadata["load_error"] = error
        metadata["failure_reason"] = error
        metadata["cleaner_required"] = True
        metadata["cleaner_status"] = cleaner_status
        metadata["cleaner_source"] = cleaner_source
        return [], metadata

    t_start = time.time()
    requested_split = ref.get("requested_split") or ref.get("source_dataset_requested_split")
    source_dataset_schema = (
        dict(ref.get("source_dataset_schema"))
        if isinstance(ref.get("source_dataset_schema"), dict)
        else {}
    )
    if isinstance(schema_override, dict):
        source_dataset_schema.update(
            {key: value for key, value in schema_override.items() if key != "cleaner_cache_ref"}
        )
    source_dataset_schema = _schema_for_ready_cleaner(source_dataset_schema)
    source_dataset_first_row = (
        dict(ref.get("source_dataset_first_row"))
        if isinstance(ref.get("source_dataset_first_row"), dict)
        else {}
    )
    try:
        raw, cleaner_stats = apply_cleaner_to_rows(
            dataset,
            cleaner_cache_ref,
            dataset_id=str(dataset_id),
            source_dataset_split=str(split) if split is not None else None,
            source_dataset_subset=str(subset) if subset is not None else None,
            source_dataset_requested_split=str(requested_split) if requested_split is not None else None,
            source_dataset_split_names=[str(item) for item in ref.get("source_dataset_split_names", []) if str(item)],
            source_dataset_columns=[str(item) for item in ref.get("source_dataset_columns", []) if str(item)],
            source_dataset_first_row=source_dataset_first_row,
            source_dataset_schema=source_dataset_schema,
            max_items=limit,
            offset=offset,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[screening_entry] Failed to clean {dataset_id}: {error}")
        metadata = _window_metadata(ref, offset, limit, 0)
        metadata["load_error"] = error
        metadata["failure_reason"] = error
        metadata["cleaner_required"] = True
        metadata["cleaner_status"] = cleaner_status
        metadata["cleaner_source"] = cleaner_source
        return [], metadata

    questions: list[dict] = []
    for q in raw:
        questions.append({
            **q,
            "added_round": int((state or {}).get("round_id", 0) or 0),
            "module": infer_module(q["question_text"]),
        })

    elapsed = time.time() - t_start
    print(
        f"[screening_entry] Loaded window {offset}:{offset + limit} "
        f"({len(questions)} questions) from {dataset_id} "
        f"in {elapsed:.2f}s"
    )
    metadata = _window_metadata(ref, offset, limit, len(questions))
    metadata["cleaner_required"] = True
    metadata["cleaner_status"] = cleaner_status
    metadata["cleaner_source"] = cleaner_source
    metadata["cleaner_stats"] = dict(cleaner_stats or {})
    for key in ("cleaned_count", "rejected_count", "processed_count"):
        if isinstance(cleaner_stats, dict) and key in cleaner_stats:
            metadata[key] = cleaner_stats[key]
    if not questions:
        metadata["failure_reason"] = "no_questions_loaded"
    for q in questions:
        q["dataset_window_id"] = metadata["window_id"]
        q["dataset_window_offset"] = offset
        q["dataset_window_limit"] = limit
    return questions, metadata


def _dataset_identity(ref: dict) -> tuple[str, str, str]:
    return (
        str(ref.get("dataset_id", "")),
        str(ref.get("subset") or ""),
        str(ref.get("split") or "train"),
    )


def _same_dataset(left: dict, right: dict) -> bool:
    return _dataset_identity(left) == _dataset_identity(right)


def _dataset_row_count(state: EvoState | dict, ref: dict) -> int | None:
    schema_candidates = []
    schema_info = state.get("dataset_schema_info") or {}
    if isinstance(schema_info, dict):
        schema_candidates.append(schema_info)
    for info in schema_candidates:
        schema_ref = info.get("dataset_ref")
        same_ref = isinstance(schema_ref, dict) and _same_dataset(schema_ref, ref)
        same_id = str(info.get("dataset_id", "") or "") == str(ref.get("dataset_id", ""))
        if not (same_ref or same_id):
            continue
        inspect_result = info.get("inspect_result")
        if not isinstance(inspect_result, dict):
            continue
        raw_rows = inspect_result.get("num_rows")
        if raw_rows is None:
            continue
        try:
            rows = int(raw_rows)
        except (TypeError, ValueError):
            continue
        # inspect_dataset streams at most 2000 rows; 2000 is a lower bound, not a safe total size.
        if 0 < rows < 2000:
            return rows
    return None


def _dataset_state_manager(state: EvoState | dict) -> DatasetStateManager | None:
    cache_path = state.get("dataset_states_path", "")
    if not cache_path:
        return None
    from pathlib import Path

    return DatasetStateManager(Path(str(cache_path)))


def _dataset_exhausted(mgr: DatasetStateManager | None, dataset_id: str) -> bool:
    return bool(mgr is not None and dataset_id and mgr.all_exhausted(dataset_id))


def _dataset_review_blacklisted(mgr: DatasetStateManager | None, dataset_id: str) -> bool:
    return bool(mgr is not None and dataset_id and mgr.is_review_blacklisted(dataset_id))


def _dataset_in_use(mgr: DatasetStateManager | None, dataset_id: str) -> bool:
    if mgr is None or not dataset_id:
        return False
    ds = mgr.datasets.get(dataset_id) or mgr.load_cached(dataset_id)
    return bool(ds is not None and ds.state == DATASET_STATE_IN_USE and mgr.count_unused(dataset_id) > 0)


def _dominant_target_style(questions: list[dict]) -> str | None:
    counts = Counter(
        style for style in (normalize_target_style(q.get("target_style")) for q in questions)
        if style is not None
    )
    return str(counts.most_common(1)[0][0]) if counts else None


def _dataset_target_style(state: EvoState | dict, dataset_id: str) -> str | None:
    if not dataset_id:
        return None
    cross_pool = state.get("cross_dataset_pool") or {}
    if isinstance(cross_pool, dict):
        pool = cross_pool.get(dataset_id)
        if isinstance(pool, list):
            style = _dominant_target_style([q for q in pool if isinstance(q, dict)])
            if style:
                return style
    schema_info = state.get("dataset_schema_info") or {}
    if isinstance(schema_info, dict):
        schema_dataset_id = str(schema_info.get("dataset_id", "") or "")
        schema_ref = schema_info.get("dataset_ref")
        ref_dataset_id = str(schema_ref.get("dataset_id", "") or "") if isinstance(schema_ref, dict) else ""
        if dataset_id in {schema_dataset_id, ref_dataset_id}:
            schema = schema_info.get("schema")
            if isinstance(schema, dict):
                return normalize_target_style(schema.get("target_style"))
    return None


def _desired_target_style(state: EvoState | dict) -> str | None:
    active_dataset_id = str(state.get("active_dataset_id", "") or "")
    active_style = _dataset_target_style(state, active_dataset_id)
    if active_style:
        return active_style
    round_stats = state.get("round_data_stats") or {}
    if isinstance(round_stats, dict):
        return normalize_target_style(round_stats.get("target_style"))
    return None


def _matches_target_style(state: EvoState | dict, dataset_id: str, desired_style: str | None) -> bool:
    if desired_style is None:
        return True
    dataset_style = _dataset_target_style(state, dataset_id)
    return dataset_style == desired_style


def _question_index_from_id(question_id: str) -> int | None:
    try:
        suffix = str(question_id).rsplit("_", 1)[1]
    except IndexError:
        return None
    if not suffix.isdigit():
        return None
    return int(suffix)


def _item_window_offset(item, window_size: int) -> int | None:
    explicit = getattr(item, "dataset_window_offset", None)
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    idx = _question_index_from_id(str(getattr(item, "question_id", "") or ""))
    if idx is None:
        return None
    size = max(1, int(window_size or 1))
    return (idx // size) * size


def _dataset_next_offset(mgr: DatasetStateManager | None, dataset_id: str, fallback: int = 0) -> int:
    if mgr is None or not dataset_id:
        return max(0, int(fallback or 0))
    ds = mgr.datasets.get(dataset_id) or mgr.load_cached(dataset_id)
    if ds is None:
        return max(0, int(fallback or 0))
    indices = [
        idx for idx in (_question_index_from_id(qid) for qid in ds.items)
        if idx is not None
    ]
    if indices:
        return max(max(indices) + 1, int(fallback or 0))
    return max(0, int(ds.total_loaded or fallback or 0))


def _loaded_questions_next_offset(questions: list[dict], fallback: int = 0) -> int:
    indices = [
        idx for idx in (_question_index_from_id(str(q.get("question_id", ""))) for q in questions)
        if idx is not None
    ]
    if indices:
        return max(indices) + 1
    return max(0, int(fallback or 0))


def _new_profile_item_count(questions: list[dict], cursor_before: int) -> int:
    parsed_indices: list[int] = []
    unknown_count = 0
    for question in questions:
        idx = _question_index_from_id(str(question.get("question_id", "")))
        if idx is None:
            unknown_count += 1
        else:
            parsed_indices.append(idx)
    if not parsed_indices:
        return len(questions)
    return sum(1 for idx in parsed_indices if idx >= cursor_before) + unknown_count


def _account_window_exploration(
    state: EvoState | dict,
    ref: dict,
    questions: list[dict],
    window_meta: dict,
) -> tuple[int, int, dict]:
    """Return per-round budget deltas for a loaded window.

    The per-round window budget counts only newly explored dataset territory.
    Re-loading an already registered window at the start of a later round is
    useful for mining remaining unused items, but it must not consume that
    later round's exploration budget.
    """
    dataset_id = str(ref.get("dataset_id") or window_meta.get("dataset_id") or "")
    if not questions or not dataset_id:
        return 0, 0, dict(window_meta)
    mgr = _dataset_state_manager(state)
    cursor_before = _dataset_next_offset(mgr, dataset_id, 0)
    offset = int(window_meta.get("offset", 0) or 0)
    fallback_after = offset + len(questions)
    cursor_after = _loaded_questions_next_offset(questions, fallback_after)
    new_exploration = cursor_after > cursor_before
    window_delta = 1 if new_exploration else 0
    item_delta = _new_profile_item_count(questions, cursor_before) if new_exploration else 0
    annotated_meta = {
        **window_meta,
        "new_exploration": new_exploration,
        "exploration_cursor_before": cursor_before,
        "exploration_cursor_after": max(cursor_before, cursor_after),
        "new_profile_items": item_delta,
    }
    return window_delta, item_delta, annotated_meta


def _dataset_reuse_offset(mgr: DatasetStateManager | None, dataset_id: str, window_size: int) -> int | None:
    if mgr is None or not dataset_id:
        return None
    ds = mgr.datasets.get(dataset_id) or mgr.load_cached(dataset_id)
    if ds is None or ds.state != DATASET_STATE_IN_USE:
        return None
    offsets = [
        offset
        for item in ds.items.values()
        if item.state == ITEM_STATE_UNUSED
        for offset in [_item_window_offset(item, window_size)]
        if offset is not None
    ]
    if offsets:
        return min(offsets)
    if mgr.count_unused(dataset_id) > 0:
        return 0
    return None


def _cached_replenishment_seen(state: EvoState | dict) -> dict[str, set[int]]:
    round_id = int(state.get("round_id", 0) or 0)
    raw_seen_round = state.get("cached_replenishment_windows_seen_round_id", -1)
    try:
        seen_round = int(raw_seen_round)
    except (TypeError, ValueError):
        seen_round = -1
    if seen_round != round_id:
        return {}
    raw_seen = state.get("cached_replenishment_windows_seen") or {}
    if not isinstance(raw_seen, dict):
        return {}
    seen: dict[str, set[int]] = {}
    for dataset_id, offsets in raw_seen.items():
        if not isinstance(offsets, list):
            continue
        parsed: set[int] = set()
        for offset in offsets:
            try:
                parsed.add(int(offset))
            except (TypeError, ValueError):
                continue
        if parsed:
            seen[str(dataset_id)] = parsed
    return seen


def _round_loaded_windows(state: EvoState | dict) -> dict[str, set[int]]:
    round_id = int(state.get("round_id", 0) or 0)
    raw_round = state.get("replenishment_loaded_windows_round_id", -1)
    try:
        loaded_round = int(raw_round)
    except (TypeError, ValueError):
        loaded_round = -1
    if loaded_round != round_id:
        return {}
    raw_loaded = state.get("replenishment_loaded_windows") or {}
    if not isinstance(raw_loaded, dict):
        return {}
    loaded: dict[str, set[int]] = {}
    for dataset_id, offsets in raw_loaded.items():
        if not isinstance(offsets, list):
            continue
        parsed: set[int] = set()
        for offset in offsets:
            try:
                parsed.add(int(offset))
            except (TypeError, ValueError):
                continue
        if parsed:
            loaded[str(dataset_id)] = parsed
    return loaded


def _round_attempted_windows(state: EvoState | dict) -> dict[str, set[int]]:
    round_id = int(state.get("round_id", 0) or 0)
    raw_round = state.get("replenishment_attempted_windows_round_id", -1)
    try:
        attempted_round = int(raw_round)
    except (TypeError, ValueError):
        attempted_round = -1
    if attempted_round != round_id:
        return {}
    raw_attempted = state.get("replenishment_attempted_windows") or {}
    if not isinstance(raw_attempted, dict):
        return {}
    attempted: dict[str, set[int]] = {}
    for dataset_id, offsets in raw_attempted.items():
        if not isinstance(offsets, list):
            continue
        parsed: set[int] = set()
        for offset in offsets:
            try:
                parsed.add(int(offset))
            except (TypeError, ValueError):
                continue
        if parsed:
            attempted[str(dataset_id)] = parsed
    return attempted


def _mark_seen_replenishment_window(
    seen: dict[str, set[int]],
    dataset_id: str,
    offset: int | None,
) -> None:
    if not dataset_id or offset is None:
        return
    seen.setdefault(str(dataset_id), set()).add(max(0, int(offset or 0)))


def _seen_with_current_window(state: EvoState | dict) -> dict[str, set[int]]:
    seen = _cached_replenishment_seen(state)
    for dataset_id, offsets in _round_loaded_windows(state).items():
        seen.setdefault(dataset_id, set()).update(offsets)
    for dataset_id, offsets in _round_attempted_windows(state).items():
        seen.setdefault(dataset_id, set()).update(offsets)
    current_window = (state.get("round_data_stats") or {}).get("window", {})
    if isinstance(current_window, dict):
        dataset_id = str(current_window.get("dataset_id") or state.get("active_dataset_id") or "")
        try:
            offset = int(current_window.get("offset", state.get("current_window_offset", 0)) or 0)
        except (TypeError, ValueError):
            offset = None
        _mark_seen_replenishment_window(seen, dataset_id, offset)
    else:
        dataset_id = str(state.get("active_dataset_id") or "")
        try:
            offset = int(state.get("current_window_offset", 0) or 0)
        except (TypeError, ValueError):
            offset = None
        _mark_seen_replenishment_window(seen, dataset_id, offset)
    current_round = int(state.get("round_id", 0) or 0)
    for source in (
        list(state.get("quota_accumulated_questions") or []),
        [
            q
            for pool in (state.get("cross_dataset_pool") or {}).values()
            if isinstance(pool, list)
            for q in pool
        ] if isinstance(state.get("cross_dataset_pool"), dict) else [],
    ):
        for question in source:
            if not isinstance(question, dict):
                continue
            try:
                added_round = int(question.get("added_round", current_round))
            except (TypeError, ValueError):
                added_round = current_round
            if added_round != current_round:
                continue
            dataset_id = str(question.get("source_dataset_id") or "")
            if not dataset_id:
                continue
            try:
                offset = int(question.get("dataset_window_offset", 0) or 0)
            except (TypeError, ValueError):
                continue
            _mark_seen_replenishment_window(seen, dataset_id, offset)
    return seen


def _current_round_selected_ids(state: EvoState | dict) -> set[str]:
    current_round = int(state.get("round_id", 0) or 0)
    selected: set[str] = set()
    sources: list[dict] = [
        q for q in list(state.get("quota_accumulated_questions") or [])
        if isinstance(q, dict)
    ]
    cross_pool = state.get("cross_dataset_pool") or {}
    if isinstance(cross_pool, dict):
        sources.extend(
            q
            for pool in cross_pool.values()
            if isinstance(pool, list)
            for q in pool
            if isinstance(q, dict)
        )
    for question in sources:
        if not isinstance(question, dict):
            continue
        try:
            added_round = int(question.get("added_round", current_round))
        except (TypeError, ValueError):
            added_round = current_round
        if added_round != current_round:
            continue
        question_id = str(question.get("question_id") or "")
        if question_id:
            selected.add(question_id)
    return selected


def _serialized_seen_with_window(state: EvoState | dict, window_meta: dict, round_id: int) -> dict:
    seen = _cached_replenishment_seen(state)
    loaded = _round_loaded_windows(state)
    attempted = _round_attempted_windows(state)
    dataset_id = str(window_meta.get("dataset_id") or state.get("active_dataset_id") or "")
    try:
        offset = int(window_meta.get("offset", state.get("current_window_offset", 0)) or 0)
    except (TypeError, ValueError):
        offset = None
    _mark_seen_replenishment_window(seen, dataset_id, offset)
    _mark_seen_replenishment_window(loaded, dataset_id, offset)
    _mark_seen_replenishment_window(attempted, dataset_id, offset)
    return {
        "cached_replenishment_windows_seen": {
            dataset_id: sorted(offsets)
            for dataset_id, offsets in seen.items()
        },
        "cached_replenishment_windows_seen_round_id": round_id,
        "replenishment_loaded_windows": {
            dataset_id: sorted(offsets)
            for dataset_id, offsets in loaded.items()
        },
        "replenishment_loaded_windows_round_id": round_id,
        "replenishment_attempted_windows": {
            dataset_id: sorted(offsets)
            for dataset_id, offsets in attempted.items()
        },
        "replenishment_attempted_windows_round_id": round_id,
    }


def _positive_demand(shortfall: dict | None) -> dict[str, float]:
    if not isinstance(shortfall, dict):
        return {}
    demand: dict[str, float] = {}
    for diff, count in shortfall.items():
        try:
            value = float(count or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            demand[str(diff)] = value
    return demand


def _initial_cached_reuse_demand(state: EvoState | dict) -> dict[str, float]:
    demand = _positive_demand(state.get("quota_shortfall") if isinstance(state.get("quota_shortfall"), dict) else None)
    if demand:
        return demand
    sampling_plan = state.get("sampling_plan") or {}
    weights = sampling_plan.get("difficulty_weights") if isinstance(sampling_plan, dict) else {}
    demand = _positive_demand(weights if isinstance(weights, dict) else None)
    if not demand:
        return {}
    selected_counts = Counter(
        str(q.get("dynamic_difficulty") or "unknown")
        for q in list(state.get("quota_accumulated_questions") or [])
        if isinstance(q, dict)
    )
    return {
        diff: value
        for diff, value in demand.items()
        if selected_counts.get(diff, 0) <= 0
    } or demand


def _cached_shortfall_reuse_offset(
    mgr: DatasetStateManager | None,
    dataset_id: str,
    window_size: int,
    shortfall: dict,
    seen_offsets: set[int] | None = None,
    blocked_question_ids: set[str] | None = None,
) -> int | None:
    if mgr is None or not dataset_id:
        return None
    ds = mgr.datasets.get(dataset_id) or mgr.load_cached(dataset_id)
    if ds is None or ds.state != DATASET_STATE_IN_USE:
        return None
    demand = _positive_demand(shortfall)
    wanted = set(demand)
    if not wanted:
        wanted = {"easy", "medium", "hard", "unknown"}
    seen_offsets = seen_offsets or set()
    blocked_question_ids = blocked_question_ids or set()
    size = max(1, int(window_size or 1))
    offset_counts: dict[int, Counter[str]] = {}
    for qid, item in ds.items.items():
        if str(qid) in blocked_question_ids:
            continue
        if item.state != ITEM_STATE_UNUSED or item.rollout_count <= 0:
            continue
        difficulty = str(item.difficulty or "unknown")
        if difficulty not in wanted:
            continue
        offset = _item_window_offset(item, size)
        if offset is None:
            continue
        if offset not in seen_offsets:
            offset_counts.setdefault(offset, Counter())[difficulty] += 1
    if not offset_counts:
        return None
    if not demand:
        return min(offset_counts)

    def score(offset: int) -> tuple[float, int, int]:
        counts = offset_counts[offset]
        contribution = 0.0
        total = 0
        for difficulty, needed in demand.items():
            count = int(counts.get(difficulty, 0) or 0)
            total += count
            contribution += min(float(count), needed) * needed
        return contribution, total, -offset

    return max(offset_counts, key=score)


def _cached_initial_reuse_offset(
    mgr: DatasetStateManager | None,
    dataset_id: str,
    window_size: int,
    state: EvoState | dict,
    seen_offsets: set[int] | None = None,
    blocked_question_ids: set[str] | None = None,
) -> int | None:
    demand = _initial_cached_reuse_demand(state)
    if not demand:
        return _dataset_reuse_offset(mgr, dataset_id, window_size)
    return _cached_shortfall_reuse_offset(
        mgr,
        dataset_id,
        window_size,
        demand,
        seen_offsets,
        blocked_question_ids,
    )


def _with_dataset_window(
    ref: dict,
    state: EvoState | dict,
    current_size: int,
    mgr: DatasetStateManager | None,
    *,
    prefer_reusable: bool = False,
    force_offset: int | None = None,
) -> dict:
    shifted = dict(ref)
    dataset_id = str(shifted.get("dataset_id", "") or "")
    if force_offset is not None:
        shifted["shard_start"] = max(0, int(force_offset or 0))
        shifted["shard_size"] = current_size
        return shifted
    if prefer_reusable:
        seen_offsets = _seen_with_current_window(state).get(dataset_id, set())
        blocked_question_ids = _current_round_selected_ids(state)
        reuse_offset = _cached_initial_reuse_offset(
            mgr,
            dataset_id,
            current_size,
            state,
            seen_offsets,
            blocked_question_ids,
        )
        if reuse_offset is not None:
            shifted["shard_start"] = reuse_offset
            shifted["shard_size"] = current_size
            return shifted
    active_dataset_id = str(state.get("active_dataset_id", "") or "")
    current_offset = int(state.get("current_window_offset", 0) or 0)
    if dataset_id == active_dataset_id:
        fallback_offset = current_offset + max(1, current_size)
    else:
        fallback_offset = 0
    shifted["shard_start"] = _dataset_next_offset(mgr, dataset_id, fallback_offset)
    shifted["shard_size"] = current_size
    return shifted


def _without_window(ref: dict) -> dict:
    cleaned = dict(ref)
    cleaned.pop("shard_start", None)
    cleaned.pop("shard_size", None)
    cleaned.pop("shard_end", None)
    cleaned.pop("shard_id", None)
    return cleaned


def _next_unseen_offset(
    mgr: DatasetStateManager | None,
    dataset_id: str,
    start_offset: int,
    window_size: int,
    seen_offsets: set[int],
) -> int:
    size = max(1, int(window_size or 1))
    offset = _dataset_next_offset(mgr, dataset_id, start_offset)
    while offset in seen_offsets:
        offset = _dataset_next_offset(mgr, dataset_id, offset + size)
    return offset


def _window_budget_remaining(state: EvoState | dict) -> bool:
    max_windows = max(1, int(state.get("max_windows_per_round", MAX_PROFILE_WINDOWS_PER_ROUND) or MAX_PROFILE_WINDOWS_PER_ROUND))
    windows_loaded = int(state.get("windows_loaded_this_round", 0) or 0)
    max_items = max(1, int(state.get("max_profile_items_per_round", MAX_PROFILE_ITEMS_PER_ROUND) or MAX_PROFILE_ITEMS_PER_ROUND))
    items_loaded = int(state.get("profile_items_loaded_this_round", 0) or 0)
    return windows_loaded < max_windows and items_loaded < max_items


def _next_dataset_ref(
    unique_refs: list[dict],
    active_ref: dict,
    current_size: int,
    mgr: DatasetStateManager | None = None,
    state: EvoState | dict | None = None,
) -> dict | None:
    state = state or {}
    seen_by_dataset = _seen_with_current_window(state)
    blocked_question_ids = _current_round_selected_ids(state)
    for ref in unique_refs:
        if _same_dataset(ref, active_ref):
            continue
        if _dataset_exhausted(mgr, str(ref.get("dataset_id", ""))):
            continue
        if _dataset_review_blacklisted(mgr, str(ref.get("dataset_id", ""))):
            continue
        base_ref = _without_window(ref)
        dataset_id = str(base_ref.get("dataset_id", ""))
        seen_offsets = seen_by_dataset.get(dataset_id, set())
        reuse_offset = _cached_shortfall_reuse_offset(
            mgr,
            dataset_id,
            current_size,
            {},
            seen_offsets,
            blocked_question_ids,
        )
        if reuse_offset is not None:
            return _with_dataset_window(base_ref, state, current_size, mgr, force_offset=reuse_offset)
        next_offset = _next_unseen_offset(mgr, dataset_id, 0, current_size, seen_offsets)
        return _with_dataset_window(base_ref, state, current_size, mgr, force_offset=next_offset)
    return None


def _cached_schema_for_ref(state: EvoState | dict, ref: dict) -> tuple[dict | None, dict | None]:
    schema_info = state.get("dataset_schema_info") or {}
    if isinstance(schema_info, dict):
        schema_ref = schema_info.get("dataset_ref")
        schema_dataset_id = str(schema_info.get("dataset_id", "") or "")
        same_schema_ref = isinstance(schema_ref, dict) and _same_dataset(schema_ref, ref)
        same_schema_id = schema_dataset_id and schema_dataset_id == str(ref.get("dataset_id", ""))
        if same_schema_ref or same_schema_id:
            schema = schema_info.get("schema")
            return (schema if isinstance(schema, dict) else None), schema_info
    return None, None


def _emit_schema_request(state: EvoState | dict, ref: dict, inspect_result: dict) -> RoutedMessage:
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    return RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.SCREENING_ENTRY,
            receiver=AgentName.DATASET_SCHEMA_AGENT,
            message_type=MessageType.DATASET_SCHEMA_REQUEST,
        ),
        payload=DatasetSchemaRequestPayload(
            dataset_ref=DatasetRef.model_validate(ref),
            inspect_result=inspect_result,
        ),
    )


def _ref_for_next_window(state: EvoState | dict) -> dict | None:
    """Return a dataset ref for shortfall replenishment, advancing offset safely."""
    active_dataset_id = str(state.get("active_dataset_id", "") or "")
    current_offset = int(state.get("current_window_offset", 0) or 0)
    current_size = int(state.get("current_window_size", 0) or DATA_WINDOW_SIZE or SCREENING_ENTRY_MAX_QUESTIONS or 500)
    mgr = _dataset_state_manager(state)
    candidate_refs: list[dict] = []

    next_ref = state.get("next_dataset_ref") or {}
    if isinstance(next_ref, dict) and next_ref.get("dataset_id"):
        candidate_refs.append(dict(next_ref))
    schema_info = state.get("dataset_schema_info") or {}
    if isinstance(schema_info, dict):
        schema_ref = schema_info.get("dataset_ref")
        if isinstance(schema_ref, dict) and schema_ref.get("dataset_id"):
            candidate_refs.append(dict(schema_ref))
    for ref in state.get("dataset_pool") or []:
        if isinstance(ref, dict):
            candidate_refs.append(dict(ref))
    for ref in state.get("dataset_review_pending_refs") or []:
        if isinstance(ref, dict):
            candidate_refs.append(dict(ref))
    for ref in state.get("previous_dataset_refs") or []:
        if isinstance(ref, dict):
            candidate_refs.append(dict(ref))

    seen: set[tuple[str, str, str]] = set()
    unique_refs: list[dict] = []
    for ref in candidate_refs:
        identity = _dataset_identity(ref)
        if not identity[0]:
            continue
        if identity in seen:
            continue
        if identity[0] != active_dataset_id and _dataset_exhausted(mgr, identity[0]):
            continue
        if _dataset_review_blacklisted(mgr, identity[0]):
            continue
        seen.add(identity)
        unique_refs.append(_without_window(ref))

    active_ref: dict | None = None
    for ref in unique_refs:
        if str(ref.get("dataset_id", "")) == active_dataset_id:
            active_ref = ref
            break
    if active_ref is None and unique_refs:
        active_ref = unique_refs[0]
    if active_ref is None:
        return None

    seen_by_dataset = _seen_with_current_window(state)
    blocked_question_ids = _current_round_selected_ids(state)
    shortfall = state.get("quota_shortfall") or {}
    if isinstance(shortfall, dict):
        for ref in unique_refs:
            dataset_id = str(ref.get("dataset_id", ""))
            if _dataset_exhausted(mgr, dataset_id) or _dataset_review_blacklisted(mgr, dataset_id):
                continue
            offset = _cached_shortfall_reuse_offset(
                mgr,
                dataset_id,
                current_size,
                shortfall,
                seen_by_dataset.get(dataset_id, set()),
                blocked_question_ids,
            )
            if offset is not None:
                return _with_dataset_window(ref, state, current_size, mgr, force_offset=offset)

    current_window = (state.get("round_data_stats") or {}).get("window", {})
    if isinstance(current_window, dict) and current_window.get("exhausted"):
        return _next_dataset_ref(unique_refs, active_ref, current_size, mgr, state)

    if not _window_budget_remaining(state):
        return None

    fallback_next_offset = current_offset + max(1, current_size)
    next_offset = _next_unseen_offset(
        mgr,
        active_dataset_id,
        fallback_next_offset,
        current_size,
        seen_by_dataset.get(active_dataset_id, set()),
    )
    row_count = _dataset_row_count(state, active_ref)
    if row_count is not None:
        remaining = max(0, row_count - next_offset)
        if remaining <= 0:
            next_ref = _next_dataset_ref(unique_refs, active_ref, current_size, mgr, state)
            return next_ref

    if _dataset_exhausted(mgr, str(active_ref.get("dataset_id", ""))):
        return _next_dataset_ref(unique_refs, active_ref, current_size, mgr, state)

    return _with_dataset_window(active_ref, state, current_size, mgr, force_offset=next_offset)


def _inspect_ref(ref: dict) -> dict | None:
    ref_inspect = _inspect_from_ref_metadata(ref)
    if ref_inspect is not None:
        return ref_inspect
    dataset_id = str(ref.get("dataset_id", ""))
    try:
        from src.tools.data_pipeline.inspect_dataset import inspect_dataset

        inspect_result = inspect_dataset(
            source=dataset_id,
            subset=ref.get("subset"),
            split=ref.get("split") or "train",
            streaming=True,
        )
    except Exception:
        return None
    return inspect_result if inspect_result and not inspect_result.get("error") else None


def _column_dtype_from_value(value: object) -> str:
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "struct"
    if value is None:
        return "none"
    return type(value).__name__.lower()


def _inspect_from_ref_metadata(ref: dict) -> dict | None:
    first_row = ref.get("source_dataset_first_row")
    schema = ref.get("source_dataset_schema")
    columns_raw = ref.get("source_dataset_columns") or []
    split_names = [str(item) for item in ref.get("source_dataset_split_names", []) if str(item)]
    if not isinstance(first_row, dict) and not columns_raw and not isinstance(schema, dict):
        return None

    column_names = [str(item) for item in columns_raw if str(item)]
    if not column_names and isinstance(first_row, dict):
        column_names = [str(key) for key in first_row.keys()]
    columns = []
    for name in column_names:
        value = first_row.get(name) if isinstance(first_row, dict) else None
        text = "" if value is None else str(value)
        columns.append({
            "name": name,
            "dtype": _column_dtype_from_value(value),
            "avg_len": len(text) if isinstance(value, str) else 0,
            "null_ratio": 1.0 if value is None else 0.0,
        })

    question_keywords = ("question", "problem", "input", "instruction", "prompt", "query")
    answer_keywords = ("answer", "output", "response", "target", "solution", "completion")
    candidates = {
        "likely_question_cols": [],
        "likely_answer_cols": [],
        "other_string_cols": [],
        "numeric_cols": [],
        "list_dict_cols": [],
    }
    for column in columns:
        name = str(column.get("name", ""))
        lowered = name.lower()
        dtype = str(column.get("dtype", "")).lower()
        if "string" in dtype or "text" in dtype:
            if any(item in lowered for item in question_keywords):
                candidates["likely_question_cols"].append(name)
            if any(item in lowered for item in answer_keywords):
                candidates["likely_answer_cols"].append(name)
            if not any(item in lowered for item in question_keywords + answer_keywords):
                candidates["other_string_cols"].append(name)
        elif any(item in dtype for item in ("int", "float", "number")):
            candidates["numeric_cols"].append(name)
        elif any(item in dtype for item in ("list", "dict", "sequence", "struct")):
            candidates["list_dict_cols"].append(name)

    requested_split = ref.get("requested_split") or ref.get("source_dataset_requested_split") or ref.get("split") or "train"
    selected_split = ref.get("split") or (split_names[0] if split_names else requested_split)
    return {
        "source": ref.get("dataset_id", ""),
        "subset": ref.get("subset"),
        "split": selected_split,
        "requested_split": requested_split,
        "num_rows": None,
        "columns": columns,
        "splits_available": split_names,
        "first_row": dict(first_row) if isinstance(first_row, dict) else {},
        "sample_rows": [dict(first_row)] if isinstance(first_row, dict) and first_row else [],
        "column_candidates": candidates,
        "schema": dict(schema) if isinstance(schema, dict) else {},
        "error": None,
    }


def _ref_with_inspected_split(ref: dict, inspect_result: dict | None) -> dict:
    updated = dict(ref)
    if not isinstance(inspect_result, dict):
        return updated
    requested_split = inspect_result.get("requested_split") or ref.get("split") or "train"
    selected_split = inspect_result.get("split") or ref.get("split") or requested_split
    if selected_split:
        updated["split"] = selected_split
    updated["requested_split"] = requested_split
    return updated


def _upsert_dataset_pool_ref(dataset_pool: list[dict], original_ref: dict, selected_ref: dict) -> list[dict]:
    updated_pool: list[dict] = []
    replaced = False
    original_key = (
        str(original_ref.get("dataset_id", "")),
        str(original_ref.get("subset") or ""),
        str(original_ref.get("split") or "train"),
    )
    selected_key = (
        str(selected_ref.get("dataset_id", "")),
        str(selected_ref.get("subset") or ""),
        str(selected_ref.get("split") or "train"),
    )
    for item in dataset_pool:
        item_key = (
            str(item.get("dataset_id", "")),
            str(item.get("subset") or ""),
            str(item.get("split") or "train"),
        )
        if item_key in {original_key, selected_key}:
            if not replaced:
                updated_pool.append(selected_ref)
                replaced = True
            continue
        updated_pool.append(item)
    if not replaced:
        updated_pool.append(selected_ref)
    return updated_pool


def _select_next_ref(refs: list[dict], dataset_pool: list[dict], pool_cursor: int, state: EvoState | dict) -> tuple[dict | None, dict | None, int, list[dict]]:
    mgr = _dataset_state_manager(state)
    consumed_cursor = pool_cursor
    current_size = int(state.get("current_window_size", 0) or DATA_WINDOW_SIZE or SCREENING_ENTRY_MAX_QUESTIONS or 500)
    desired_style = _desired_target_style(state)
    failures = [
        failure for failure in list(state.get("screening_load_failures") or [])
        if isinstance(failure, dict)
    ]
    refs = _skip_failed_refs(refs, failures)
    dataset_pool = _skip_failed_refs(dataset_pool, failures)

    seen_refs: set[tuple[str, str, str]] = set()
    reusable_refs: list[dict] = []
    for ref in list(dataset_pool) + list(refs):
        identity = _dataset_identity(ref)
        if not identity[0] or identity in seen_refs:
            continue
        seen_refs.add(identity)
        dataset_id = str(ref.get("dataset_id", ""))
        if _dataset_exhausted(mgr, dataset_id):
            continue
        if _dataset_review_blacklisted(mgr, dataset_id):
            continue
        if _dataset_in_use(mgr, dataset_id) and _matches_target_style(state, dataset_id, desired_style):
            reusable_refs.append(ref)

    for ref in reusable_refs:
        ref_with_window = _with_dataset_window(ref, state, current_size, mgr, prefer_reusable=True)
        inspect_result = _inspect_ref(ref_with_window)
        if inspect_result is not None:
            ref_with_window = _ref_with_inspected_split(ref_with_window, inspect_result)
            dataset_pool = _upsert_dataset_pool_ref(dataset_pool, ref, ref_with_window)
            return ref_with_window, inspect_result, consumed_cursor, dataset_pool

    while consumed_cursor < len(dataset_pool):
        ref = dataset_pool[consumed_cursor]
        dataset_id = str(ref.get("dataset_id", ""))
        consumed_cursor += 1
        if _dataset_exhausted(mgr, dataset_id):
            continue
        if _dataset_review_blacklisted(mgr, dataset_id):
            continue
        if _dataset_in_use(mgr, dataset_id) and not _matches_target_style(state, dataset_id, desired_style):
            continue
        ref_with_window = _with_dataset_window(ref, state, current_size, mgr)
        inspect_result = _inspect_ref(ref_with_window)
        if inspect_result is not None:
            ref_with_window = _ref_with_inspected_split(ref_with_window, inspect_result)
            dataset_pool = _upsert_dataset_pool_ref(dataset_pool, ref, ref_with_window)
            return ref_with_window, inspect_result, consumed_cursor, dataset_pool

    for ref in refs:
        dataset_id = str(ref.get("dataset_id", ""))
        if _dataset_exhausted(mgr, dataset_id):
            continue
        if _dataset_review_blacklisted(mgr, dataset_id):
            continue
        if _dataset_in_use(mgr, dataset_id) and not _matches_target_style(state, dataset_id, desired_style):
            continue
        ref_with_window = _with_dataset_window(ref, state, current_size, mgr)
        inspect_result = _inspect_ref(ref_with_window)
        if inspect_result is not None:
            ref_with_window = _ref_with_inspected_split(ref_with_window, inspect_result)
            dataset_pool = _upsert_dataset_pool_ref(dataset_pool, ref, ref_with_window)
            return ref_with_window, inspect_result, len(dataset_pool), dataset_pool

    return None, None, consumed_cursor, dataset_pool


def _emit_materialized_message(state: EvoState, dataset_refs: list[DatasetRef], all_questions: list[dict]) -> RoutedMessage:
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    questions_ref = write_json_artifact(
        trace_id=trace_id,
        round_id=round_id,
        producer="screening_entry",
        name="materialized_questions",
        data=all_questions,
    )
    payload = MaterializedDatasetPayload(
        dataset_refs=dataset_refs,
        questions=[],
        questions_ref=questions_ref,
        materialization_summary=f"materialized {len(all_questions)} questions",
    )
    return RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.SCREENING_ENTRY,
            receiver=AgentName.FILTER,
            message_type=MessageType.MATERIALIZED_DATASET,
        ),
        payload=payload,
    )


def _register_loaded_questions(state: EvoState | dict, dataset_id: str, questions: list[dict]) -> None:
    mgr = _dataset_state_manager(state)
    if mgr is None or not dataset_id or not questions:
        return
    mgr.init_dataset_from_questions(dataset_id, questions)


def _mark_dataset_exhausted_after_empty_tail(state: EvoState | dict, ref: dict, window_meta: dict) -> None:
    mgr = _dataset_state_manager(state)
    dataset_id = str(ref.get("dataset_id", "") or window_meta.get("dataset_id", "") or "")
    if mgr is None or not dataset_id:
        return
    if str(window_meta.get("failure_reason") or "") != "no_questions_loaded":
        return
    cleaner_stats = window_meta.get("cleaner_stats")
    processed_count = None
    if isinstance(cleaner_stats, dict):
        processed_count = cleaner_stats.get("processed_count")
    if processed_count is None:
        processed_count = window_meta.get("processed_count")
    try:
        processed = int(processed_count)
    except (TypeError, ValueError):
        return
    if processed <= 0:
        mgr.mark_exhausted(dataset_id)


def _screening_failure_record(ref: dict, window_meta: dict, reason: str | None = None) -> dict:
    return {
        "dataset_id": str(ref.get("dataset_id", "") or window_meta.get("dataset_id", "") or ""),
        "subset": ref.get("subset"),
        "split": ref.get("split") or window_meta.get("split") or "train",
        "requested_split": ref.get("requested_split") or window_meta.get("requested_split"),
        "offset": int(window_meta.get("offset", ref.get("shard_start", 0)) or 0),
        "limit": int(window_meta.get("limit", ref.get("shard_size", 0)) or 0),
        "reason": str(reason or window_meta.get("failure_reason") or "no_questions_loaded"),
    }


def _screening_failure_key(ref_or_failure: dict) -> tuple[str, str, str, int]:
    try:
        offset = int(ref_or_failure.get("offset", ref_or_failure.get("shard_start", 0)) or 0)
    except (TypeError, ValueError):
        offset = 0
    return (
        str(ref_or_failure.get("dataset_id", "")),
        str(ref_or_failure.get("subset") or ""),
        str(ref_or_failure.get("split") or "train"),
        max(0, offset),
    )


def _serialized_failed_window_attempt(state: EvoState | dict, failure: dict, round_id: int) -> dict:
    seen = _cached_replenishment_seen(state)
    attempted = _round_attempted_windows(state)
    dataset_id = str(failure.get("dataset_id") or state.get("active_dataset_id") or "")
    try:
        offset = int(failure.get("offset", state.get("current_window_offset", 0)) or 0)
    except (TypeError, ValueError):
        offset = None
    _mark_seen_replenishment_window(seen, dataset_id, offset)
    _mark_seen_replenishment_window(attempted, dataset_id, offset)
    return {
        "cached_replenishment_windows_seen": {
            dataset_id: sorted(offsets)
            for dataset_id, offsets in seen.items()
        },
        "cached_replenishment_windows_seen_round_id": round_id,
        "replenishment_attempted_windows": {
            dataset_id: sorted(offsets)
            for dataset_id, offsets in attempted.items()
        },
        "replenishment_attempted_windows_round_id": round_id,
    }


def _skip_failed_refs(refs: list[dict], failures: list[dict]) -> list[dict]:
    failed = {
        _screening_failure_key(failure)
        for failure in failures
        if isinstance(failure, dict)
    }
    return [
        ref for ref in refs
        if _screening_failure_key(ref) not in failed
    ]


def _failed_dataset_ids(failures: list[dict]) -> set[str]:
    return {
        str(failure.get("dataset_id", ""))
        for failure in failures
        if isinstance(failure, dict) and str(failure.get("dataset_id", ""))
    }


def _fresh_search_request_after_load_failures(
    state: EvoState | dict,
    failures: list[dict],
    consumed_ids: set[str],
    *,
    reason: str,
) -> dict:
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    goal = str(state.get("user_goal", "") or "reasoning instruction")
    sampling_plan = dict(state.get("sampling_plan") or {})
    recovery_attempts = int(state.get("loader_recovery_attempts", 0) or 0)
    max_attempts = max(1, int(DATA_WINDOW_RETRY_LIMIT or 1))
    skipped_dataset_ids = _failed_dataset_ids(failures)
    consumed = set(str(item) for item in consumed_ids if str(item))
    consumed.update(skipped_dataset_ids)
    if recovery_attempts >= max_attempts:
        msg = RoutedMessage(
            header=MessageHeader(
                trace_id=trace_id,
                round_id=round_id,
                sender=AgentName.SCREENING_ENTRY,
                receiver=AgentName.FORMAT_ERROR,
                message_type=MessageType.FORMAT_ERROR,
            ),
            payload=FormatErrorPayload(
                bad_message_type=MessageType.MATERIALIZED_DATASET.value,
                missing_fields=[],
                reason=(
                    "dataset loader recovery exhausted; no loadable dataset refs "
                    f"after {len(failures)} failure(s)"
                ),
            ),
        )
        print(
            "[screening_entry] Loader recovery exhausted; stopping before empty training data "
            f"after attempts={recovery_attempts} failures={len(failures)}"
        )
        return {
            "materialized_dataset_questions": [],
            "rollout_runs": None,
            "pending_message": msg,
            "round_data_stats": {
                **(state.get("round_data_stats") or {}),
                "total_loaded": 0,
                "loader_recovery_action": "exhausted",
                "loader_recovery_reason": reason,
                "screening_load_failures": list(failures),
                "low_train_signal": True,
            },
            "screening_load_failures": failures,
            "consumed_dataset_ids": sorted(consumed),
            "dataset_pool": _skip_failed_refs(
                [ref for ref in state.get("dataset_pool") or [] if isinstance(ref, dict)],
                failures,
            ),
            "previous_dataset_refs": _skip_failed_refs(
                [ref for ref in state.get("previous_dataset_refs") or [] if isinstance(ref, dict)],
                failures,
            ),
            "pool_cursor": 0,
            "data_replenishment_needed": False,
            "data_replenishment_exhausted": True,
            "replenishment_cycle_active": False,
            "quota_met": False,
            "quota_accumulated_questions": list(state.get("quota_accumulated_questions") or []),
            "quota_shortfall": dict(state.get("quota_shortfall") or {"unknown": SCREENING_ENTRY_MAX_QUESTIONS}),
            "next_dataset_ref": {},
            "cached_rollout_scored_questions": [],
            "loader_recovery_attempts": recovery_attempts,
        }
    query = normalize_content_search_query(
        sampling_plan.get("search_query") or state.get("last_search_feedback", {}).get("requested_query") or goal,
        goal=goal,
        fallback=fallback_query_from_goal(goal),
    )
    sampling_plan["loader_recovery"] = {
        "reason": reason,
        "failed_dataset_ids": sorted(skipped_dataset_ids),
        "failure_count": len(failures),
        "attempt": recovery_attempts + 1,
        "max_attempts": max_attempts,
    }
    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=trace_id,
            round_id=round_id,
            sender=AgentName.SCREENING_ENTRY,
            receiver=AgentName.SEARCHER,
            message_type=MessageType.SEARCH_REQUEST,
        ),
        payload=SearchRequestPayload(
            search_sources=["huggingface", "web"],
            search_query=query,
            goal=goal,
            retrieval_mode="full_dataset",
            dataset_role="train_dataset",
            sampling_owner="filter",
            target_labels=get_classifier_labels(),
            sampling_plan=sampling_plan,
        ),
    )
    print(
        "[screening_entry] No loadable dataset refs remain; requesting fresh search "
        f"after {len(failures)} load failure(s): reason={reason}"
    )
    return {
        "materialized_dataset_questions": [],
        "rollout_runs": None,
        "pending_message": msg,
        "round_data_stats": {
            **(state.get("round_data_stats") or {}),
            "total_loaded": 0,
            "loader_recovery_action": "fresh_search",
            "loader_recovery_reason": reason,
            "screening_load_failures": list(failures),
            "low_train_signal": True,
        },
        "screening_load_failures": failures,
        "consumed_dataset_ids": sorted(consumed),
        "dataset_pool": _skip_failed_refs(
            [ref for ref in state.get("dataset_pool") or [] if isinstance(ref, dict)],
            failures,
        ),
        "previous_dataset_refs": _skip_failed_refs(
            [ref for ref in state.get("previous_dataset_refs") or [] if isinstance(ref, dict)],
            failures,
        ),
        "pool_cursor": 0,
        "data_replenishment_needed": True,
        "data_replenishment_exhausted": False,
        "replenishment_cycle_active": False,
        "quota_met": False,
        "quota_accumulated_questions": list(state.get("quota_accumulated_questions") or []),
        "quota_shortfall": dict(state.get("quota_shortfall") or {"unknown": SCREENING_ENTRY_MAX_QUESTIONS}),
        "next_dataset_ref": {},
        "cached_rollout_scored_questions": [],
        "loader_recovery_attempts": recovery_attempts + 1,
    }


def _schema_request_for_next_available_ref(
    state: EvoState | dict,
    refs: list[dict],
    dataset_pool: list[dict],
    pool_cursor: int,
    failures: list[dict],
) -> dict | None:
    selectable_refs = _skip_failed_refs(refs, failures)
    selectable_pool = _skip_failed_refs(dataset_pool, failures)
    ref, inspect_result, consumed_cursor, updated_pool = _select_next_ref(
        selectable_refs,
        selectable_pool,
        min(pool_cursor, len(selectable_pool)),
        {**state, "screening_load_failures": failures},
    )
    if ref is None or inspect_result is None:
        return None
    return {
        "pending_message": _emit_schema_request(state, ref, inspect_result),
        "previous_dataset_refs": refs,
        "dataset_pool": updated_pool,
        "pool_cursor": consumed_cursor,
        "consumed_dataset_ids": list(state.get("consumed_dataset_ids") or []),
        "screening_load_failures": failures,
        "next_dataset_ref": ref,
    }


def screening_entry_node(state: EvoState) -> dict:
    t0 = time.time()
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    trace_id = str(state.get("trace_id", ""))
    round_id = int(state.get("round_id", 0) or 0)
    message_type = pending_message.header.message_type
    consumed_ids = set(state.get("consumed_dataset_ids") or [])
    replenishment_mode = bool(state.get("data_replenishment_needed"))
    screening_load_failures = [
        failure for failure in list(state.get("screening_load_failures") or [])
        if isinstance(failure, dict)
    ]
    window_delta = 0
    item_delta = 0

    if replenishment_mode:
        next_ref = _ref_for_next_window(state)
        if next_ref is None:
            msg = _emit_materialized_message(state, [], [])
            windows_loaded = int(state.get("windows_loaded_this_round", 0) or 0)
            items_loaded = int(state.get("profile_items_loaded_this_round", 0) or 0)
            mgr = _dataset_state_manager(state)
            all_candidates_exhausted = all(
                _dataset_exhausted(mgr, str(ref.get("dataset_id", ""))) or _dataset_review_blacklisted(mgr, str(ref.get("dataset_id", "")))
                for ref in (
                    [state.get("next_dataset_ref")] +
                    [state.get("dataset_schema_info", {}).get("dataset_ref")] +
                    list(state.get("dataset_pool", [])) +
                    list(state.get("previous_dataset_refs", []))
                )
                if isinstance(ref, dict) and ref.get("dataset_id")
            )
            return {
                "materialized_dataset_questions": [],
                "rollout_runs": None,
                "pending_message": msg,
                "round_data_stats": {
                    **(state.get("round_data_stats") or {}),
                    "total_loaded": 0,
                    "window": {"exhausted": True, "reason": "no_replenishment_ref"},
                },
                "windows_loaded_this_round": windows_loaded,
                "profile_items_loaded_this_round": items_loaded,
                "data_replenishment_needed": False,
                "data_replenishment_exhausted": True,
                "replenishment_cycle_active": False,
                "quota_met": all_candidates_exhausted,
                "quota_accumulated_questions": [],
            }
        schema_override, schema_info = _cached_schema_for_ref(state, next_ref)
        if schema_info is None:
            inspect_result = _inspect_ref(next_ref)
            if inspect_result is not None:
                next_ref = _ref_with_inspected_split(next_ref, inspect_result)
                return {
                    "pending_message": _emit_schema_request(state, next_ref, inspect_result),
                    "next_dataset_ref": next_ref,
                    "data_replenishment_needed": True,
                    "data_replenishment_exhausted": False,
                    "replenishment_cycle_active": True,
                    "windows_loaded_this_round": int(state.get("windows_loaded_this_round", 0) or 0),
                    "profile_items_loaded_this_round": int(state.get("profile_items_loaded_this_round", 0) or 0),
                }
            schema_info = {
                "dataset_id": next_ref.get("dataset_id", ""),
                "dataset_ref": next_ref,
                "schema": {},
                "inspect_result": {},
            }
        all_questions, window_meta = _load_dataset_from_ref(
            next_ref,
            schema_override=schema_override,
            state={**state, "current_window_offset": next_ref.get("shard_start"), "current_window_size": next_ref.get("shard_size")},
        )
        window_delta, item_delta, window_meta = _account_window_exploration(
            state,
            next_ref,
            all_questions,
            window_meta,
        )
        _register_loaded_questions(state, str(next_ref.get("dataset_id", "")), all_questions)
        if all_questions:
            consumed_ids.add(str(next_ref.get("dataset_id", "")))
        else:
            _mark_dataset_exhausted_after_empty_tail(state, next_ref, window_meta)
            failure = _screening_failure_record(next_ref, window_meta)
            screening_load_failures.append(failure)
            failed_seen_update = _serialized_failed_window_attempt(state, failure, round_id)
            fallback = _schema_request_for_next_available_ref(
                {
                    **state,
                    **failed_seen_update,
                    "consumed_dataset_ids": list(consumed_ids),
                },
                [
                    ref for ref in state.get("previous_dataset_refs") or []
                    if isinstance(ref, dict)
                ],
                [
                    ref for ref in state.get("dataset_pool") or []
                    if isinstance(ref, dict)
                ],
                int(state.get("pool_cursor", 0) or 0),
                screening_load_failures,
            )
            if fallback is not None:
                print(
                    "[screening_entry] Replenishment load produced no questions; "
                    f"trying next dataset after {failure['dataset_id']}: {failure['reason']}"
                )
                return {
                    **fallback,
                    **failed_seen_update,
                    "data_replenishment_needed": True,
                    "data_replenishment_exhausted": False,
                    "replenishment_cycle_active": True,
                    "windows_loaded_this_round": int(state.get("windows_loaded_this_round", 0) or 0),
                    "profile_items_loaded_this_round": int(state.get("profile_items_loaded_this_round", 0) or 0),
                }
            recovery = _fresh_search_request_after_load_failures(
                {
                    **state,
                    **failed_seen_update,
                    "consumed_dataset_ids": list(consumed_ids),
                },
                screening_load_failures,
                consumed_ids,
                reason=str(failure.get("reason", "no_questions_loaded")),
            )
            recovery.update(failed_seen_update)
            recovery.update({
                "windows_loaded_this_round": int(state.get("windows_loaded_this_round", 0) or 0),
                "profile_items_loaded_this_round": int(state.get("profile_items_loaded_this_round", 0) or 0),
            })
            if recovery["pending_message"].header.message_type != MessageType.FORMAT_ERROR:
                recovery["data_replenishment_needed"] = True
                recovery["replenishment_cycle_active"] = True
            return recovery
        msg = _emit_materialized_message(state, [DatasetRef.model_validate(next_ref)], all_questions)
        windows_loaded = int(state.get("windows_loaded_this_round", 0) or 0) + window_delta
        items_loaded = int(state.get("profile_items_loaded_this_round", 0) or 0) + item_delta
        round_data_stats = {
            **(state.get("round_data_stats") or {}),
            "total_loaded": len(all_questions),
            "window": window_meta,
            "replenishment": True,
        }
        seen_update = _serialized_seen_with_window(state, window_meta, round_id)
        print(
            f"[screening_entry] Replenishment window {windows_loaded}/"
            f"{int(state.get('max_windows_per_round', MAX_PROFILE_WINDOWS_PER_ROUND) or MAX_PROFILE_WINDOWS_PER_ROUND)} "
            f"loaded={len(all_questions)} new_items={item_delta} "
            f"new_exploration={bool(window_delta)} shortfall={state.get('quota_shortfall') or {}}"
        )
        return {
            "materialized_dataset_questions": all_questions,
            "rollout_runs": None,
            "pending_message": msg,
            "round_data_stats": round_data_stats,
            "consumed_dataset_ids": list(consumed_ids),
            "dataset_schema_info": schema_info,
            "active_dataset_id": window_meta.get("dataset_id", ""),
            "current_window_offset": int(window_meta.get("offset", 0) or 0),
            "current_window_size": int(window_meta.get("limit", 0) or 0),
            "windows_loaded_this_round": windows_loaded,
            "profile_items_loaded_this_round": items_loaded,
            **seen_update,
            "screening_load_failures": screening_load_failures,
            "data_replenishment_needed": False,
            "data_replenishment_exhausted": not bool(all_questions),
            "replenishment_cycle_active": True,
            "next_dataset_ref": next_ref,
        }

    if message_type == MessageType.DATASET_SCHEMA_RESULT:
        schema_replenishment = bool(
            state.get("replenishment_cycle_active")
            or state.get("data_replenishment_needed")
        )
        schema_result = DatasetSchemaResultPayload.model_validate(pending_message.payload)
        ref = _ref_with_inspected_split(
            schema_result.dataset_ref.model_dump(),
            schema_result.inspect_result,
        )
        refs = state.get("previous_dataset_refs") or [ref]
        dataset_pool = state.get("dataset_pool") or []
        pool_cursor = int(state.get("pool_cursor", 0) or 0)

        all_questions, window_meta = _load_dataset_from_ref(
            ref,
            schema_override=schema_result.schema,
            state=state,
        )
        window_delta, item_delta, window_meta = _account_window_exploration(
            state,
            ref,
            all_questions,
            window_meta,
        )
        _register_loaded_questions(state, str(ref.get("dataset_id", "")), all_questions)
        if all_questions:
            consumed_ids.add(str(ref.get("dataset_id", "")))
            if ref not in dataset_pool:
                dataset_pool.append(ref)
        else:
            _mark_dataset_exhausted_after_empty_tail(state, ref, window_meta)
            failure = _screening_failure_record(ref, window_meta)
            screening_load_failures.append(failure)
            failed_seen_update = _serialized_failed_window_attempt(state, failure, round_id)
            fallback = _schema_request_for_next_available_ref(
                {
                    **state,
                    **failed_seen_update,
                    "consumed_dataset_ids": list(consumed_ids),
                },
                [
                    candidate for candidate in refs
                    if isinstance(candidate, dict)
                ],
                [
                    candidate for candidate in dataset_pool
                    if isinstance(candidate, dict)
                ],
                pool_cursor,
                screening_load_failures,
            )
            if fallback is not None:
                print(
                    "[screening_entry] Load produced no questions; "
                    f"trying next dataset after {failure['dataset_id']}: {failure['reason']}"
                )
                return {
                    **fallback,
                    **failed_seen_update,
                    "data_replenishment_needed": bool(schema_replenishment),
                    "data_replenishment_exhausted": False,
                    "replenishment_cycle_active": bool(schema_replenishment),
                    "windows_loaded_this_round": int(state.get("windows_loaded_this_round", 0) or 0),
                    "profile_items_loaded_this_round": int(state.get("profile_items_loaded_this_round", 0) or 0),
                }
            recovery = _fresh_search_request_after_load_failures(
                {
                    **state,
                    **failed_seen_update,
                    "consumed_dataset_ids": list(consumed_ids),
                },
                screening_load_failures,
                consumed_ids,
                reason=str(failure.get("reason", "no_questions_loaded")),
            )
            recovery.update(failed_seen_update)
            recovery.update({
                "windows_loaded_this_round": int(state.get("windows_loaded_this_round", 0) or 0),
                "profile_items_loaded_this_round": int(state.get("profile_items_loaded_this_round", 0) or 0),
            })
            if recovery["pending_message"].header.message_type != MessageType.FORMAT_ERROR:
                recovery["data_replenishment_needed"] = True
                recovery["data_replenishment_exhausted"] = False
                recovery["replenishment_cycle_active"] = bool(schema_replenishment)
            return recovery

        schema_info = {
            "dataset_id": ref.get("dataset_id", ""),
            "dataset_ref": ref,
            "schema": schema_result.schema,
            "inspect_result": schema_result.inspect_result,
            "window": window_meta,
        }
        print(f"[screening_entry] Schema {schema_info['dataset_id']}: {schema_info['schema']}")
        msg = _emit_materialized_message(state, [DatasetRef.model_validate(ref)], all_questions)
        consumed_cursor = pool_cursor
    else:
        schema_replenishment = False
        search_result = SearchResultPayload.model_validate(pending_message.payload)
        refs = [d.model_dump() for d in search_result.datasets]
        dataset_pool = state.get("dataset_pool") or []
        pool_cursor = int(state.get("pool_cursor", 0) or 0)

        ref, inspect_result, consumed_cursor, dataset_pool = _select_next_ref(
            refs,
            dataset_pool,
            pool_cursor,
            state,
        )
        if ref is not None and inspect_result is not None:
            schema_request = RoutedMessage(
                header=MessageHeader(
                    trace_id=trace_id,
                    round_id=round_id,
                    sender=AgentName.SCREENING_ENTRY,
                    receiver=AgentName.DATASET_SCHEMA_AGENT,
                    message_type=MessageType.DATASET_SCHEMA_REQUEST,
                ),
                payload=DatasetSchemaRequestPayload(
                    dataset_ref=DatasetRef.model_validate(ref),
                    inspect_result=inspect_result,
                ),
            )
            return {
                "pending_message": schema_request,
                "previous_dataset_refs": refs,
                "dataset_pool": dataset_pool,
                "pool_cursor": consumed_cursor,
                "consumed_dataset_ids": list(consumed_ids),
            }
        if screening_load_failures:
            return _fresh_search_request_after_load_failures(
                {**state, "previous_dataset_refs": refs, "dataset_pool": dataset_pool},
                screening_load_failures,
                consumed_ids,
                reason="no_loadable_refs_after_failures",
            )

        all_questions = []
        window_meta = {}
        schema_info = state.get("dataset_schema_info") or {}
        msg = _emit_materialized_message(state, search_result.datasets, all_questions)

    print(f"[screening_entry] Processing {len(refs)} refs, pool={len(dataset_pool)} cursor={consumed_cursor}")
    if not all_questions:
        print("[screening_entry] WARNING: No questions loaded!")

        # If we have no questions and no available refs (all rejected by reviewer or failed),
        # trigger a fresh search instead of continuing with empty data
        if len(refs) == 0 and len(dataset_pool) == 0:
            print("[screening_entry] No accepted datasets and no pool remaining; requesting fresh search to avoid empty training")
            # Create a synthetic failure record to track this situation
            synthetic_failure = {
                "dataset_id": "all_datasets",
                "reason": "all_datasets_rejected_or_unavailable",
                "offset": 0,
                "limit": 0,
            }
            return _fresh_search_request_after_load_failures(
                {**state, "previous_dataset_refs": refs, "dataset_pool": dataset_pool},
                [synthetic_failure],
                consumed_ids,
                reason="all_datasets_rejected_or_unavailable",
            )

    elapsed = time.time() - t0
    print(f"[screening_entry] Node completed in {elapsed:.2f}s")

    round_data_stats = {
        "total_loaded": len(all_questions),
        "datasets_tried": len(refs),
        "window": window_meta,
    }
    windows_loaded_prior = int(state.get("windows_loaded_this_round", 0) or 0)
    items_loaded_prior = int(state.get("profile_items_loaded_this_round", 0) or 0)
    if message_type == MessageType.DATASET_SCHEMA_RESULT and schema_replenishment:
        windows_loaded = windows_loaded_prior + window_delta
        items_loaded = items_loaded_prior + item_delta
    else:
        windows_loaded = window_delta
        items_loaded = item_delta
    seen_update = _serialized_seen_with_window(state, window_meta, round_id) if window_meta else {}

    return {
        "materialized_dataset_questions": all_questions,
        "rollout_runs": None,
        "pending_message": msg,
        "previous_dataset_refs": refs,
        "round_data_stats": round_data_stats,
        "dataset_pool": dataset_pool,
        "pool_cursor": consumed_cursor,
        "consumed_dataset_ids": list(consumed_ids),
        "screening_load_failures": screening_load_failures,
        "dataset_schema_info": schema_info,
        "active_dataset_id": window_meta.get("dataset_id", ""),
        "current_window_offset": int(window_meta.get("offset", 0) or 0),
        "current_window_size": int(window_meta.get("limit", 0) or 0),
        "windows_loaded_this_round": windows_loaded,
        "profile_items_loaded_this_round": items_loaded,
        **seen_update,
        "max_windows_per_round": max(1, int(state.get("max_windows_per_round", MAX_PROFILE_WINDOWS_PER_ROUND) or MAX_PROFILE_WINDOWS_PER_ROUND)),
        "max_profile_items_per_round": max(1, int(state.get("max_profile_items_per_round", MAX_PROFILE_ITEMS_PER_ROUND) or MAX_PROFILE_ITEMS_PER_ROUND)),
        "data_replenishment_needed": False,
        "data_replenishment_exhausted": False,
        "replenishment_cycle_active": bool(message_type == MessageType.DATASET_SCHEMA_RESULT and schema_replenishment),
        "quota_met": True,
        "quota_shortfall": {},
        "next_dataset_ref": ref if all_questions else {},
    }
