from config.settings import SEARCH_DATASET_REPO_LIMIT, load_harness_dataset_repo_limit
from src.models.messages import (
    AgentName,
    DatasetRef,
    MessageHeader,
    MessageType,
    RoutedMessage,
    SearchResultPayload,
    SearchRequestPayload,
)
from src.models.state import EvoState
from src.tools.dataset_state import DatasetStateManager
from src.tools.hf_search import build_dataset_shard_refs, search_hf_datasets


def _ref_key(ref: dict) -> tuple[str, str, str, int | None]:
    raw_start = ref.get("shard_start")
    try:
        shard_start = int(raw_start) if raw_start is not None else None
    except (TypeError, ValueError):
        shard_start = None
    return (
        str(ref.get("dataset_id", "")),
        str(ref.get("subset") or ""),
        str(ref.get("split") or "train"),
        shard_start,
    )


def _merge_unique_refs(existing: list[dict], additions: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str, str, int | None]] = set()
    for ref in list(existing) + list(additions):
        if not isinstance(ref, dict):
            continue
        key = _ref_key(ref)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        merged.append(dict(ref))
    return merged


def hf_search_tool_node(state: EvoState) -> dict:
    pending_message = state.get("pending_message")
    if pending_message is None:
        raise ValueError("pending_message missing")
    request = SearchRequestPayload.model_validate(pending_message.payload)

    # Progressive search: each generation fetches more results;
    # consumed_ids filtering removes duplicates, yielding 100-200, 200-300 etc.
    generation = int(state.get("search_generation", 0) or 0) + 1
    progressive_limit = max(SEARCH_DATASET_REPO_LIMIT, SEARCH_DATASET_REPO_LIMIT * generation)

    datasets = []
    if "huggingface" in request.search_sources:
        datasets.extend(
            search_hf_datasets(
                query=request.search_query,
                dataset_repo_limit=progressive_limit,
            )
        )
    action_metadata = state.get("current_action_metadata", {})
    if (
        isinstance(action_metadata, dict)
        and action_metadata.get("dataset_selection_mode") == "merge_shards"
        and datasets
        and all(d.get("dataset_id") == datasets[0].get("dataset_id") for d in datasets)
    ):
        primary_id = datasets[0].get("dataset_id", "")
        subset = datasets[0].get("subset", "main")
        split = datasets[0].get("split", "train")
        if primary_id:
            datasets = build_dataset_shard_refs(primary_id, limit=load_harness_dataset_repo_limit(), subset=subset, split=split)

    consumed_ids = set(state.get("consumed_dataset_ids") or [])
    new_refs = [DatasetRef.model_validate(d).model_dump(mode="json") for d in datasets]
    filtered_refs = [r for r in new_refs if str(r.get("dataset_id", "")) not in consumed_ids]

    # --- cache hit tracking ---
    cache_hit_ids: list[str] = []
    cache_miss_ids: list[str] = []
    mgr: DatasetStateManager | None = None
    cache_path = state.get("dataset_states_path", "")
    if cache_path:
        from pathlib import Path
        mgr = DatasetStateManager(Path(str(cache_path)))
    for ref in new_refs:
        ds_id = str(ref.get("dataset_id", ""))
        if not ds_id:
            continue
        if mgr and (mgr.datasets.get(ds_id) or mgr.load_cached(ds_id)):
            cache_hit_ids.append(ds_id)
        else:
            cache_miss_ids.append(ds_id)
    # --- end cache hit tracking ---

    print(
        f"[hf_search_tool] Round {state.get('round_id', 0)}: "
        f"query='{request.search_query}' "
        f"results={len(new_refs)} filtered_new={len(filtered_refs)} "
        f"cache_hit={len(cache_hit_ids)} cache_miss={len(cache_miss_ids)}"
    )

    pending_refs = _merge_unique_refs(
        list(state.get("dataset_review_pending_refs") or []),
        filtered_refs,
    )
    payload = SearchResultPayload(
        datasets=[DatasetRef.model_validate(d) for d in filtered_refs],
        search_summary=f"found {len(filtered_refs)} new candidate datasets",
    )

    msg = RoutedMessage(
        header=MessageHeader(
            trace_id=str(state.get("trace_id", "")),
            round_id=int(state.get("round_id", 0) or 0),
            sender=AgentName.HF_SEARCH_TOOL,
            receiver=AgentName.DATASET_REVIEWER,
            message_type=MessageType.SEARCH_RESULT,
        ),
        payload=payload,
    )
    last_search_feedback = {
        "round_id": state.get("round_id", 0),
        "requested_query": request.search_query,
        "search_sources": list(request.search_sources),
        "result_count": len(new_refs),
        "new_result_count": len(filtered_refs),
        "top_dataset_ids": [str(r.get("dataset_id", "")) for r in new_refs[:10]],
        "cache_hit_count": len(cache_hit_ids),
        "cache_miss_count": len(cache_miss_ids),
        "cache_hit_ids": cache_hit_ids[:10],
        "search_summary": payload.search_summary,
    }

    return {
        "candidate_dataset_refs": new_refs,
        # Keep unreviewed HF refs out of dataset_pool. screening_entry/filter only
        # consume dataset_pool, so pending refs must stay isolated until
        # dataset_reviewer accepts them.
        "dataset_pool": list(state.get("dataset_pool") or []),
        "dataset_review_pending_refs": pending_refs,
        "dataset_review_job_id": "",
        "dataset_review_active": bool(pending_refs),
        "dataset_review_completed": not bool(pending_refs),
        "dataset_review_drained_count": 0,
        "pool_cursor": int(state.get("pool_cursor", 0) or 0),
        "search_generation": generation,
        "last_search_feedback": last_search_feedback,
        "pending_message": msg,
    }
