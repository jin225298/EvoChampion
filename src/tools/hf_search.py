"""
Hugging Face dataset search tool.

Searches HF Hub for datasets matching a query. When primary search
yields no results, expands to token-variant queries derived from the
input. Falls back to configured dataset IDs only as last resort.
Fallback format: "dataset_id:subset:split" (subset/split optional).
"""

from collections.abc import Iterable
from importlib import import_module
from typing import Protocol, cast

from config.settings import DATASET_SHARD_COUNT, DATASET_SHARD_SIZE, SEARCH_DATASET_REPO_LIMIT, SEARCH_FALLBACK_DATASETS, SEARCH_FALLBACK_MODE
from src.tools.search_query import is_difficulty_only_search_query


def _local_code_benchmark_ref() -> dict | None:
    """Return a ref for the local code benchmark when DOMAIN=code.

    For the code-domain smoke run the training data is the local
    ``BENCHMARK_DATASET_ID`` directory (e.g. data/code_smoke). Routing it
    through search→review→screening lets it be standardized, rollout-judged by
    execution, and trained on, exactly like a searched HF dataset — without
    depending on HF search returning a code-with-tests dataset.
    """
    import os
    from pathlib import Path

    if os.getenv("DOMAIN", "").strip().lower() != "code":
        return None
    bench = os.getenv("BENCHMARK_DATASET_ID", "").strip()
    if not bench:
        return None
    # Only short-circuit for a real local path; HF repo ids fall through.
    if not Path(bench).expanduser().exists():
        return None
    split = os.getenv("BENCHMARK_SPLIT", "train").strip() or "train"
    return {
        "dataset_id": bench,
        "source": "local",
        "subset": None,
        "split": split,
        "requested_split": split,
    }


class _DatasetInfoLike(Protocol):
    id: str


class _HfApiLike(Protocol):
    def list_datasets(self, *, search: str, limit: int) -> Iterable[_DatasetInfoLike]: ...


def HfApi() -> _HfApiLike:
    api_class = getattr(import_module("huggingface_hub"), "HfApi")
    return cast(_HfApiLike, api_class())


def _is_hf_http_error(exc: BaseException) -> bool:
    try:
        error_type = getattr(import_module("huggingface_hub.errors"), "HfHubHTTPError")
    except Exception:
        return False
    return isinstance(error_type, type) and isinstance(exc, error_type)


def build_dataset_shard_refs(dataset_id: str, limit: int | None = None, subset: str = "main", split: str = "train") -> list[dict]:
    """为单个数据集生成 N 个分片引用，供并行加载。

    大数据集一次性加载会 OOM / 超时，拆成 shard_count 个分片，
    每个分片包含起止偏移 (shard_start, shard_end) 和合并组标识 (merge_group)，
    下游 screening_entry 可以按组并行物化后合并。
    """
    shard_count = max(1, int(DATASET_SHARD_COUNT or 1))
    selected_count = min(shard_count, int(limit or shard_count))
    return [
        {
            "dataset_id": dataset_id,
            "source": "huggingface",
            "subset": subset,
            "split": split,
            "local_cache_path": f"{dataset_id.replace('/', '_')}_shard_{idx}",
            "score_hint": 1.0 - (idx / max(shard_count, 1)) * 0.01,
            "shard_id": idx,
            "shard_size": DATASET_SHARD_SIZE,
            "shard_start": idx * DATASET_SHARD_SIZE,
            "shard_end": (idx + 1) * DATASET_SHARD_SIZE,
            "merge_group": f"{dataset_id}_demo_{shard_count}x{DATASET_SHARD_SIZE}",
        }
        for idx in range(selected_count)
    ]


def _hf_api_search(query: str, limit: int) -> list[dict]:
    """调用 HuggingFace Hub API 按关键词搜索数据集。

    返回 list[dict]，每个 dict 包含 dataset_id / source。
    网络异常、API 限流、HTTP 错误全部静默返回空列表，不抛异常。
    """
    try:
        api = HfApi()
        results = list(api.list_datasets(search=query, limit=limit))
        print(f"[hf_search] Found {len(results)} datasets via HF API for query='{query}'")
        return [
            {"dataset_id": item.id, "source": "huggingface", "subset": None, "split": None}
            for item in results
        ]
    except Exception as exc:
        if isinstance(exc, (ConnectionError, OSError)) or _is_hf_http_error(exc):
            print(f"[hf_search] HF API unavailable for query='{query}' ({type(exc).__name__}: {exc})")
            return []
        print(f"[hf_search] Unexpected HF error for query='{query}': {type(exc).__name__}: {exc}")
        return []


def _expand_queries_from_goal(primary_query: str) -> list[str]:
    """将主查询词拆分为多个变体，提高命中率。

    去停用词后生成：原始查询 → 全关键词拼接 → 前两词 → 后两词。
    例如 "improve math reasoning ability" →
    ["improve math reasoning ability", "math reasoning ability", "math reasoning", "reasoning ability"]
    """
    tokens = primary_query.lower().split()
    stopwords = {"a", "an", "the", "and", "or", "of", "in", "to", "for", "with", "is", "on", "at", "by"}
    keywords = [t for t in tokens if t not in stopwords and len(t) > 1]
    if not keywords:
        return [primary_query]
    variants = [primary_query]
    if len(keywords) >= 2:
        variants.append(" ".join(keywords))
    if len(keywords) >= 3:
        variants.append(" ".join(keywords[:2]))
        variants.append(" ".join(keywords[-2:]))
    return list(dict.fromkeys(variants))


def search_hf_datasets(
    query: str,
    dataset_repo_limit: int | None = None,
) -> list[dict]:
    """HuggingFace 数据集搜索主入口，三级降级策略。

    1. 主查询 → 直接调 HF API
    2. 主查询无结果 → 扩展为多个词条变体，逐个查询直到命中或耗尽
    3. 全部无结果 → 根据 SEARCH_FALLBACK_MODE 决定：
       - "predefined": 返回 .env 中配置的兜底数据集列表
       - "empty": 返回空列表

    返回 list[dict]，每个 dict 包含 dataset_id / source / subset / split。
    """
    # Code-domain smoke run: use the local benchmark directory as the training
    # data source directly, bypassing HF search (which rarely returns a
    # code-with-tests dataset and tends to surface unrelated repos).
    local_ref = _local_code_benchmark_ref()
    if local_ref is not None:
        print(f"[hf_search] Code domain: using local benchmark as training data: {local_ref['dataset_id']}")
        return [local_ref]

    limit = dataset_repo_limit or SEARCH_DATASET_REPO_LIMIT

    if is_difficulty_only_search_query(query):
        print(f"[hf_search] Skipping difficulty-only query='{query}'")
        return []

    datasets = _hf_api_search(query, limit)

    if not datasets:
        expansion_queries = _expand_queries_from_goal(query)
        for expansion_query in expansion_queries:
            if expansion_query == query:
                continue
            sub_datasets = _hf_api_search(expansion_query, limit=max(1, limit // 2))
            datasets.extend(sub_datasets)
            if len(datasets) >= limit:
                break
        if datasets:
            seen: set[str] = set()
            unique: list[dict] = []
            for d in datasets:
                key = d["dataset_id"]
                if key not in seen:
                    seen.add(key)
                    unique.append(d)
            datasets = unique[:limit]
            print(f"[hf_search] Multi-query expansion: {len(datasets)} unique datasets across queries")

    if datasets:
        return datasets

    if SEARCH_FALLBACK_MODE == "predefined" and SEARCH_FALLBACK_DATASETS:
        fallback_entries = [ds.strip() for ds in SEARCH_FALLBACK_DATASETS.split(",") if ds.strip()]
        for entry in fallback_entries[:limit]:
            parts = entry.split(":")
            ds_id = parts[0]
            subset = parts[1] if len(parts) > 1 and parts[1] else None
            split = parts[2] if len(parts) > 2 and parts[2] else "train"
            datasets.append({
                "dataset_id": ds_id,
                "source": "huggingface",
                "subset": subset,
                "split": split,
            })
        print(f"[hf_search] Fallback (predefined): returning {len(datasets)} datasets: {[d['dataset_id'] for d in datasets]}")
        return datasets

    print("[hf_search] Fallback (empty): no datasets available offline")
    return []
