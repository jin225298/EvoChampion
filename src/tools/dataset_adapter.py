"""
机制层：自动检测 HuggingFace 数据集的字段 schema，归一化为统一的题目-答案格式。

策略无关：只提供纯数据转换接口。
上层（screening_entry、filter）负责战略决策——加载哪些数据集、取多少条、用哪个子集。
"""

import ast
import importlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.settings import USE_LLM_AGENTS
from src.tools.agent_prompts import DATASET_INSPECTOR_PROMPT
from src.tools.llm_decision import decide_json_leaf, prompt_for_agent
from src.tools.question_fields import TARGET_STYLE_ANSWER, TARGET_STYLE_COT, infer_target_style, normalize_target_style

# 题目字段候选名（按优先级排列，扫描时先命中先用）
_QUESTION_FIELD_CANDIDATES = (
    "informal_statement",
    "formal_statement",
    "theorem_statement",
    "statement",
    "question",
    "problem",
    "input",
    "instruction",
    "prompt",
    "query",
    "text",
    "context",
)

# 答案字段候选名（按优先级排列）
_ANSWER_FIELD_CANDIDATES = (
    "answer",
    "golden_answer",
    "gold_answer",
    "final_answer",
    "expected_answer",
    "output",
    "response",
    "target",
    "solution",
    "completion",
    "label",
)

_FINAL_ANSWER_FIELD_CANDIDATES = (
    "golden_answer",
    "gold_answer",
    "final_answer",
    "expected_answer",
    "answer",
    "target",
    "label",
)

_PROOF_FIELD_NAMES = (
    "formal_proof",
    "proof",
    "lean_proof",
    "coq_proof",
    "isabelle_proof",
)

_PROOF_REFERENCE_FIELD_CANDIDATES = (
    "informal_proof",
    "proof",
    "solution",
    "reasoning",
    "rationale",
    "explanation",
)

_TRAIN_OUTPUT_FIELD_CANDIDATES = (
    "informal_proof",
    "think_response",
    "cot",
    "chain_of_thought",
    "reasoning",
    "rationale",
    "solution",
    "thinking",
    "thought",
    "think",
    "no_think_response",
    "output",
    "response",
    "completion",
)

_COT_TEXT_MARKERS = (
    "<think",
    "</think>",
    "####",
    "step by step",
    "let's solve",
    "let us solve",
    "we need",
    "we have",
    "therefore",
    "thus",
    "so the answer",
    "solution:",
    "reasoning",
    "解：",
    "解析",
    "因此",
    "所以",
)
_COT_LENGTH_THRESHOLD = 600

_FORMAL_CODE_MARKERS = (
    "import ",
    "theorem ",
    "lemma ",
    "example ",
    "begin",
    "end",
    ":=",
    "#eval",
    "def ",
    "by ",
    "qed",
    "coq",
    "lean",
    "isabelle",
)

# 多轮对话字段名。此类字段不是普通文本列，必须由确定性解析器拆出 user/assistant。
_CONVERSATION_FIELDS = (
    "messages",
    "conversations",
    "conversation",
    "dialogue",
    "dialog",
    "chat",
    "turns",
    "chosen",
)

_CONVERSATION_ROLE_KEYS = ("role", "from", "speaker")
_CONVERSATION_CONTENT_KEYS = ("content", "value", "text")
_USER_ROLES = {"user", "human", "prompt", "instruction"}
_ASSISTANT_ROLES = {"assistant", "gpt", "bot", "model", "completion", "response"}
_METADATA_FIELD_NAMES = {
    "id",
    "uid",
    "uuid",
    "idx",
    "index",
    "conversation_id",
    "unique_id",
    "source",
    "src",
    "data_source",
    "dataset",
    "dataset_id",
    "domain",
    "category",
    "system",
    "split",
    "subset",
    "tag",
    "label_name",
    "language",
    "lang",
    "metadata",
    "tok_len",
    "token_count",
    "length",
}

_TRANSCRIPT_ROLE_RE = re.compile(
    r"(?im)(?:^|\n)\s*(user|human|prompt|instruction|assistant|gpt|bot|model|completion|response)\s*:\s*"
)

# 字符串类型白名单（dtype 包含这些 token 的字段被认为是文本）
_STRING_DTYPES = {"string", "large_string", "utf8", "text"}


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _hfd_cache_only_enabled() -> bool:
    cache_mode = os.getenv("DATASET_CACHE_MODE", "").strip().lower()
    return (
        _env_truthy("HFD_DATASET_CACHE_ONLY", default=False)
        or _env_truthy("DATASET_OFFSET_CACHE_MODE", default=False)
        or cache_mode in {"offset", "offline", "cache_only", "cache-only"}
    )


def _safe_hfd_dataset_dir_name(dataset_id: str) -> str:
    name = dataset_id.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "dataset"


def _hfd_dataset_cache_root() -> Path:
    configured = os.getenv("HFD_DATASET_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
    if hf_datasets_cache:
        return Path(hf_datasets_cache).expanduser() / "hfd"
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    return hf_home / "datasets" / "hfd"


def _hfd_script_path() -> str | None:
    configured = os.getenv("HFD_SCRIPT_PATH")
    if configured:
        return configured
    for candidate in ("hfd", "hfd.sh"):
        found = shutil.which(candidate)
        if found:
            return found
    repo_script = Path.cwd() / "hfd.sh"
    if repo_script.exists():
        return str(repo_script)
    return None


def _hfd_download_command(script_path: str, dataset_id: str, local_dir: Path) -> list[str]:
    cmd = [
        script_path,
        dataset_id,
        "--dataset",
        "--local-dir",
        str(local_dir),
    ]
    tool = os.getenv("HFD_DOWNLOAD_TOOL", "").strip()
    if tool:
        cmd.extend(["--tool", tool])
    threads = os.getenv("HFD_DOWNLOAD_THREADS", "").strip()
    if threads:
        cmd.extend(["-x", threads])
    jobs = os.getenv("HFD_DOWNLOAD_JOBS", "").strip()
    if jobs:
        cmd.extend(["-j", jobs])
    revision = os.getenv("HFD_DATASET_REVISION", "").strip()
    if revision:
        cmd.extend(["--revision", revision])
    exclude = os.getenv("HFD_DATASET_EXCLUDE", "*.md README* .gitattributes").split()
    if exclude:
        cmd.append("--exclude")
        cmd.extend(exclude)
    return cmd


def prepare_hfd_dataset_source(dataset_id: str) -> str:
    """Download a HF dataset with hfd when configured, returning the load source.

    If hfd is disabled or no hfd script is available, returns the original HF
    dataset id so callers keep the standard datasets.load_dataset behavior.
    """
    if not _env_truthy("USE_HFD_DATASET_DOWNLOAD", default=False):
        return dataset_id

    local_dir = _hfd_dataset_cache_root() / _safe_hfd_dataset_dir_name(dataset_id)
    if _hfd_cache_only_enabled():
        if local_dir.is_dir():
            print(f"[dataset_adapter] Using existing hfd dataset cache only: {local_dir}")
            return str(local_dir)
        print(
            "[dataset_adapter] hfd cache-only mode: no local cache found; "
            f"using datasets.load_dataset cache path for {dataset_id}"
        )
        return dataset_id

    script_path = _hfd_script_path()
    if not script_path:
        print("[dataset_adapter] hfd.sh not found; falling back to datasets.load_dataset")
        return dataset_id

    if local_dir.is_dir() and any(local_dir.iterdir()):
        print(f"[dataset_adapter] Using cached HF dataset: {local_dir}")
        return str(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cmd = _hfd_download_command(script_path, dataset_id, local_dir)
    print(
        "[dataset_adapter] Downloading HF dataset via hfd: "
        f"dataset_id={dataset_id} local_dir={local_dir}"
    )
    subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    return str(local_dir)


def get_existing_hfd_dataset_source(dataset_id: str) -> str | None:
    """Return an existing hfd dataset cache path without downloading anything."""
    local_dir = _hfd_dataset_cache_root() / _safe_hfd_dataset_dir_name(dataset_id)
    if local_dir.is_dir():
        return str(local_dir)
    return None


def load_hf_dataset_with_fallback(
    dataset_id: str,
    subset: str | None,
    split: str,
    *,
    streaming: bool = False,
    allow_hfd: bool = True,
):
    """加载 HF 数据集，自动处理 subset 参数的双向重试。

    坑：有些数据集必须传 name="main"（如 GSM8K），有些传了反而报错。
    此函数先按调用方给的 subset 加载，失败后切换 subset 模式重试一次，
    两次都失败才抛原始异常。
    """
    load_dataset = importlib.import_module("datasets").load_dataset
    # Local dataset directories (code-domain smoke data shipped in the repo)
    # cannot be streamed the way hub datasets are; force non-streaming so the
    # loader reads the local train/test files directly without network access.
    if os.path.isdir(dataset_id):
        streaming = False
        allow_hfd = False
        dataset_source = dataset_id
    else:
        dataset_source = prepare_hfd_dataset_source(dataset_id) if allow_hfd else dataset_id
    load_kwargs: dict[str, Any] = {}
    if dataset_source != dataset_id:
        load_kwargs["cache_dir"] = str(_hfd_dataset_cache_root())

    normalized_subset = subset if subset and subset != "default" else None
    try:
        return load_dataset(
            dataset_source,
            name=normalized_subset,
            split=split,
            streaming=streaming,
            **load_kwargs,
        )
    except Exception as first_exc:
        alternate_subset = None if normalized_subset else "main"
        if alternate_subset == normalized_subset:
            raise
        try:
            return load_dataset(
                dataset_source,
                name=alternate_subset,
                split=split,
                streaming=streaming,
                **load_kwargs,
            )
        except Exception:
            raise first_exc


def load_cached_hf_dataset(dataset_id: str, subset: str | None, split: str):
    """Load a dataset from the HF hub cache, working in offline mode.

    For script-based datasets (e.g. openai_humaneval) that cannot be loaded
    via ``load_dataset`` when ``HF_HUB_OFFLINE=1``, this function finds the
    cached parquet/jsonl files and loads them directly.  Falls back to the
    normal ``load_hf_dataset_with_fallback`` when the cache lookup fails.

    The cache layout is::

        $HF_HOME/hub/datasets--<id>/snapshots/<rev>/
            <subset>/train-*.parquet      # when subset is given
            test-*.parquet                # when no subset (root level)

    Subset directory names match the HF config name (e.g. ``full`` for MBPP).
    """
    from datasets import load_dataset

    # First try the normal path (works for non-script datasets even offline).
    try:
        return load_hf_dataset_with_fallback(dataset_id, subset, split, allow_hfd=False)
    except Exception:
        pass

    # Locate the cache directory for this dataset.
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    hub_dir = Path(hf_home) / "hub"
    cache_name = "datasets--" + dataset_id.replace("/", "--")
    cache_dir = hub_dir / cache_name
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"Dataset {dataset_id!r} not found in HF cache at {cache_dir}"
        )

    # Pick the latest snapshot.
    snapshot = sorted(snapshots_dir.iterdir())[-1]

    # Determine search directory: subset subdirectory or snapshot root.
    search_dir = snapshot
    if subset and subset not in ("default", "main", ""):
        candidate = snapshot / subset
        if candidate.is_dir():
            search_dir = candidate

    # Look for parquet files matching the requested split, then jsonl.
    # Use rglob to find files in nested subdirectories (e.g. HumanEval stores
    # parquet under openai_humaneval/test/0000.parquet).
    parquet_files = sorted(search_dir.rglob("*.parquet"))
    jsonl_files = sorted(search_dir.rglob("*.jsonl")) + sorted(search_dir.rglob("*.jsonl.gz"))
    json_files = sorted(search_dir.rglob("*.json"))

    data_files = parquet_files or jsonl_files or json_files
    if not data_files:
        # Also try subdirectories that match the split name.
        split_dir = search_dir / split
        if split_dir.is_dir():
            data_files = sorted(split_dir.rglob("*.parquet")) or sorted(split_dir.rglob("*.jsonl"))
    if not data_files:
        raise FileNotFoundError(
            f"No parquet/jsonl files found for {dataset_id!r} (subset={subset!r}, "
            f"split={split!r}) in {search_dir}"
        )

    # Prefer files whose name or parent directory matches the split string.
    # HumanEval stores parquet under openai_humaneval/test/0000.parquet (parent
    # dir = "test"); MBPP stores under full/train-*.parquet (filename = "train").
    split_files = [f for f in data_files if split in f.name or f.parent.name == split]
    if split_files:
        data_files = split_files

    fmt = "parquet" if data_files[0].suffix == ".parquet" else "json"
    ds = load_dataset(fmt, data_files=[str(f) for f in data_files], split="train")
    return ds


def detect_schema(features: Any) -> dict[str, Any]:
    """从数据集的 features metadata 自动检测题目字段和答案字段。

    按优先级扫描候选名列表，选第一个存在且 dtype 为 string 的字段。
    两套候选名都无法命中最优选择时，调用 _fallback_pair_detect 盲取。
    返回 {"question_field": "problem", "answer_field": "answer"} 或失败时两个都为 None。
    """
    if features is None:
        return {"question_field": None, "answer_field": None}

    # Normalise features to a dict of {name: dtype_str}
    field_types: dict[str, str] = {}
    if hasattr(features, "items"):
        for name, feat in features.items():
            dtype = str(getattr(feat, "dtype", "")).lower()
            field_types[str(name)] = dtype
    elif isinstance(features, dict):
        for name, feat in features.items():
            if hasattr(feat, "dtype"):
                field_types[str(name)] = str(feat.dtype).lower()
            elif isinstance(feat, dict):
                field_types[str(name)] = str(feat.get("dtype", "")).lower()
            else:
                field_types[str(name)] = str(type(feat).__name__).lower()

    formal_schema = _formal_proof_reference_schema_from_columns(list(field_types.keys()))
    if formal_schema is not None:
        return formal_schema

    question_field = _pick_alpaca_question_field(field_types.keys()) or _pick_field(field_types, _QUESTION_FIELD_CANDIDATES)
    answer_field = _pick_field(field_types, _ANSWER_FIELD_CANDIDATES)
    conversation_field = _pick_conversation_field(field_types.keys())

    if question_field is not None and answer_field is not None:
        schema: dict[str, Any] = {"question_field": question_field, "answer_field": answer_field}
        rollout_field = _pick_field_by_name(field_types.keys(), ("answer", "target", "label"))
        train_field = _pick_field_by_name(field_types.keys(), _TRAIN_OUTPUT_FIELD_CANDIDATES)
        if rollout_field:
            schema["rollout_gold_field"] = rollout_field
        if train_field:
            schema["train_output_field"] = train_field
        schema["target_style"] = TARGET_STYLE_COT if train_field and (rollout_field or answer_field) and train_field != (rollout_field or answer_field) else TARGET_STYLE_ANSWER
        dedup_field = _pick_dedup_field(field_types.keys())
        if dedup_field:
            schema["dedup_key_field"] = dedup_field
        return schema

    if conversation_field is not None:
        return _conversation_schema(
            conversation_field,
            dedup_key_field=_pick_dedup_field(field_types.keys()),
        )

    if question_field is None and answer_field is None:
        question_field, answer_field = _fallback_pair_detect(field_types)

    return {"question_field": question_field, "answer_field": answer_field}


def detect_schema_from_item(item: dict) -> dict[str, Any]:
    """从首行数据反推 schema（流式加载时 features metadata 不可用的降级方案）。

    根据字段的 Python 运行时类型推断：str → "string"，int/float/bool → 类型名，None → "none"。
    """
    field_types: dict[str, str] = {}
    for name, value in item.items():
        if isinstance(value, str):
            field_types[str(name)] = "string"
        elif isinstance(value, (int, float, bool)):
            field_types[str(name)] = type(value).__name__.lower()
        elif value is None:
            field_types[str(name)] = "none"
        else:
            field_types[str(name)] = type(value).__name__.lower()

    formal_schema = _formal_proof_reference_schema_from_columns([str(name) for name in item.keys()])
    if formal_schema is not None:
        return formal_schema

    question_field = _pick_alpaca_question_field(item.keys(), item) or _pick_field(field_types, _QUESTION_FIELD_CANDIDATES)
    answer_field = _pick_field(field_types, _ANSWER_FIELD_CANDIDATES)
    if question_field is not None and answer_field is not None:
        schema: dict[str, Any] = {"question_field": question_field, "answer_field": answer_field}
        rollout_field = _pick_field_by_name(item.keys(), _FINAL_ANSWER_FIELD_CANDIDATES)
        train_field = _pick_field_by_name(item.keys(), _TRAIN_OUTPUT_FIELD_CANDIDATES)
        if rollout_field:
            schema["rollout_gold_field"] = rollout_field
        if train_field:
            schema["train_output_field"] = train_field
        schema["target_style"] = _target_style_from_item(item, train_field, rollout_field or answer_field)
        dedup_field = _pick_dedup_field(item.keys())
        if dedup_field:
            schema["dedup_key_field"] = dedup_field
        return schema

    conversation_field = _find_parseable_conversation_field(item)
    if conversation_field is not None:
        return _conversation_schema(
            conversation_field,
            dedup_key_field=_pick_dedup_field(item.keys()),
        )

    transcript_field = _find_parseable_transcript_field(item)
    if transcript_field is not None:
        return _conversation_schema(
            transcript_field,
            dedup_key_field=_pick_dedup_field(item.keys()),
            reason="deterministic role-prefixed transcript schema",
        )

    if question_field is None and answer_field is None:
        question_field, answer_field = _fallback_pair_detect(field_types)
    return {"question_field": question_field, "answer_field": answer_field}


def _pick_field(field_types: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    """从候选名列表中选中第一个存在于数据集中且类型兼容的字段。

    两轮扫描：第一轮只选 string 类型，第二轮放宽到任意类型。
    """
    for candidate in candidates:
        if candidate in field_types:
            dtype = field_types[candidate]
            if _is_string_dtype(dtype):
                return candidate
    for candidate in candidates:
        if candidate in field_types:
            return candidate
    return None


def _pick_alpaca_question_field(field_names: Any, item: Mapping[str, Any] | None = None) -> str | None:
    input_col = _pick_field_by_name(field_names, ("input",))
    instruction_col = _pick_field_by_name(field_names, ("instruction",))
    output_col = _pick_field_by_name(field_names, ("output",))
    if not (input_col and instruction_col and output_col):
        return None
    if item is not None and not str(item.get(input_col) or "").strip():
        return instruction_col if str(item.get(instruction_col) or "").strip() else input_col
    return input_col


def _repair_blank_question_schema(item: Mapping[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    q_field = schema.get("question_field")
    if not isinstance(q_field, str) or _extract_text(dict(item), q_field):
        return schema
    repaired = detect_schema_from_item(dict(item))
    if not _schema_can_normalize_rows(repaired):
        return schema
    repaired_q_field = repaired.get("question_field")
    if not isinstance(repaired_q_field, str) or not _extract_text(dict(item), repaired_q_field):
        return schema
    return {**schema, **repaired}


def _merge_schema_metadata(base_schema: dict[str, Any], metadata_schema: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata_schema:
        return dict(base_schema)
    merged = dict(metadata_schema)
    merged.update(base_schema)
    return merged


def _row_id_from_item(item: Mapping[str, Any], schema: Mapping[str, Any], idx: int) -> str:
    row_id_field = schema.get("row_id_field")
    if isinstance(row_id_field, str) and row_id_field in item:
        value = item.get(row_id_field)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return str(idx)


def _question_id_part(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _question_id_for_row(dataset_id: str, row_id: str, *, subset: str | None = None, split: str | None = None) -> str:
    parts = [_question_id_part(dataset_id)]
    if subset:
        parts.append(_question_id_part(str(subset)))
    if split:
        parts.append(_question_id_part(str(split)))
    parts.append(_question_id_part(row_id))
    return "_".join(parts)


def _schema_with_row_id(base_schema: dict[str, Any], row_id_field: object) -> dict[str, Any]:
    schema = dict(base_schema)
    if isinstance(row_id_field, str) and row_id_field.strip():
        schema["row_id_field"] = row_id_field
    return schema


def _is_string_dtype(dtype: str) -> bool:
    """判断 dtype 是否属于字符串类型（子串匹配白名单）。"""
    return any(token in dtype for token in _STRING_DTYPES)


def _conversation_schema(
    conversation_field: str,
    *,
    dedup_key_field: str | None = None,
    reason: str = "deterministic conversation schema",
) -> dict[str, Any]:
    return {
        "schema_type": "conversation",
        "conversation_field": conversation_field,
        "question_field": conversation_field,
        "answer_field": conversation_field,
        "rollout_gold_field": conversation_field,
        "train_output_field": conversation_field,
        "target_style": "cot",
        "dedup_key_field": dedup_key_field or conversation_field,
        "usable": True,
        "reason": reason,
    }


def _normalise_name(name: object) -> str:
    return str(name).strip().lower()


def _pick_conversation_field(field_names: Any) -> str | None:
    names = [str(name) for name in field_names]
    lowered = {_normalise_name(name): name for name in names}
    for candidate in _CONVERSATION_FIELDS:
        if candidate in lowered:
            return lowered[candidate]
    for name in names:
        lower = _normalise_name(name)
        if lower in _METADATA_FIELD_NAMES or lower.endswith("_id"):
            continue
        if any(candidate in lower for candidate in _CONVERSATION_FIELDS):
            return name
    return None


def _pick_dedup_field(field_names: Any) -> str | None:
    names = [str(name) for name in field_names]
    lowered = {_normalise_name(name): name for name in names}
    for candidate in ("question_id", "conversation_id", "unique_id", "id", "uid", "uuid"):
        if candidate in lowered:
            return lowered[candidate]
    return None


def _pick_field_by_name(field_names: Any, candidates: tuple[str, ...]) -> str | None:
    names = [str(name) for name in field_names]
    lowered = {_normalise_name(name): name for name in names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for name in names:
        lower = _normalise_name(name)
        if any(candidate in lower for candidate in candidates):
            return name
    return None


def _pick_first_existing(field_names: Any, candidates: Any) -> str | None:
    names = [str(name) for name in field_names]
    lowered = {_normalise_name(name): name for name in names}
    for candidate in candidates or []:
        lower = _normalise_name(candidate)
        if lower in lowered:
            return lowered[lower]
    return None


def _sample_texts_for_column(
    inspect_result: dict[str, Any], column: str | None, limit: int = 5, max_chars: int | None = None
) -> list[str]:
    if not column:
        return []
    texts: list[str] = []
    for row in inspect_result.get("sample_rows", [])[:limit]:
        if not isinstance(row, Mapping):
            continue
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            texts.append(text[:max_chars])
    return texts


def _column_has_nonempty_samples(inspect_result: dict[str, Any], column: str | None) -> bool:
    return bool(_sample_texts_for_column(inspect_result, column, limit=10))


def _looks_like_cot_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if len(stripped) >= _COT_LENGTH_THRESHOLD:
        return True
    return any(marker in lowered for marker in _COT_TEXT_MARKERS)


def _column_looks_like_cot(inspect_result: dict[str, Any], column: str | None) -> bool:
    return any(_looks_like_cot_text(text) for text in _sample_texts_for_column(inspect_result, column, limit=5))


def _target_style_for_columns(inspect_result: dict[str, Any], train_col: str | None, rollout_col: str | None) -> str:
    if train_col and rollout_col and train_col != rollout_col:
        return TARGET_STYLE_COT
    if _column_looks_like_cot(inspect_result, train_col or rollout_col):
        return TARGET_STYLE_COT
    return TARGET_STYLE_ANSWER


def _target_style_from_item(item: Mapping[str, Any], train_col: str | None, rollout_col: str | None) -> str:
    if train_col and rollout_col and train_col != rollout_col:
        return TARGET_STYLE_COT
    if train_col and _looks_like_cot_text(str(item.get(train_col) or "")):
        return TARGET_STYLE_COT
    if rollout_col and _looks_like_cot_text(str(item.get(rollout_col) or "")):
        return TARGET_STYLE_COT
    return TARGET_STYLE_ANSWER


def _looks_like_formal_proof_code(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    hits = sum(1 for marker in _FORMAL_CODE_MARKERS if marker in text)
    return hits >= 2 or ("import " in text and ("begin" in text or ":=" in text))


def _is_proof_like_field(field_name: object | None) -> bool:
    if not isinstance(field_name, str) or not field_name:
        return False
    lower = _normalise_name(field_name)
    return any(name in lower for name in _PROOF_FIELD_NAMES)


def _is_formal_proof_gold(field_name: object | None, value: Any) -> bool:
    return _is_proof_like_field(field_name) and _looks_like_formal_proof_code(value)


def _formal_proof_reference_schema_from_columns(col_names: list[str]) -> dict[str, Any] | None:
    q_col = _pick_field_by_name(col_names, ("informal_statement", "formal_statement", "theorem_statement", "statement"))
    proof_col = _pick_field_by_name(col_names, ("formal_proof", "lean_proof", "coq_proof", "isabelle_proof"))
    train_col = _pick_field_by_name(col_names, ("informal_proof", "solution", "reasoning", "rationale", "explanation"))
    if not q_col or not proof_col or not train_col:
        return None
    return {
        "schema_type": "flat",
        "question_field": q_col,
        "answer_field": None,
        "rollout_gold_field": None,
        "train_output_field": train_col,
        "target_style": TARGET_STYLE_COT,
        "dedup_key_field": _pick_dedup_field(col_names) or q_col,
        "evaluation_method": "llm_judge",
        "needs_judge": True,
        "usable": True,
        "reason": "formal proof reference schema",
    }


def _pick_question_column(inspect_result: dict[str, Any], col_names: list[str], question_candidates: Any) -> str | None:
    input_col = _pick_field_by_name(col_names, ("input",))
    instruction_col = _pick_field_by_name(col_names, ("instruction",))
    output_col = _pick_field_by_name(col_names, ("output",))
    if input_col and instruction_col and output_col and _column_has_nonempty_samples(inspect_result, input_col):
        return input_col
    if input_col and instruction_col and output_col and _column_has_nonempty_samples(inspect_result, instruction_col):
        return instruction_col
    semantic_col = _pick_field_by_name(col_names, _QUESTION_FIELD_CANDIDATES)
    if semantic_col:
        return semantic_col
    return _pick_first_existing(col_names, question_candidates)


def _deterministic_flat_schema_from_inspect(inspect_result: dict[str, Any], col_names: list[str]) -> dict[str, Any] | None:
    formal_schema = _formal_proof_reference_schema_from_columns(col_names)
    if formal_schema is not None:
        return formal_schema

    candidates = inspect_result.get("column_candidates", {})
    question_candidates = candidates.get("likely_question_cols") or []
    answer_candidates = candidates.get("likely_answer_cols") or []

    q_col = _pick_question_column(inspect_result, col_names, question_candidates)
    if not q_col:
        return None

    rollout_col = (
        _pick_field_by_name(col_names, _FINAL_ANSWER_FIELD_CANDIDATES)
        or _pick_first_existing(col_names, answer_candidates)
    )
    if not rollout_col or rollout_col == q_col:
        return None

    train_col = (
        _pick_field_by_name(col_names, _TRAIN_OUTPUT_FIELD_CANDIDATES)
        or rollout_col
    )
    if train_col == q_col:
        train_col = rollout_col
    target_style = _target_style_for_columns(inspect_result, train_col, rollout_col)

    return {
        "schema_type": "flat",
        "question_field": q_col,
        "answer_field": rollout_col,
        "rollout_gold_field": rollout_col,
        "train_output_field": train_col,
        "target_style": target_style,
        "dedup_key_field": _pick_dedup_field(col_names) or q_col,
        "usable": True,
        "reason": "deterministic flat QA schema from inspect_result",
    }


def _find_parseable_conversation_field(item: Mapping[str, Any]) -> str | None:
    preferred = _pick_conversation_field(item.keys())
    ordered = []
    if preferred is not None:
        ordered.append(preferred)
    ordered.extend(str(name) for name in item.keys() if str(name) not in ordered)
    for name in ordered:
        if _extract_conversation_pair(item.get(name)) is not None:
            return name
    return None


def _find_parseable_transcript_field(item: Mapping[str, Any]) -> str | None:
    preferred = _pick_field_by_name(item.keys(), ("text", "prompt", "conversation", "messages"))
    ordered = []
    if preferred is not None:
        ordered.append(preferred)
    ordered.extend(str(name) for name in item.keys() if str(name) not in ordered)
    for name in ordered:
        lower = _normalise_name(name)
        if lower in _METADATA_FIELD_NAMES or lower.endswith("_id"):
            continue
        value = item.get(name)
        if isinstance(value, str) and _extract_role_prefixed_transcript_pair(value) is not None:
            return name
    return None


def _fallback_pair_detect(field_types: dict[str, str]) -> tuple[str | None, str | None]:
    """候选名全部未命中时的兜底策略：找任意两个 string 字段凑成题-答对。

    先从字段名中猜哪个像题目（含 question/problem/input 等关键词）、哪个像答案。
    猜不出就盲取前两个 string 字段。
    """
    string_fields = [name for name, dt in field_types.items() if _is_string_dtype(dt)]
    if len(string_fields) < 2:
        return (None, None)

    question_like: list[str] = []
    answer_like: list[str] = []
    for name in string_fields:
        lower = name.lower()
        if any(kw in lower for kw in ("question", "problem", "input", "prompt", "query")):
            question_like.append(name)
        elif any(kw in lower for kw in ("answer", "output", "response", "target", "solution")):
            answer_like.append(name)

    q = question_like[0] if question_like else string_fields[0]
    a = answer_like[0] if answer_like else (
        string_fields[1] if string_fields[0] == q else string_fields[0]
    )
    return (q, a)


def normalize_item(
    item: dict,
    schema: dict[str, Any],
    idx: int,
    dataset_id: str,
    source_dataset_split: str | None = None,
    source_dataset_subset: str | None = None,
    source_dataset_requested_split: str | None = None,
    source_dataset_split_names: list[str] | None = None,
    source_dataset_columns: list[str] | None = None,
    source_dataset_first_row: dict[str, Any] | None = None,
    source_dataset_schema: dict[str, Any] | None = None,
) -> dict | None:
    """将单行数据归一化为系统标准题目格式。

    根据 schema 提取 question_text 和 gold_answer，
    任一为空返回 None（跳过脏数据）。
    生成全局唯一的 question_id："{dataset_id}_{行号}"。
    """
    if schema.get("usable") is False:
        return None

    conversation_normalized = _normalize_conversation_item(
        item,
        schema,
        idx,
        dataset_id,
        source_dataset_split=source_dataset_split,
        source_dataset_subset=source_dataset_subset,
        source_dataset_requested_split=source_dataset_requested_split,
        source_dataset_split_names=source_dataset_split_names,
        source_dataset_columns=source_dataset_columns,
        source_dataset_first_row=source_dataset_first_row,
        source_dataset_schema=source_dataset_schema,
    )
    if conversation_normalized is not None:
        return conversation_normalized

    transcript_field = _find_parseable_transcript_field(item)
    if transcript_field is not None and schema.get("schema_type") != "flat":
        transcript_normalized = _normalize_conversation_item(
            item,
            _schema_with_row_id(_conversation_schema(
                transcript_field,
                dedup_key_field=schema.get("dedup_key_field") if isinstance(schema.get("dedup_key_field"), str) else None,
                reason="deterministic role-prefixed transcript repair",
            ), schema.get("row_id_field")),
            idx,
            dataset_id,
            source_dataset_split=source_dataset_split,
            source_dataset_subset=source_dataset_subset,
            source_dataset_requested_split=source_dataset_requested_split,
            source_dataset_split_names=source_dataset_split_names,
            source_dataset_columns=source_dataset_columns,
            source_dataset_first_row=source_dataset_first_row,
            source_dataset_schema=source_dataset_schema,
        )
        if transcript_normalized is not None:
            return transcript_normalized

    q_field = schema.get("question_field")
    a_field = schema.get("answer_field")
    rollout_field = schema.get("rollout_gold_field") or a_field
    train_field = schema.get("train_output_field") or a_field
    dedup_field = schema.get("dedup_key_field") or q_field
    raw_evaluation_method = str(schema.get("evaluation_method") or "").strip().lower()

    if q_field and (a_field == q_field or rollout_field == q_field or train_field == q_field):
        return None

    question_text = _extract_text(item, q_field if isinstance(q_field, str) else None)
    gold_answer = _extract_text(item, a_field if isinstance(a_field, str) else None)
    rollout_gold_answer = _extract_text(item, rollout_field if isinstance(rollout_field, str) else None)
    train_output = _extract_text(item, train_field if isinstance(train_field, str) else None)
    dedup_key = _extract_text(item, dedup_field if isinstance(dedup_field, str) else None)

    if not _valid_question_text(question_text, q_field):
        return None
    proof_gold = _is_formal_proof_gold(a_field, gold_answer) or _is_formal_proof_gold(rollout_field, rollout_gold_answer)
    if proof_gold and train_output:
        gold_answer = ""
        rollout_gold_answer = ""
        raw_evaluation_method = "llm_judge"

    evaluation_method = "gold" if (gold_answer or rollout_gold_answer) else (
        "llm_judge" if raw_evaluation_method == "llm_judge" or train_output else "gold"
    )
    if evaluation_method == "gold" and not (gold_answer or rollout_gold_answer):
        return None
    if evaluation_method == "llm_judge" and not train_output:
        return None
    if evaluation_method == "gold" and not _valid_answer_text(gold_answer or rollout_gold_answer, a_field):
        return None
    if evaluation_method == "gold" and _looks_like_serialized_conversation(gold_answer or rollout_gold_answer):
        return None

    explicit_target_style = schema.get("target_style")
    target_style = infer_target_style(
        train_output,
        rollout_gold_answer or gold_answer,
        gold_answer,
        explicit=explicit_target_style,
    )
    if (
        target_style == TARGET_STYLE_ANSWER
        and normalize_target_style(explicit_target_style) != TARGET_STYLE_ANSWER
        and train_field == rollout_field
        and _looks_like_cot_text(train_output or rollout_gold_answer or gold_answer)
    ):
        target_style = TARGET_STYLE_COT

    row_id = _row_id_from_item(item, schema, idx)

    # Code-domain fields are passed through so the judge can execute the tests.
    test_code = str(item.get("test") or item.get("tests") or item.get("test_code") or "").strip()
    entry_point = str(item.get("entry_point") or item.get("function_name") or "").strip()

    return {
        "question_id": _question_id_for_row(
            dataset_id,
            row_id,
            subset=source_dataset_subset,
            split=source_dataset_split,
        ),
        "question_text": question_text,
        "gold_answer": gold_answer or rollout_gold_answer,
        "rollout_gold_answer": rollout_gold_answer or gold_answer,
        "train_output": train_output or rollout_gold_answer or gold_answer,
        "target_style": target_style,
        "evaluation_method": evaluation_method,
        "needs_judge": evaluation_method == "llm_judge",
        "test": test_code,
        "entry_point": entry_point,
        "dedup_key": dedup_key or question_text,
        "source_dataset_id": dataset_id,
        "source_dataset_row_id": row_id,
        "source_dataset_split": source_dataset_split,
        "source_dataset_subset": source_dataset_subset,
        "source_dataset_requested_split": source_dataset_requested_split,
        "source_dataset_split_names": list(source_dataset_split_names or []),
        "source_dataset_columns": list(source_dataset_columns or item.keys()),
        "source_dataset_first_row": dict(source_dataset_first_row or {}),
        "source_dataset_schema": dict(source_dataset_schema or schema),
    }


def _normalize_conversation_item(
    item: dict,
    schema: dict[str, Any],
    idx: int,
    dataset_id: str,
    source_dataset_split: str | None = None,
    source_dataset_subset: str | None = None,
    source_dataset_requested_split: str | None = None,
    source_dataset_split_names: list[str] | None = None,
    source_dataset_columns: list[str] | None = None,
    source_dataset_first_row: dict[str, Any] | None = None,
    source_dataset_schema: dict[str, Any] | None = None,
) -> dict | None:
    conversation_field = schema.get("conversation_field")
    if not isinstance(conversation_field, str) or not conversation_field:
        if _schema_points_at_conversation(schema):
            conversation_field = _find_parseable_conversation_field(item)
    if not conversation_field:
        return None

    pair = _extract_conversation_pair(item.get(conversation_field))
    if pair is None:
        return None
    question_text, answer_text = pair
    if not _valid_question_text(question_text, conversation_field) or not answer_text:
        return None

    dedup_field = schema.get("dedup_key_field")
    dedup_key = _extract_text(item, dedup_field if isinstance(dedup_field, str) else None)
    row_id = _row_id_from_item(item, schema, idx)

    return {
        "question_id": _question_id_for_row(
            dataset_id,
            row_id,
            subset=source_dataset_subset,
            split=source_dataset_split,
        ),
        "question_text": question_text,
        "gold_answer": answer_text,
        "rollout_gold_answer": answer_text,
        "train_output": answer_text,
        "target_style": "cot",
        "dedup_key": dedup_key or question_text,
        "source_dataset_id": dataset_id,
        "source_dataset_row_id": row_id,
        "source_dataset_split": source_dataset_split,
        "source_dataset_subset": source_dataset_subset,
        "source_dataset_requested_split": source_dataset_requested_split,
        "source_dataset_split_names": list(source_dataset_split_names or []),
        "source_dataset_columns": list(source_dataset_columns or item.keys()),
        "source_dataset_first_row": dict(source_dataset_first_row or {}),
        "source_dataset_schema": dict(source_dataset_schema or schema),
    }


def _schema_points_at_conversation(schema: dict[str, Any]) -> bool:
    if schema.get("schema_type") == "flat":
        return False
    if schema.get("schema_type") == "conversation":
        return True
    fields = [
        schema.get("question_field"),
        schema.get("answer_field"),
        schema.get("rollout_gold_field"),
        schema.get("train_output_field"),
    ]
    return any(_normalise_name(field) in _CONVERSATION_FIELDS for field in fields if field)


def _schema_has_supervision_source(schema: dict[str, Any]) -> bool:
    gold_fields = (schema.get("answer_field"), schema.get("rollout_gold_field"))
    if any(isinstance(field, str) and field for field in gold_fields):
        return True
    raw_method = str(schema.get("evaluation_method") or "").strip().lower()
    needs_judge = bool(schema.get("needs_judge")) or raw_method == "llm_judge"
    train_field = schema.get("train_output_field")
    return needs_judge and isinstance(train_field, str) and bool(train_field)


def _schema_can_normalize_rows(schema: dict[str, Any]) -> bool:
    if schema.get("usable") is False:
        return False
    if _schema_points_at_conversation(schema):
        return True
    q_field = schema.get("question_field")
    return isinstance(q_field, str) and bool(q_field) and _schema_has_supervision_source(schema)


def _extract_text(item: dict, field_name: str | None) -> str:
    """从数据行中安全提取文本字段值。

    field_name 为 None → 空串；值为 None → 空串；
    list/dict → str() 序列化；其他 → strip 后返回。
    """
    if field_name is None:
        return ""
    value = item.get(field_name)
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return str(value)
    return str(value).strip()


def _extract_conversation_pair(value: Any) -> tuple[str, str] | None:
    """Return first user/human turn and last assistant/gpt turn from a conversation payload."""
    if isinstance(value, str):
        transcript_pair = _extract_role_prefixed_transcript_pair(value)
        if transcript_pair is not None:
            return transcript_pair

    turns = _coerce_conversation_turns(value)
    if not turns:
        return None

    user_messages: list[str] = []
    assistant_messages: list[str] = []
    positional_texts: list[str] = []

    for turn in turns:
        if isinstance(turn, str):
            text = turn.strip()
            if text:
                positional_texts.append(text)
            continue
        if not isinstance(turn, Mapping):
            continue
        role = _extract_turn_role(turn)
        content = _extract_turn_content(turn)
        if not content:
            continue
        role_lower = role.lower()
        if role_lower in _USER_ROLES:
            user_messages.append(content)
        elif role_lower in _ASSISTANT_ROLES:
            assistant_messages.append(content)
        else:
            positional_texts.append(content)

    if user_messages and assistant_messages:
        return (user_messages[0], assistant_messages[-1])

    if len(positional_texts) >= 2:
        return (positional_texts[0], positional_texts[-1])

    return None


def _extract_role_prefixed_transcript_pair(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped:
        return None

    matches = list(_TRANSCRIPT_ROLE_RE.finditer(stripped))
    if len(matches) < 2:
        return None

    turns: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        role = match.group(1).strip().lower()
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(stripped)
        content = stripped[content_start:content_end].strip()
        if content:
            turns.append((role, content))

    user_messages = [content for role, content in turns if role in _USER_ROLES]
    assistant_messages = [content for role, content in turns if role in _ASSISTANT_ROLES]
    if user_messages and assistant_messages:
        return (user_messages[0], assistant_messages[-1])
    return None


def _coerce_conversation_turns(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        for key in ("messages", "conversations", "conversation", "turns"):
            nested = value.get(key)
            if isinstance(nested, (list, tuple)):
                return list(nested)
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            return _coerce_conversation_turns(parsed)
    return []


def _extract_turn_role(turn: Mapping[str, Any]) -> str:
    for key in _CONVERSATION_ROLE_KEYS:
        value = turn.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def _extract_turn_content(turn: Mapping[str, Any]) -> str:
    for key in _CONVERSATION_CONTENT_KEYS:
        value = turn.get(key)
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        text = text.strip()
        if text:
            return text
    return ""


def _valid_question_text(text: str, field_name: object | None) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.isdigit():
        return False
    if field_name and _normalise_name(field_name) in _METADATA_FIELD_NAMES:
        return False
    if _looks_like_serialized_conversation(stripped):
        return False
    return True


def _valid_answer_text(text: str, field_name: object | None) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if field_name and _normalise_name(field_name) in _METADATA_FIELD_NAMES:
        return False
    return True


def _looks_like_serialized_conversation(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if not (
        ("role" in lowered or "from" in lowered or "speaker" in lowered)
        and ("content" in lowered or "value" in lowered or "text" in lowered)
    ):
        return stripped.startswith("[") and "user" in lowered and "assistant" in lowered
    return stripped.startswith("[") or stripped.startswith("{")


def adapt_dataset(
    dataset: Any,
    dataset_id: str,
    max_items: int | None = None,
    offset: int = 0,
    schema_override: dict[str, str | None] | None = None,
    source_dataset_split: str | None = None,
    source_dataset_subset: str | None = None,
    source_dataset_requested_split: str | None = None,
    source_dataset_split_names: list[str] | None = None,
    source_dataset_columns: list[str] | None = None,
    source_dataset_first_row: dict[str, Any] | None = None,
    source_dataset_schema: dict[str, Any] | None = None,
) -> list[dict]:
    """将任意 HuggingFace 数据集适配为归一化题目列表（主入口）。

    纯机制层——不决定取多少条、从哪偏移、选哪些数据集，这些由调用方负责。

    Args:
        dataset: HuggingFace Dataset 可迭代对象。
        dataset_id: 数据集标识符。
        max_items: 最多返回题目数，None=全部。
        offset: 起始索引，默认 0。
        schema_override: 显式指定 question_field / answer_field，跳过自动检测。
    """
    reviewer_schema = dict(source_dataset_schema) if isinstance(source_dataset_schema, dict) else None
    if schema_override is not None:
        schema = dict(schema_override)
        fallback_schema = detect_schema(getattr(dataset, "features", None))
        for key in ["question_field", "answer_field", "rollout_gold_field", "train_output_field", "target_style", "dedup_key_field", "row_id_field"]:
            if key not in schema:
                schema[key] = fallback_schema.get(key)
    else:
        schema = detect_schema(getattr(dataset, "features", None))

    questions: list[dict] = []
    loaded = 0
    for idx, item in enumerate(dataset):
        if idx < offset:
            continue
        if not _schema_can_normalize_rows(schema):
            schema = detect_schema_from_item(item)
        if not _schema_can_normalize_rows(schema):
            continue
        item_schema = _repair_blank_question_schema(item, schema)
        metadata_schema = _merge_schema_metadata(item_schema, reviewer_schema)
        normalized = normalize_item(
            item,
            metadata_schema,
            idx,
            dataset_id,
            source_dataset_split=source_dataset_split,
            source_dataset_subset=source_dataset_subset,
            source_dataset_requested_split=source_dataset_requested_split,
            source_dataset_split_names=source_dataset_split_names,
            source_dataset_columns=source_dataset_columns,
            source_dataset_first_row=source_dataset_first_row,
            source_dataset_schema=metadata_schema,
        )
        if normalized:
            questions.append(normalized)
            loaded += 1
            if max_items is not None and loaded >= max_items:
                break

    return questions


def build_schema_from_llm_columns(llm_col_decision: dict) -> dict[str, Any]:
    if not isinstance(llm_col_decision, dict):
        return {}
    row_id_source = llm_col_decision.get("row_id_source")
    conversation_field = llm_col_decision.get("conversation_field")
    if isinstance(conversation_field, str) and conversation_field:
        return _schema_with_row_id(_conversation_schema(
            conversation_field,
            dedup_key_field=llm_col_decision.get("dedup_key_source") or None,
            reason=str(llm_col_decision.get("reason", "llm conversation schema")),
        ), row_id_source)
    rollout_source = llm_col_decision.get("rollout_gold_source")
    train_source = llm_col_decision.get("train_output_source") or rollout_source
    evaluation_method = "gold" if rollout_source else "llm_judge"
    return _schema_with_row_id({
        "question_field": llm_col_decision.get("question_text_source"),
        "answer_field": rollout_source,
        "rollout_gold_field": rollout_source,
        "train_output_field": train_source,
        "target_style": infer_target_style(
            "sample-cot" if train_source and train_source != rollout_source else "",
            "sample-answer" if rollout_source else "",
            "sample-answer" if rollout_source else "",
            explicit=llm_col_decision.get("target_style"),
        ),
        "dedup_key_field": llm_col_decision.get("dedup_key_source") or llm_col_decision.get("question_text_source"),
        "evaluation_method": evaluation_method,
        "needs_judge": evaluation_method == "llm_judge",
        "usable": bool(llm_col_decision.get("usable", True)),
        "reason": str(llm_col_decision.get("reason", "")),
    }, row_id_source)


_LEAF_PROMPTS = {
    "question_text_source": "你是数据集 schema 解释器的题目列子决策器。只选择 question_text_source。输出 JSON 字段: question_text_source。值只能从 available_columns 中选择一个。不要选择 data_source、id、tag、split 这类元数据列。",
    "rollout_gold_source": "你是数据集 schema 解释器的答案列子决策器。只选择 rollout_gold_source。输出 JSON 字段: rollout_gold_source。值只能从 available_columns 中选择一个，或在没有独立最终答案/短答案/gold 列时输出 null。优先选择最终答案或短答案列。若只有思考过程/证明/完整解法列无独立答案列（thought-only 场景），必须输出 null，不要选择 train_output_source。",
    "train_output_source": "你是数据集 schema 解释器的训练输出子决策器。只选择 train_output_source。输出 JSON 字段: train_output_source。值只能从 available_columns 中选择一个。优先选择完整推理/思考过程列（reasoning/thinking/cot/rationale/solution/thought）；若数据集有独立推理列和答案列（分离场景），选推理列；若只有思考过程列无独立答案（thought-only），选该列；没有推理列时再使用最终答案列。",
    "target_style": "你是数据集 schema 解释器的训练目标格式子决策器。只选择 target_style。输出 JSON 字段: target_style。值只能是 answer 或 cot。若 train_output_source 是完整推理/详细解答/思考过程列，输出 cot；若 train_output_source 与 rollout_gold_source 不同（分离场景）或 rollout_gold_source 为 null 但有 train_output_source（thought-only），输出 cot；若 train_output_column_samples 的完整内容本身是一段多步推理/解题过程（即使与 rollout 同列），也输出 cot；若只适合最终答案，输出 answer。",
    "final_answer_marker": "你是最终答案分隔符子决策器。只输出 final_answer_marker。输出 JSON 字段: final_answer_marker。观察 rollout_gold_column_samples 的完整内容：若最终答案前有固定分隔标记（如 #### 或 'Answer:'），输出该标记字符串；若答案已在 \\boxed{} 中或无固定分隔标记，输出空字符串。",
    "dedup_key_source": "你是数据集 schema 解释器的去重键子决策器。只选择 dedup_key_source。输出 JSON 字段: dedup_key_source。值只能从 available_columns 中选择一个。优先选择真正表示题目唯一性的列，通常应与题目正文一致。",
    "row_id_source": "你是数据集 schema 解释器的行身份列子决策器。只选择 row_id_source。输出 JSON 字段: row_id_source。值只能从 available_columns 中选择一个，或在没有明确逐行样本唯一身份列时输出 null。这个字段用于区分同一题面的一题多解/多样本行；不要用题目正文列替代行身份。",
}


def _fallback_schema(col_names: list[str]) -> dict[str, Any]:
    formal_schema = _formal_proof_reference_schema_from_columns(col_names)
    if formal_schema is not None:
        return formal_schema

    q_col = _pick_field_by_name(col_names, _QUESTION_FIELD_CANDIDATES)
    r_col = _pick_field_by_name(col_names, ("answer", "target", "label")) or _pick_field_by_name(col_names, _ANSWER_FIELD_CANDIDATES)
    t_col = _pick_field_by_name(col_names, ("solution", "rationale", "reasoning", "output", "response", "completion")) or r_col
    if q_col and r_col:
        return {
            "question_field": q_col,
            "answer_field": r_col,
            "rollout_gold_field": r_col,
            "train_output_field": t_col,
            "target_style": "cot" if t_col != r_col else "answer",
            "dedup_key_field": _pick_dedup_field(col_names) or q_col,
            "usable": True,
            "reason": "fallback flat schema",
        }
    conversation_field = _pick_conversation_field(col_names)
    if conversation_field:
        return _conversation_schema(
            conversation_field,
            dedup_key_field=_pick_dedup_field(col_names),
            reason="fallback conversation schema",
        )
    return {
        "question_field": col_names[0] if col_names else None,
        "answer_field": col_names[1] if len(col_names) > 1 else None,
        "rollout_gold_field": col_names[1] if len(col_names) > 1 else None,
        "train_output_field": col_names[1] if len(col_names) > 1 else None,
        "target_style": "answer",
        "dedup_key_field": col_names[0] if col_names else None,
        "usable": len(col_names) >= 2,
        "reason": "fallback schema",
    }


def detect_columns_via_llm(
    inspect_result: dict,
    state: dict,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """用标准 leaf agent 逐字段识别列语义。"""
    columns = inspect_result.get("columns", [])
    candidates = inspect_result.get("column_candidates", {})
    col_names = [c["name"] for c in columns]
    sample_rows = inspect_result.get("sample_rows", [])[:10]

    for row in sample_rows:
        if isinstance(row, Mapping):
            transcript_field = _find_parseable_transcript_field(row)
            if transcript_field is not None:
                return _conversation_schema(
                    transcript_field,
                    dedup_key_field=_pick_dedup_field(col_names),
                    reason="deterministic role-prefixed transcript schema from inspect_result",
                )

    if not USE_LLM_AGENTS or not col_names:
        flat_schema = _deterministic_flat_schema_from_inspect(inspect_result, col_names)
        if flat_schema is not None:
            return flat_schema
        flat_question_col = _pick_field_by_name(col_names, _QUESTION_FIELD_CANDIDATES)
        flat_answer_col = _pick_field_by_name(col_names, _ANSWER_FIELD_CANDIDATES)
        conversation_field = _pick_conversation_field(col_names)
        if conversation_field and not (flat_question_col and flat_answer_col):
            return _conversation_schema(
                conversation_field,
                dedup_key_field=_pick_dedup_field(col_names),
                reason="deterministic conversation schema from inspect_result",
            )
        return _fallback_schema(col_names)

    sample_rows_compact = []
    for row in sample_rows:
        short_row = {}
        for k, v in row.items():
            sv = str(v)
            short_row[str(k)] = sv[:80] + ("..." if len(sv) > 80 else "")
        sample_rows_compact.append(short_row)
    trace_id = str(state.get("trace_id", "") or "")
    round_id = int(state.get("round_id", 0) or 0)

    context = {
        "available_columns": col_names,
        "columns": columns,
        "column_candidates": candidates,
        "sample_rows": sample_rows_compact,
        "source": inspect_result.get("source", ""),
        "subset": inspect_result.get("subset"),
        "split": inspect_result.get("split", "train"),
        "num_rows": inspect_result.get("num_rows", 0),
    }
    base_prompt = prompt_for_agent(state, "dataset_inspector", DATASET_INSPECTOR_PROMPT)

    def _ask(field: str, fallback_val: str | None) -> str | None:
        prompt = _LEAF_PROMPTS[field] + "\n\n总规则参考:\n" + base_prompt.strip()
        val, _ok = decide_json_leaf(
            agent_name=f"dataset_inspector.{field}",
            prompt=prompt,
            context=context,
            field_name=field,
            fallback_value=fallback_val,
            trace_id=trace_id,
            round_id=round_id,
            max_new_tokens=max_new_tokens,
        )
        val_str = str(val) if val is not None else None
        return val_str if val_str in col_names else None

    q_col = _ask("question_text_source", None)
    r_col = _ask("rollout_gold_source", None)
    t_col = _ask("train_output_source", None)
    d_col = _ask("dedup_key_source", None)
    row_id_col = _ask("row_id_source", None)
    full_samples = {
        "train_output_column_samples": _sample_texts_for_column(inspect_result, t_col, limit=3, max_chars=1500),
        "rollout_gold_column_samples": _sample_texts_for_column(inspect_result, r_col, limit=3, max_chars=1500),
    }
    raw_target_style, _ok = decide_json_leaf(
        agent_name="dataset_inspector.target_style",
        prompt=_LEAF_PROMPTS["target_style"] + "\n\n总规则参考:\n" + base_prompt.strip(),
        context={**context, "rollout_gold_source": r_col, "train_output_source": t_col, **full_samples},
        field_name="target_style",
        fallback_value=None,
        trace_id=trace_id,
        round_id=round_id,
        max_new_tokens=max_new_tokens,
    )
    raw_marker, _marker_ok = decide_json_leaf(
        agent_name="dataset_inspector.final_answer_marker",
        prompt=_LEAF_PROMPTS["final_answer_marker"] + "\n\n总规则参考:\n" + base_prompt.strip(),
        context={**context, "rollout_gold_source": r_col, "train_output_source": t_col, **full_samples},
        field_name="final_answer_marker",
        fallback_value=None,
        trace_id=trace_id,
        round_id=round_id,
        max_new_tokens=max_new_tokens,
    )
    final_answer_marker = raw_marker.strip() if isinstance(raw_marker, str) and raw_marker.strip() else None

    if q_col and r_col and (q_col == r_col or _normalise_name(r_col) in _METADATA_FIELD_NAMES):
        q_col = None
        r_col = None
    if t_col and q_col and t_col == q_col:
        t_col = r_col

    thought_only_schema = bool(q_col and t_col and not r_col)
    if not q_col or (not r_col and not thought_only_schema):
        flat_schema = _deterministic_flat_schema_from_inspect(inspect_result, col_names)
        if flat_schema is not None:
            return flat_schema
        flat_question_col = _pick_field_by_name(col_names, _QUESTION_FIELD_CANDIDATES)
        flat_answer_col = _pick_field_by_name(col_names, _ANSWER_FIELD_CANDIDATES)
        conversation_field = _pick_conversation_field(col_names)
        if conversation_field and not (flat_question_col and flat_answer_col):
            return _conversation_schema(
                conversation_field,
                dedup_key_field=_pick_dedup_field(col_names),
                reason="deterministic conversation schema from inspect_result",
            )
        return _fallback_schema(col_names)

    if not t_col:
        t_col = r_col
    if not d_col:
        d_col = q_col

    usable = bool(q_col and (r_col or t_col))
    reason = "llm leaf decision"

    explicit_style = normalize_target_style(raw_target_style)
    target_style = explicit_style or _target_style_for_columns(inspect_result, t_col, r_col)
    evaluation_method = "gold" if r_col else "llm_judge"

    return {
        "question_field": q_col,
        "answer_field": r_col,
        "rollout_gold_field": r_col,
        "train_output_field": t_col,
        "target_style": target_style,
        "final_answer_marker": final_answer_marker,
        "dedup_key_field": d_col,
        "evaluation_method": evaluation_method,
        "needs_judge": evaluation_method == "llm_judge",
        **({"row_id_field": row_id_col} if row_id_col else {}),
        "usable": usable,
        "reason": reason,
    }
