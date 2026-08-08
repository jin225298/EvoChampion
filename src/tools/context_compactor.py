"""Domain-neutral structured context compaction for LLM agent requests.

Compacts large context dicts at the LLM agent request boundary by:
  - Summarizing large dict/list/string values (replacing with length info + sample)
  - Keeping small scalar fields intact
  - Preserving priority fields when present:
      round_id, goal, user_goal, fallback, deterministic_policy,
      hard_ratio, probe_acc, rollback_streak, target fields,
      decision fields, schema-like fields
  - Optionally writing full context to an artifact under the session dir
    when trace_id is available

The compaction policy is generic and configurable via the PRESERVED_PRIORITY_FIELDS
set and size thresholds. No domain-specific hardcoding in the mechanism layer.
"""

import json
import os
from pathlib import Path
from typing import Any

from config.settings import CONTEXT_COMPACTION_PRETTY_MAX_BYTES, get_session_dir


# Fields that are always preserved regardless of size (priority fields).
PRESERVED_PRIORITY_FIELDS: set[str] = {
    # Round/identification
    "round_id", "goal", "user_goal",
    # Decision/fallback
    "fallback", "deterministic_policy",
    # Data quality
    "hard_ratio", "probe_acc", "rollback_streak",
    # Strategy/schema
    "target_labels", "target_questions_per_round", "finetuning_type",
    "lora_rank", "lora_alpha", "sampling_method",
}

# Fields whose names contain any of these substrings are considered
# schema-like / decision-like / target-like and are preserved.
_PRESERVED_SUBSTRINGS: tuple[str, ...] = (
    "_schema", "_decision", "_target", "_policy",
    "target_", "decision_", "schema_",
)

# Thresholds (in characters for strings, in items for collections)
_LARGE_STRING_THRESHOLD: int = int(os.environ.get("COMPACTOR_LARGE_STRING_CHARS", "500"))
_LARGE_LIST_THRESHOLD: int = int(os.environ.get("COMPACTOR_LARGE_LIST_ITEMS", "20"))
_LARGE_DICT_THRESHOLD: int = int(os.environ.get("COMPACTOR_LARGE_DICT_KEYS", "15"))
_MAX_SAMPLE_ITEMS: int = int(os.environ.get("COMPACTOR_MAX_SAMPLE_ITEMS", "5"))


def _is_priority_field(key: str) -> bool:
    """Check if a field name is a priority field and should be preserved."""
    if key in PRESERVED_PRIORITY_FIELDS:
        return True
    key_lower = key.lower()
    for substr in _PRESERVED_SUBSTRINGS:
        if substr.lower() in key_lower:
            return True
    return False


def _compact_value(key: str, value: Any) -> Any:
    """Compact a single value, preserving priority fields."""
    # Priority fields are always kept as-is
    if _is_priority_field(key):
        return value

    # Small scalars pass through
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > _LARGE_STRING_THRESHOLD:
            return {
                "_summary": f"str(len={len(value)})",
                "_preview": value[:_MAX_SAMPLE_ITEMS * 100][:500],
            }
        return value

    # Compact large lists
    if isinstance(value, list):
        if len(value) <= _LARGE_LIST_THRESHOLD:
            return [_compact_value(f"{key}[i]", item) for item in value]
        return {
            "_summary": f"list(len={len(value)})",
            "_sample": [_compact_value(f"{key}[i]", item) for item in value[:_MAX_SAMPLE_ITEMS]],
        }

    # Compact large dicts
    if isinstance(value, dict):
        if len(value) <= _LARGE_DICT_THRESHOLD:
            return {k: _compact_value(k, v) for k, v in value.items()}
        # For large dicts, keep priority keys and summarize the rest
        compact: dict[str, Any] = {}
        summary_keys: list[str] = []
        for k, v in value.items():
            if _is_priority_field(k):
                compact[k] = _compact_value(k, v)
            else:
                summary_keys.append(k)
        compact["_summary"] = f"dict(keys={len(value)})"
        compact["_keys"] = summary_keys[:_MAX_SAMPLE_ITEMS]
        if len(summary_keys) > _MAX_SAMPLE_ITEMS:
            compact["_keys"].append(f"... and {len(summary_keys) - _MAX_SAMPLE_ITEMS} more")
        return compact

    return value


def _write_full_context_artifact(
    trace_id: str,
    round_id: int,
    agent_name: str,
    context: dict[str, Any],
) -> str | None:
    """Write the full uncompacted context to an artifact file.

    Returns the artifact path if written, None otherwise.
    """
    if not trace_id:
        return None
    try:
        session_dir = get_session_dir(trace_id)
        artifact_dir = session_dir / "context_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"r{round_id}_{agent_name}_full_context.json"
        compact = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(compact.encode("utf-8")) > CONTEXT_COMPACTION_PRETTY_MAX_BYTES:
            payload = compact
        else:
            payload = json.dumps(context, ensure_ascii=False, indent=2)
        artifact_path.write_text(
            payload,
            encoding="utf-8",
        )
        return str(artifact_path)
    except Exception:
        return None


def compact_context(
    context: dict[str, Any],
    trace_id: str = "",
    round_id: int = 0,
    agent_name: str = "",
    *,
    write_artifact: bool = True,
) -> dict[str, Any]:
    """Compact a context dict for inclusion in an LLM agent request.

    Args:
        context: The full context dict to compact.
        trace_id: Session trace ID (used for artifact writing).
        round_id: Current round number.
        agent_name: Name of the agent making the request.
        write_artifact: If True and trace_id is available, write full context
            to an artifact file and include a reference.

    Returns:
        Compacted context dict with a context_summary entry describing
        what was compacted. If write_artifact is True and trace_id is set,
        includes full_context_ref pointing to the artifact.
    """
    if not context:
        return {}

    compacted: dict[str, Any] = {}
    stats = {"original_keys": len(context), "compacted_keys": 0, "skipped_keys": 0}

    for key, value in context.items():
        compacted[key] = _compact_value(key, value)
        if compacted[key] is not value:
            stats["compacted_keys"] += 1
        else:
            stats["skipped_keys"] += 1

    compacted["context_summary"] = stats

    full_context_ref: str | None = None
    if write_artifact and trace_id:
        full_context_ref = _write_full_context_artifact(trace_id, round_id, agent_name, context)
        if full_context_ref:
            compacted["full_context_ref"] = full_context_ref

    return compacted
