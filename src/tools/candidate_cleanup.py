"""Candidate model retention helpers.

The cleanup policy is intentionally conservative:
- disabled by default;
- scoped to the current trace_id only;
- never deletes the current champion;
- only removes directories matching candidate_{trace_id}_roundN.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.settings import (
    CANDIDATE_MODEL_DIR,
    CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED,
)

NON_PROMOTED_DECISIONS = {"rollback", "prune", "provisional_promote", "keep_branch"}
KNOWN_DECISIONS = {"promote", *NON_PROMOTED_DECISIONS}


def cleanup_candidate_models(
    state: Mapping[str, Any],
    *,
    keep_recent_non_promoted: int | None = None,
    candidate_model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Delete stale candidate model directories for the current trace.

    Args:
        state: Persisted evolution state after strategy inspection.
        keep_recent_non_promoted: Number of recent rollback/prune candidates to keep.
            ``None`` uses ``CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED``. Values
            <= 0 disable cleanup and keep all candidates.
        candidate_model_dir: Override root directory for tests.

    Returns:
        A small summary dict useful for tests and log inspection.
    """
    retention = (
        CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED
        if keep_recent_non_promoted is None
        else int(keep_recent_non_promoted)
    )
    if retention <= 0:
        return {"enabled": False, "deleted": [], "kept": [], "skipped": []}

    trace_id = str(state.get("trace_id") or "").strip()
    if not trace_id:
        return {"enabled": True, "deleted": [], "kept": [], "skipped": ["missing_trace_id"]}

    root = Path(candidate_model_dir or CANDIDATE_MODEL_DIR)
    if not root.exists():
        return {"enabled": True, "deleted": [], "kept": [], "skipped": [str(root)]}
    root_resolved = root.resolve()

    keep_paths = _candidate_keep_paths(
        state,
        root=root_resolved,
        trace_id=trace_id,
        keep_recent_non_promoted=retention,
    )
    kept = sorted(str(path) for path in keep_paths)
    deleted: list[str] = []
    skipped: list[str] = []

    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_dir():
            continue
        round_id = _candidate_round_id(path, trace_id)
        if round_id is None:
            continue
        resolved = path.resolve()
        if not _is_relative_to(resolved, root_resolved):
            skipped.append(str(path))
            continue
        if resolved in keep_paths:
            continue
        try:
            print(f"[candidate_cleanup] Deleting stale candidate: {path}")
            shutil.rmtree(path)
            deleted.append(str(path))
        except OSError as exc:
            skipped.append(str(path))
            print(f"[candidate_cleanup] Could not delete {path}: {exc}")

    if deleted:
        print(
            f"[candidate_cleanup] Deleted {len(deleted)} stale candidates; "
            f"kept current champion plus {retention} recent rollback/prune candidates"
        )
    return {"enabled": True, "deleted": deleted, "kept": kept, "skipped": skipped}


def _candidate_keep_paths(
    state: Mapping[str, Any],
    *,
    root: Path,
    trace_id: str,
    keep_recent_non_promoted: int,
) -> set[Path]:
    keep_paths: set[Path] = set()

    champion_path = _safe_candidate_path(state.get("champion_model_path"), root, trace_id)
    if champion_path is not None:
        keep_paths.add(champion_path)

    for round_id in _recent_non_promoted_rounds(state, keep_recent_non_promoted):
        keep_paths.add((root / f"candidate_{trace_id}_round{round_id}").resolve())

    return keep_paths


def _recent_non_promoted_rounds(state: Mapping[str, Any], limit: int) -> list[int]:
    rounds: set[int] = set()
    raw_edges = state.get("search_dag_edges", [])
    edges = raw_edges if isinstance(raw_edges, list) else []
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        decision = _edge_decision(edge)
        if decision not in NON_PROMOTED_DECISIONS:
            continue
        round_id = _edge_round_id(edge)
        if round_id is not None:
            rounds.add(round_id)

    current_decision = str(state.get("last_inspection_decision") or "")
    if current_decision in NON_PROMOTED_DECISIONS:
        current_round = _coerce_round_id(state.get("round_id"))
        if current_round is not None:
            rounds.add(current_round)

    return sorted(rounds, reverse=True)[:limit]


def _safe_candidate_path(value: object, root: Path, trace_id: str) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.name:
        return None
    if _candidate_round_id(path, trace_id) is None:
        return None
    resolved = path.resolve()
    if not _is_relative_to(resolved, root):
        return None
    return resolved


def _candidate_round_id(path: Path, trace_id: str) -> int | None:
    pattern = re.compile(rf"^candidate_{re.escape(trace_id)}_round(\d+)$")
    match = pattern.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def _edge_decision(edge: Mapping[str, Any]) -> str:
    decision = str(edge.get("decision") or "")
    if decision in KNOWN_DECISIONS:
        return decision
    metadata = edge.get("action_metadata")
    if isinstance(metadata, Mapping):
        decision = str(metadata.get("decision") or "")
        if decision in KNOWN_DECISIONS:
            return decision
    summary = str(edge.get("action_summary") or "")
    for known in KNOWN_DECISIONS:
        if known in summary:
            return known
    return ""


def _edge_round_id(edge: Mapping[str, Any]) -> int | None:
    round_id = _coerce_round_id(edge.get("round_id"))
    if round_id is not None:
        return round_id
    summary = str(edge.get("action_summary") or "")
    match = re.search(r"round_(\d+)", summary)
    if match:
        return int(match.group(1))
    return None


def _coerce_round_id(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
