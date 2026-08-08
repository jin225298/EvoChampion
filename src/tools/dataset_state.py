"""数据集/题目两级状态机 + pass_rate 缓存。

机制层——纯数据转换和持久化，不包含策略决策。
上层（screening_entry, rollout_aggregator, data_builder）调用这些接口。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATASET_STATE_UNUSED = "unused"
DATASET_STATE_IN_USE = "in_use"
DATASET_STATE_EXHAUSTED = "exhausted"
DATASET_STATES = {DATASET_STATE_UNUSED, DATASET_STATE_IN_USE, DATASET_STATE_EXHAUSTED}
_DATASET_STATE_ALIASES = {
    "active": DATASET_STATE_IN_USE,
    "rolling_out": DATASET_STATE_IN_USE,
}

ITEM_STATE_UNUSED = "unused"
ITEM_STATE_RESERVED = "reserved"
ITEM_STATE_USED = "used"
ITEM_STATE_DEFEATED = "defeated"
ITEM_STATES = {ITEM_STATE_UNUSED, ITEM_STATE_RESERVED, ITEM_STATE_USED, ITEM_STATE_DEFEATED}


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_dataset_state(state: Any) -> str:
    value = str(state or DATASET_STATE_UNUSED)
    value = _DATASET_STATE_ALIASES.get(value, value)
    return value if value in DATASET_STATES else DATASET_STATE_UNUSED


def _normalize_item_state(state: Any) -> str:
    value = str(state or ITEM_STATE_UNUSED)
    return value if value in ITEM_STATES else ITEM_STATE_UNUSED


@dataclass
class ItemState:
    question_id: str
    state: str = ITEM_STATE_UNUSED  # unused | reserved | used | defeated
    pass_rate: float = 0.0
    difficulty: str = "unknown"  # easy | medium | hard
    rollout_count: int = 0
    defeated_round: int = -1
    defeated_probe_acc: float = 0.0
    defeat_count: int = 0
    dataset_window_offset: int | None = None
    dataset_window_limit: int | None = None
    rollout_model_key: str = ""
    rollout_config_hash: str = ""
    rollout_judge_version: str = ""
    rollout_stage: str = ""

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "state": self.state,
            "pass_rate": self.pass_rate,
            "difficulty": self.difficulty,
            "rollout_count": self.rollout_count,
            "defeated_round": self.defeated_round,
            "defeated_probe_acc": self.defeated_probe_acc,
            "defeat_count": self.defeat_count,
            "dataset_window_offset": self.dataset_window_offset,
            "dataset_window_limit": self.dataset_window_limit,
            "rollout_model_key": self.rollout_model_key,
            "rollout_config_hash": self.rollout_config_hash,
            "rollout_judge_version": self.rollout_judge_version,
            "rollout_stage": self.rollout_stage,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ItemState":
        return cls(
            question_id=str(d.get("question_id", "")),
            state=_normalize_item_state(d.get("state", ITEM_STATE_UNUSED)),
            pass_rate=float(d.get("pass_rate", 0.0)),
            difficulty=str(d.get("difficulty", "unknown")),
            rollout_count=int(d.get("rollout_count", 0)),
            defeated_round=int(d.get("defeated_round", -1)),
            defeated_probe_acc=float(d.get("defeated_probe_acc", 0.0)),
            defeat_count=int(d.get("defeat_count", 0)),
            dataset_window_offset=_optional_int(d.get("dataset_window_offset")),
            dataset_window_limit=_optional_int(d.get("dataset_window_limit")),
            rollout_model_key=str(d.get("rollout_model_key", "") or ""),
            rollout_config_hash=str(d.get("rollout_config_hash", "") or ""),
            rollout_judge_version=str(d.get("rollout_judge_version", "") or ""),
            rollout_stage=str(d.get("rollout_stage", "") or ""),
        )


@dataclass
class DatasetState:
    dataset_id: str
    state: str = DATASET_STATE_UNUSED  # unused | in_use | exhausted
    round_cached: int = -1
    probe_acc_cached: float = 0.0
    items: dict[str, ItemState] = field(default_factory=dict)
    total_loaded: int = 0
    review_fail_count: int = 0
    review_blacklisted_until: float = 0.0
    review_failure_reason: str = ""
    review_last_status: str = ""
    review_last_stage: str = ""
    review_last_ref_key: str = ""
    review_last_schema_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "state": self.state,
            "round_cached": self.round_cached,
            "probe_acc_cached": self.probe_acc_cached,
            "total_loaded": self.total_loaded,
            "review_fail_count": self.review_fail_count,
            "review_blacklisted_until": self.review_blacklisted_until,
            "review_failure_reason": self.review_failure_reason,
            "review_last_status": self.review_last_status,
            "review_last_stage": self.review_last_stage,
            "review_last_ref_key": self.review_last_ref_key,
            "review_last_schema_hash": self.review_last_schema_hash,
            "items": {qid: item.to_dict() for qid, item in self.items.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetState":
        ds = cls(
            dataset_id=str(d.get("dataset_id", "")),
            state=_normalize_dataset_state(d.get("state", DATASET_STATE_UNUSED)),
            round_cached=int(d.get("round_cached", -1)),
            probe_acc_cached=float(d.get("probe_acc_cached", 0.0)),
            total_loaded=int(d.get("total_loaded", 0)),
            review_fail_count=int(d.get("review_fail_count", 0)),
            review_blacklisted_until=float(d.get("review_blacklisted_until", 0.0)),
            review_failure_reason=str(d.get("review_failure_reason", "") or ""),
            review_last_status=str(d.get("review_last_status", "") or ""),
            review_last_stage=str(d.get("review_last_stage", "") or ""),
            review_last_ref_key=str(d.get("review_last_ref_key", "") or ""),
            review_last_schema_hash=str(d.get("review_last_schema_hash", "") or ""),
        )
        for qid, item_d in d.get("items", {}).items():
            ds.items[str(qid)] = ItemState.from_dict(item_d)
        return ds


class DatasetStateManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: dict[str, DatasetState] = {}

    def _cache_path(self, dataset_id: str) -> Path:
        safe_id = dataset_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_id}_state.json"

    def _get_or_load(self, dataset_id: str) -> DatasetState | None:
        return self.datasets.get(dataset_id) or self.load_cached(dataset_id)

    def init_dataset(self, dataset_id: str, question_ids: list[str]) -> DatasetState:
        """Register a loaded dataset/window without marking any question as used.

        Rollout is treated as pre-labeling only: newly registered questions enter
        the item state machine as ``unused``.  Re-loading another window appends
        new question ids while preserving existing ``used`` markers.
        """
        clean_ids = [str(qid) for qid in question_ids if str(qid)]
        ds = self._get_or_load(dataset_id)
        if ds is None:
            ds = DatasetState(dataset_id=dataset_id)
        for qid in clean_ids:
            ds.items.setdefault(qid, ItemState(question_id=qid))
        ds.total_loaded = len(ds.items)
        if clean_ids and ds.state != DATASET_STATE_EXHAUSTED:
            ds.state = DATASET_STATE_IN_USE
            ds.review_fail_count = 0
            ds.review_blacklisted_until = 0.0
            ds.review_failure_reason = ""
            ds.review_last_status = "ok"
            ds.review_last_stage = ""
        self.datasets[dataset_id] = ds
        self.save(dataset_id)
        return ds

    def init_dataset_from_questions(self, dataset_id: str, questions: list[dict]) -> DatasetState:
        question_ids: list[str] = []
        by_id: dict[str, dict] = {}
        for question in questions:
            if not isinstance(question, dict):
                continue
            qid = str(question.get("question_id", "") or "")
            if not qid:
                continue
            question_ids.append(qid)
            by_id[qid] = question
        ds = self.init_dataset(dataset_id, question_ids)
        for qid, question in by_id.items():
            item = ds.items.get(qid)
            if item is None:
                continue
            offset = _optional_int(question.get("dataset_window_offset"))
            limit = _optional_int(question.get("dataset_window_limit"))
            if offset is not None:
                item.dataset_window_offset = offset
            if limit is not None:
                item.dataset_window_limit = limit
        self.datasets[dataset_id] = ds
        self.save(dataset_id)
        return ds

    def mark_review_failure(
        self,
        dataset_id: str,
        reason: str,
        *,
        backoff_seconds: float,
        failures_before_backoff: int = 1,
        stage: str = "",
        status: str = "failed",
        ref_key: str = "",
        schema_hash: str = "",
        now: float | None = None,
    ) -> DatasetState:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            ds = DatasetState(dataset_id=dataset_id)
        ds.review_fail_count += 1
        ds.review_failure_reason = str(reason or "review failed")[:500]
        ds.review_last_status = str(status or "failed")[:80]
        ds.review_last_stage = str(stage or "")[:120]
        ds.review_last_ref_key = str(ref_key or "")[:240]
        ds.review_last_schema_hash = str(schema_hash or "")[:120]
        threshold = max(1, int(failures_before_backoff or 1))
        if ds.review_fail_count >= threshold and backoff_seconds > 0:
            import time

            current_time = time.time() if now is None else float(now)
            ds.review_blacklisted_until = max(
                float(ds.review_blacklisted_until or 0.0),
                current_time + float(backoff_seconds),
            )
        self.datasets[dataset_id] = ds
        self.save(dataset_id)
        return ds

    def clear_review_failure(self, dataset_id: str) -> None:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return
        ds.review_fail_count = 0
        ds.review_blacklisted_until = 0.0
        ds.review_failure_reason = ""
        ds.review_last_status = "ok"
        ds.review_last_stage = ""
        self.datasets[dataset_id] = ds
        self.save(dataset_id)

    def review_blacklist_reason(self, dataset_id: str, *, now: float | None = None) -> str | None:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return None
        until = float(ds.review_blacklisted_until or 0.0)
        if until <= 0:
            return None
        import time

        current_time = time.time() if now is None else float(now)
        if until <= current_time:
            ds.review_blacklisted_until = 0.0
            self.datasets[dataset_id] = ds
            self.save(dataset_id)
            return None
        remaining = max(0.0, until - current_time)
        reason = ds.review_failure_reason or "recent review failure"
        return f"review backoff active for {remaining:.0f}s after {ds.review_fail_count} failure(s): {reason}"

    def is_review_blacklisted(self, dataset_id: str, *, now: float | None = None) -> bool:
        return self.review_blacklist_reason(dataset_id, now=now) is not None

    def load_cached(self, dataset_id: str) -> DatasetState | None:
        path = self._cache_path(dataset_id)
        if not path.exists():
            return None
        try:
            ds = DatasetState.from_dict(json.loads(path.read_text("utf-8")))
            self.datasets[dataset_id] = ds
            return ds
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def load_all_cached(self) -> list[str]:
        dataset_ids: list[str] = []
        for path in sorted(self.cache_dir.glob("*_state.json")):
            try:
                raw = json.loads(path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(raw, dict):
                continue
            ds = DatasetState.from_dict(raw)
            if not ds.dataset_id:
                continue
            self.datasets[ds.dataset_id] = ds
            dataset_ids.append(ds.dataset_id)
        return dataset_ids

    def save(self, dataset_id: str) -> None:
        ds = self.datasets.get(dataset_id)
        if ds is None:
            return
        path = self._cache_path(dataset_id)
        path.write_text(json.dumps(ds.to_dict(), ensure_ascii=False, indent=2), "utf-8")

    def update_pass_rates(
        self,
        dataset_id: str,
        scored_items: list[dict],
        round_id: int,
        probe_acc: float,
    ) -> DatasetState:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            ds = self.init_dataset(
                dataset_id,
                [str(item.get("question_id", "")) for item in scored_items],
            )
        for item in scored_items:
            qid = str(item.get("question_id", ""))
            if not qid:
                continue
            ds.items.setdefault(qid, ItemState(question_id=qid))
            pass_rate = float(item.get("pass_rate", 0.0))
            difficulty = str(item.get("dynamic_difficulty", "unknown"))
            ds.items[qid].pass_rate = pass_rate
            ds.items[qid].difficulty = difficulty
            ds.items[qid].rollout_count = int(item.get("rollout_count", 0))
            ds.items[qid].rollout_model_key = str(item.get("rollout_model_key", "") or "")
            ds.items[qid].rollout_config_hash = str(item.get("rollout_config_hash", "") or "")
            ds.items[qid].rollout_judge_version = str(item.get("rollout_judge_version", "") or "")
            ds.items[qid].rollout_stage = str(item.get("rollout_stage", "") or "")
            offset = _optional_int(item.get("dataset_window_offset"))
            limit = _optional_int(item.get("dataset_window_limit"))
            if offset is not None:
                ds.items[qid].dataset_window_offset = offset
            if limit is not None:
                ds.items[qid].dataset_window_limit = limit
        if ds.state != DATASET_STATE_EXHAUSTED:
            ds.state = DATASET_STATE_IN_USE
        ds.round_cached = round_id
        ds.probe_acc_cached = probe_acc
        ds.total_loaded = len(ds.items)
        self.datasets[dataset_id] = ds
        self.save(dataset_id)
        return ds

    def mark_used(self, dataset_id: str, question_ids: list[str]) -> None:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return
        for qid in question_ids:
            if qid in ds.items:
                ds.items[qid].state = ITEM_STATE_USED
        self.datasets[dataset_id] = ds
        self.save(dataset_id)

    def mark_reserved(self, dataset_id: str, question_ids: list[str]) -> None:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return
        for qid in question_ids:
            if qid in ds.items and ds.items[qid].state == ITEM_STATE_UNUSED:
                ds.items[qid].state = ITEM_STATE_RESERVED
        self.datasets[dataset_id] = ds
        self.save(dataset_id)

    def unmark_reserved(self, dataset_id: str, question_ids: list[str] | None = None) -> None:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return
        allowed = {str(qid) for qid in question_ids} if question_ids is not None else None
        for qid, item in ds.items.items():
            if item.state == ITEM_STATE_RESERVED and (allowed is None or qid in allowed):
                item.state = ITEM_STATE_UNUSED
        self.datasets[dataset_id] = ds
        self.save(dataset_id)

    def mark_reserved_used(self, dataset_id: str, question_ids: list[str] | None = None) -> None:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return
        allowed = {str(qid) for qid in question_ids} if question_ids is not None else None
        for qid, item in ds.items.items():
            if item.state == ITEM_STATE_RESERVED and (allowed is None or qid in allowed):
                item.state = ITEM_STATE_USED
        self.datasets[dataset_id] = ds
        self.save(dataset_id)

    def mark_reserved_defeated(
        self,
        dataset_id: str,
        question_ids: list[str] | None = None,
        *,
        round_id: int = -1,
        probe_acc: float = 0.0,
    ) -> None:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return
        allowed = {str(qid) for qid in question_ids} if question_ids is not None else None
        for qid, item in ds.items.items():
            if item.state == ITEM_STATE_RESERVED and (allowed is None or qid in allowed):
                item.state = ITEM_STATE_DEFEATED
                item.defeated_round = int(round_id)
                item.defeated_probe_acc = float(probe_acc or 0.0)
                item.defeat_count = int(item.defeat_count or 0) + 1
        self.datasets[dataset_id] = ds
        self.save(dataset_id)

    def mark_exhausted_if_below(self, dataset_id: str, min_remaining: int) -> bool:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return False
        remaining = self.count_unused(dataset_id)
        if remaining < max(1, int(min_remaining)):
            ds.state = DATASET_STATE_EXHAUSTED
            self.datasets[dataset_id] = ds
            self.save(dataset_id)
            return True
        return False

    def mark_exhausted(self, dataset_id: str) -> DatasetState:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            ds = DatasetState(dataset_id=dataset_id)
        ds.state = DATASET_STATE_EXHAUSTED
        self.datasets[dataset_id] = ds
        self.save(dataset_id)
        return ds

    def count_refresh_candidates(self, dataset_id: str, pass_rate_threshold: float) -> int:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return 0
        return sum(
            1 for item in ds.items.values()
            if item.state == ITEM_STATE_UNUSED
            and item.pass_rate < pass_rate_threshold
            and item.rollout_count > 0
        )

    def mark_deferred(self, dataset_id: str, pass_rate_threshold: float) -> int:
        """Backward-compatible alias for the old deferred API.

        Question state is now intentionally only ``unused``/``used``.  Callers
        that still ask for deferred marking receive the count of low-pass-rate
        unused items, but no third item state is persisted.
        """
        return self.count_refresh_candidates(dataset_id, pass_rate_threshold)

    def reset_unused_rollout_labels(self, dataset_id: str) -> int:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return 0
        count = 0
        for item in ds.items.values():
            if item.state == ITEM_STATE_UNUSED and item.rollout_count > 0:
                item.pass_rate = 0.0
                item.difficulty = "unknown"
                item.rollout_count = 0
                count += 1
        if count and ds.state == DATASET_STATE_EXHAUSTED:
            ds.state = DATASET_STATE_IN_USE
        self.save(dataset_id)
        return count

    def revive_deferred(self, dataset_id: str) -> int:
        """Backward-compatible alias for refreshing unused prelabels."""
        return self.reset_unused_rollout_labels(dataset_id)

    def needs_rerollout(
        self,
        dataset_id: str,
        current_round: int,
        current_probe_acc: float,
        threshold: float,
        refresh_interval_rounds: int | None = None,
    ) -> bool:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return True
        if ds.state == DATASET_STATE_UNUSED:
            return True
        improvement = current_probe_acc - ds.probe_acc_cached
        if improvement >= threshold:
            return True
        if refresh_interval_rounds is not None and refresh_interval_rounds > 0:
            return current_round - ds.round_cached >= refresh_interval_rounds
        return False

    def get_available_by_difficulty(
        self,
        dataset_id: str,
        include_reserved: bool = True,
    ) -> dict[str, list[str]]:
        ds = self._get_or_load(dataset_id)
        result: dict[str, list[str]] = {"easy": [], "medium": [], "hard": [], "unknown": []}
        if ds is None:
            return result
        allowed_states = {ITEM_STATE_UNUSED}
        if include_reserved:
            allowed_states.add(ITEM_STATE_RESERVED)
        for item in ds.items.values():
            if item.state in allowed_states and item.rollout_count > 0:
                result.setdefault(item.difficulty, []).append(item.question_id)
        return result

    def get_unused_by_difficulty(self, dataset_id: str) -> dict[str, list[str]]:
        return self.get_available_by_difficulty(dataset_id, include_reserved=False)

    def get_selectable_cached_unused_by_difficulty(self, dataset_id: str) -> dict[str, list[str]]:
        ds = self._get_or_load(dataset_id)
        result: dict[str, list[str]] = {"easy": [], "medium": [], "hard": [], "unknown": []}
        if ds is None:
            return result
        for item in ds.items.values():
            if item.state == ITEM_STATE_UNUSED and item.rollout_count > 0:
                result.setdefault(str(item.difficulty or "unknown"), []).append(item.question_id)
        return result

    def revive_defeated_if_improved(
        self,
        dataset_id: str,
        current_probe_acc: float,
        improvement_threshold: float,
    ) -> int:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return 0
        revived = 0
        threshold = float(improvement_threshold or 0.0)
        probe_acc = float(current_probe_acc or 0.0)
        for item in ds.items.values():
            if item.state != ITEM_STATE_DEFEATED:
                continue
            if probe_acc - float(item.defeated_probe_acc or 0.0) < threshold:
                continue
            item.state = ITEM_STATE_UNUSED
            item.pass_rate = 0.0
            item.difficulty = "unknown"
            item.rollout_count = 0
            revived += 1
        if revived and ds.state == DATASET_STATE_EXHAUSTED:
            ds.state = DATASET_STATE_IN_USE
        self.datasets[dataset_id] = ds
        self.save(dataset_id)
        return revived

    def count_unused(self, dataset_id: str) -> int:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return 0
        return sum(1 for item in ds.items.values() if item.state == ITEM_STATE_UNUSED)

    def all_exhausted(self, dataset_id: str) -> bool:
        ds = self._get_or_load(dataset_id)
        if ds is None:
            return False
        return ds.state == DATASET_STATE_EXHAUSTED or bool(ds.items) and all(
            item.state != ITEM_STATE_UNUSED for item in ds.items.values()
        )
