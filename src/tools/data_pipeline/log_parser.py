"""训练日志解析工具 — 纯函数，不参与决策。

ResourceAdaptAgent 调用此工具读取训练日志，获取 OOM 信息后自行决策。
"""

import re
import json
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _parse_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _trainer_state_path_from_artifacts(
    trainer_log_jsonl_path: str,
    training_loss_jsonl_path: str,
    explicit_path: str,
) -> str:
    if explicit_path:
        return explicit_path
    for raw_path in (trainer_log_jsonl_path, training_loss_jsonl_path):
        if raw_path:
            candidate = Path(raw_path).with_name("trainer_state.json")
            if candidate.exists():
                return str(candidate)
    return ""


def _train_results_path_from_artifacts(
    trainer_log_jsonl_path: str,
    training_loss_jsonl_path: str,
    explicit_path: str,
) -> str:
    if explicit_path:
        return explicit_path
    for raw_path in (trainer_log_jsonl_path, training_loss_jsonl_path):
        if raw_path:
            candidate = Path(raw_path).with_name("train_results.json")
            if candidate.exists():
                return str(candidate)
    return ""


def _compact_reason(line: str, kind: str, exit_code: int | None = None) -> str:
    normalized = " ".join(line.strip().split())
    if kind == "cuda_oom":
        alloc_match = re.search(r"Tried to allocate ([^.;]+)", normalized, re.IGNORECASE)
        if alloc_match:
            return f"CUDA OOM while allocating {alloc_match.group(1)}"
        return "CUDA OOM"
    if kind == "timeout":
        return "training timed out" if not normalized else normalized
    if kind == "not_found":
        return normalized or "required artifact not found"
    if kind == "no_loadable_model":
        return normalized or "no loadable model found"
    if kind == "exit_code_nonzero":
        return f"Training exited with return code {exit_code}" if exit_code is not None else "training exited with a non-zero return code"
    return normalized or "training failed"


def _infer_training_outcome(
    *,
    log_text: str,
    exit_code: int | None,
    available: bool,
) -> dict[str, Any]:
    patterns: list[tuple[str, re.Pattern[str]]] = [
        (
            "cuda_oom",
            re.compile(
                r"torch\.OutOfMemoryError:.*CUDA out of memory|CUDA out of memory|out of memory",
                re.IGNORECASE,
            ),
        ),
        (
            "timeout",
            re.compile(r"\btimed out\b|\btimeout\b|time limit exceeded", re.IGNORECASE),
        ),
        (
            "no_loadable_model",
            re.compile(
                r"no loadable model|no model can be loaded|could not load .*model|cannot load .*model",
                re.IGNORECASE,
            ),
        ),
        (
            "not_found",
            re.compile(
                r"FileNotFoundError|No such file or directory|(?:checkpoint|model|artifact).{0,40}\bnot found\b",
                re.IGNORECASE,
            ),
        ),
    ]

    matched_kind = None
    matched_line = ""
    for line in log_text.splitlines():
        for kind, pattern in patterns:
            if pattern.search(line):
                matched_kind = kind
                matched_line = line
                break
        if matched_kind:
            break

    if exit_code is not None and exit_code != 0:
        kind = matched_kind or "exit_code_nonzero"
        reason = _compact_reason(matched_line or log_text, kind, exit_code=exit_code)
        return {
            "status": "failed",
            "failure_kind": kind,
            "failure_reason": reason,
        }

    if matched_kind:
        return {
            "status": "failed",
            "failure_kind": matched_kind,
            "failure_reason": _compact_reason(matched_line or log_text, matched_kind, exit_code=exit_code),
        }

    if available:
        return {
            "status": "success",
            "failure_kind": None,
            "failure_reason": None,
        }

    return {
        "status": "unknown",
        "failure_kind": None,
        "failure_reason": None,
    }


def _numeric_series_summary(items: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    values: list[tuple[int | None, float]] = []
    for item in items:
        value = _to_float(item.get(value_key))
        if value is None:
            continue
        values.append((_to_int(item.get("step")), value))
    if not values:
        return {
            "first": None,
            "last": None,
            "min": None,
            "min_at_step": None,
            "max": None,
            "max_at_step": None,
            "trend": "unknown",
            "slope": None,
            "std": None,
            "count": 0,
        }

    first_step, first = values[0]
    last_step, last = values[-1]
    min_step, min_value = min(values, key=lambda pair: pair[1])
    max_step, max_value = max(values, key=lambda pair: pair[1])
    mean = sum(value for _, value in values) / len(values)
    variance = sum((value - mean) ** 2 for _, value in values) / len(values)
    slope = None
    if first_step is not None and last_step is not None and last_step != first_step:
        slope = (last - first) / (last_step - first_step)
    elif len(values) > 1:
        slope = (last - first) / (len(values) - 1)

    eps = max(abs(first) * 0.05, 1e-12)
    if last < first - eps:
        trend = "falling"
    elif last > first + eps:
        trend = "rising"
    else:
        trend = "flat"

    return {
        "first": first,
        "last": last,
        "min": min_value,
        "min_at_step": min_step,
        "max": max_value,
        "max_at_step": max_step,
        "trend": trend,
        "slope": slope,
        "std": variance ** 0.5,
        "count": len(values),
    }


def _loss_spike_count(items: list[dict[str, Any]]) -> int:
    values = [_to_float(item.get("loss")) for item in items]
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return 0
    spikes = 0
    for previous, current in zip(clean, clean[1:]):
        if current > previous * 1.20:
            spikes += 1
    return spikes


def _learning_rate_summary(learning_rates: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _numeric_series_summary(learning_rates, "learning_rate")
    values = [_to_float(item.get("learning_rate")) for item in learning_rates]
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        shape = "unknown" if not clean else "constant"
    else:
        rising = any(curr > prev for prev, curr in zip(clean, clean[1:]))
        falling = any(curr < prev for prev, curr in zip(clean, clean[1:]))
        if rising and falling:
            shape = "warmup_then_decay"
        elif falling:
            shape = "decaying"
        elif rising:
            shape = "warming_up"
        else:
            shape = "constant"
    return {
        "first": summary["first"],
        "last": summary["last"],
        "min": summary["min"],
        "max": summary["max"],
        "trend": summary["trend"],
        "schedule_shape": shape,
        "count": summary["count"],
    }


def diagnose_loss_phase(
    loss_history: list[dict[str, Any]],
    eval_loss_history: list[dict[str, Any]],
    learning_rates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify the training trajectory into a compact phase diagnosis."""
    train = _numeric_series_summary(loss_history, "loss")
    eval_loss = _numeric_series_summary(eval_loss_history, "eval_loss")
    lr = _learning_rate_summary(learning_rates)
    train_count = int(train.get("count") or 0)
    eval_count = int(eval_loss.get("count") or 0)
    if train_count < 2:
        return {
            "phase": "insufficient_data",
            "scheduler_diagnosis": "unknown",
            "plateau_detected": False,
            "plateau_start_step": None,
            "recommended_knobs": [],
        }

    train_trend = str(train.get("trend") or "unknown")
    eval_trend = str(eval_loss.get("trend") or "unknown")
    lr_shape = str(lr.get("schedule_shape") or "unknown")
    train_std = _to_float(train.get("std")) or 0.0
    train_first = _to_float(train.get("first")) or 0.0
    train_last = _to_float(train.get("last")) or 0.0
    eval_last = _to_float(eval_loss.get("last"))
    gap = eval_last - train_last if eval_last is not None else None
    reduction_ratio = (train_first - train_last) / max(abs(train_first), 1e-12) if train_first else 0.0

    plateau_detected = train_trend == "flat" or (eval_count >= 2 and eval_trend == "flat")
    plateau_start_step = None
    if plateau_detected:
        source = eval_loss_history if eval_count >= 2 else loss_history
        plateau_start_step = _to_int(source[-2].get("step")) if len(source) >= 2 else None

    if train_trend == "rising" or train_std > max(abs(train_first) * 0.35, 0.35):
        phase = "unstable"
        scheduler_diagnosis = "lr_may_be_too_high_or_schedule_unstable"
        knobs = ["lower_learning_rate", "increase_warmup", "prefer_conservative_update"]
    elif gap is not None and gap > 0.10 and train_trend == "falling":
        phase = "overfit_or_distribution_gap"
        scheduler_diagnosis = "train_eval_gap_widening"
        knobs = ["increase_replay", "reduce_epochs", "consider_replace_dataset"]
    elif plateau_detected:
        phase = "plateau"
        scheduler_diagnosis = "eval_or_train_loss_plateau"
        knobs = ["adjust_scheduler", "shift_or_replace_data_if_evaluator_flat"]
    elif reduction_ratio < 0.20:
        phase = "underfit"
        scheduler_diagnosis = "insufficient_loss_reduction"
        knobs = ["increase_steps_or_epochs", "check_learning_rate_floor"]
    elif lr_shape == "warmup_then_decay":
        phase = "warmup_then_decay"
        scheduler_diagnosis = "normal_warmup_decay"
        knobs = ["keep_schedule", "use_evaluator_for_next_decision"]
    else:
        phase = "converging" if train_trend == "falling" else "stable"
        scheduler_diagnosis = "normal_or_unclassified"
        knobs = ["use_evaluator_for_next_decision"]

    return {
        "phase": phase,
        "scheduler_diagnosis": scheduler_diagnosis,
        "plateau_detected": plateau_detected,
        "plateau_start_step": plateau_start_step,
        "recommended_knobs": knobs,
    }


def _build_training_summary(
    *,
    trainer_log_jsonl_path: str,
    training_loss_jsonl_path: str,
    trainer_state_path: str,
    all_results_path: str,
    train_results_path: str,
    loss_history: list[dict[str, Any]],
    eval_loss_history: list[dict[str, Any]],
    learning_rates: list[dict[str, Any]],
    epochs: list[dict[str, Any]],
    throughput: list[dict[str, Any]],
    rows_found: bool,
    log_text: str = "",
    exit_code: int | None = None,
) -> dict[str, Any]:
    train_stats = _numeric_series_summary(loss_history, "loss")
    eval_stats = _numeric_series_summary(eval_loss_history, "eval_loss")
    lr_stats = _learning_rate_summary(learning_rates)
    all_results = _parse_json_file(Path(all_results_path)) if all_results_path else {}
    train_results = _parse_json_file(Path(train_results_path)) if train_results_path else {}
    trainer_state = _parse_json_file(Path(trainer_state_path)) if trainer_state_path else {}

    all_eval_loss = _to_float(all_results.get("eval_loss"))
    if eval_stats["last"] is None and all_eval_loss is not None:
        eval_stats = {**eval_stats, "first": all_eval_loss, "last": all_eval_loss, "min": all_eval_loss, "count": 1}

    train_first = _to_float(train_stats.get("first"))
    train_last = _to_float(train_stats.get("last"))
    reduction_abs = train_first - train_last if train_first is not None and train_last is not None else None
    reduction_ratio = reduction_abs / max(abs(train_first), 1e-12) if reduction_abs is not None and train_first else None
    eval_last = _to_float(eval_stats.get("last"))
    eval_train_gap_last = eval_last - train_last if eval_last is not None and train_last is not None else None
    epoch_values = [_to_float(item.get("epoch")) for item in epochs]
    epoch_clean = [value for value in epoch_values if value is not None]
    step_values = [_to_int(item.get("step")) for item in loss_history + eval_loss_history + learning_rates + epochs]
    step_clean = [value for value in step_values if value is not None]
    throughput_values = [_to_float(item.get("throughput")) for item in throughput]
    throughput_clean = [value for value in throughput_values if value is not None]
    phase = diagnose_loss_phase(loss_history, eval_loss_history, learning_rates)

    loss_diagnosis = "missing_loss"
    evidence: list[str] = []
    if train_stats.get("count"):
        train_trend = str(train_stats.get("trend") or "unknown")
        if train_trend == "falling":
            evidence.append("train_loss_falling")
        if eval_stats.get("trend") == "falling":
            evidence.append("eval_loss_falling")
        if eval_train_gap_last is not None and eval_train_gap_last > 0.10:
            evidence.append("eval_train_gap_positive")
        if phase["phase"] == "unstable":
            loss_diagnosis = "unstable"
        elif phase["phase"] == "underfit":
            loss_diagnosis = "underfit"
        elif phase["phase"] == "overfit_or_distribution_gap":
            loss_diagnosis = "overfit"
        elif phase["phase"] == "plateau":
            loss_diagnosis = "learning_saturated"
        else:
            loss_diagnosis = "normal_training"

    summary = {
        "available": bool(rows_found or all_results or train_results or trainer_state),
        "artifact_paths": {
            "trainer_log_jsonl_path": trainer_log_jsonl_path,
            "training_loss_jsonl_path": training_loss_jsonl_path,
            "trainer_state_path": trainer_state_path,
            "all_results_path": all_results_path,
            "train_results_path": train_results_path,
        },
        "train_loss": {
            "first": train_stats["first"],
            "last": train_stats["last"],
            "min": train_stats["min"],
            "min_at_step": train_stats["min_at_step"],
            "reduction_abs": reduction_abs,
            "reduction_ratio": reduction_ratio,
            "trend": train_stats["trend"],
            "slope": train_stats["slope"],
            "std": train_stats["std"],
            "spike_count": _loss_spike_count(loss_history),
        },
        "eval_loss": {
            "first": eval_stats["first"],
            "last": eval_stats["last"],
            "min": eval_stats["min"],
            "min_at_step": eval_stats["min_at_step"],
            "eval_count": eval_stats["count"],
            "trend": eval_stats["trend"],
            "eval_train_gap_last": eval_train_gap_last,
        },
        "training_process": {
            "global_steps": max(step_clean) if step_clean else _to_int(trainer_state.get("global_step")),
            "epochs_completed": max(epoch_clean) if epoch_clean else _to_float(trainer_state.get("epoch")),
            "learning_rate_first": lr_stats["first"],
            "learning_rate_last": lr_stats["last"],
            "learning_rate_min": lr_stats["min"],
            "learning_rate_max": lr_stats["max"],
            "learning_rate_trend": lr_stats["trend"],
            "learning_rate_schedule_shape": lr_stats["schedule_shape"],
            "train_runtime": _to_float(train_results.get("train_runtime", all_results.get("train_runtime"))),
            "train_samples_per_second": _to_float(train_results.get("train_samples_per_second", all_results.get("train_samples_per_second"))) or (throughput_clean[-1] if throughput_clean else None),
            "lr_scheduler_type": None,
            "warmup_ratio": None,
            "warmup_steps": None,
            "best_metric": _to_float(trainer_state.get("best_metric")),
            "best_global_step": _to_int(trainer_state.get("best_global_step")),
            "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
        },
        "loss_phase": phase,
        "diagnosis": {
            "loss_diagnosis": loss_diagnosis,
            "confidence": 0.0 if loss_diagnosis == "missing_loss" else 0.7,
            "evidence": evidence,
        },
    }
    summary.update(
        _infer_training_outcome(
            log_text=log_text,
            exit_code=exit_code,
            available=bool(rows_found or all_results or train_results or trainer_state),
        )
    )
    return summary


def parse_structured_training_logs(
    trainer_log_jsonl_path: str = "",
    training_loss_jsonl_path: str = "",
    all_results_path: str = "",
    trainer_state_path: str = "",
    train_results_path: str = "",
    log_text: str = "",
    exit_code: int | None = None,
) -> dict[str, Any]:
    """Parse LLaMA-Factory JSONL trainer artifacts when present."""
    trainer_state_path = _trainer_state_path_from_artifacts(
        trainer_log_jsonl_path,
        training_loss_jsonl_path,
        trainer_state_path,
    )
    train_results_path = _train_results_path_from_artifacts(
        trainer_log_jsonl_path,
        training_loss_jsonl_path,
        train_results_path,
    )
    rows: list[dict[str, Any]] = []
    if trainer_log_jsonl_path:
        trainer_log_path = Path(trainer_log_jsonl_path)
        if trainer_log_path.exists():
            rows.extend(_parse_jsonl(trainer_log_path))
    if training_loss_jsonl_path:
        training_loss_path = Path(training_loss_jsonl_path)
        if training_loss_path.exists():
            rows.extend(_parse_jsonl(training_loss_path))
    trainer_state = _parse_json_file(Path(trainer_state_path)) if trainer_state_path else {}
    log_history = trainer_state.get("log_history")
    if isinstance(log_history, list):
        rows.extend(row for row in log_history if isinstance(row, dict))

    loss_history = []
    eval_loss_history = []
    learning_rates = []
    epochs = []
    throughput = []
    for row in rows:
        step = row.get("current_steps", row.get("step", row.get("global_step")))
        if row.get("loss") is not None:
            loss_history.append({"step": step, "loss": row.get("loss")})
        if row.get("eval_loss") is not None:
            eval_loss_history.append({"step": step, "eval_loss": row.get("eval_loss")})
        if row.get("learning_rate") is not None:
            learning_rates.append({"step": step, "learning_rate": row.get("learning_rate")})
        if row.get("epoch") is not None:
            epochs.append({"step": step, "epoch": row.get("epoch")})
        throughput_value = row.get("train_samples_per_second", row.get("train_tokens_per_second", row.get("total_tokens_per_second")))
        if throughput_value is not None:
            throughput.append({"step": step, "throughput": throughput_value})

    training_summary = _build_training_summary(
        trainer_log_jsonl_path=trainer_log_jsonl_path,
        training_loss_jsonl_path=training_loss_jsonl_path,
        trainer_state_path=trainer_state_path,
        all_results_path=all_results_path,
        train_results_path=train_results_path,
        loss_history=loss_history,
        eval_loss_history=eval_loss_history,
        learning_rates=learning_rates,
        epochs=epochs,
        throughput=throughput,
        rows_found=bool(rows),
        log_text=log_text,
        exit_code=exit_code,
    )

    return {
        "structured_log_found": bool(rows),
        "loss_history": loss_history,
        "eval_loss_history": eval_loss_history,
        "learning_rates": learning_rates,
        "epochs": epochs,
        "throughput": throughput,
        "training_summary": training_summary,
    }


def _structured_artifact_paths_for_log(path: Path) -> tuple[str, str]:
    nearby_trainer_log = path.with_name("trainer_log.jsonl")
    nearby_training_loss = path.with_name("training_loss.jsonl")
    if nearby_trainer_log.exists() or nearby_training_loss.exists():
        return str(nearby_trainer_log), str(nearby_training_loss)
    match = re.fullmatch(r"train_log_(.+)_round(\d+)\.txt", path.name)
    if not match:
        return str(nearby_trainer_log), str(nearby_training_loss)
    candidate_dir = path.parent / f"candidate_{match.group(1)}_round{match.group(2)}"
    return str(candidate_dir / "trainer_log.jsonl"), str(candidate_dir / "training_loss.jsonl")


def parse_training_log(log_path: str) -> dict[str, Any]:
    """扫描训练日志，提取关键事件供 Agent 决策。

    Args:
        log_path: 训练日志文件路径（train_log_*.txt）。

    Returns:
        {
            "oom_detected": bool,
            "oom_lines": ["CUDA out of memory...", ...],
            "peak_gpu_mb": float | None,
            "available_gpu_mb": float | None,
            "current_batch_size": int | None,
            "current_gradient_accumulation": int | None,
            "exit_code": int | None,
            "warnings": ["..."],
        }
    """
    path = Path(log_path)
    trainer_log_jsonl_path, training_loss_jsonl_path = _structured_artifact_paths_for_log(path)

    if not path.exists():
        structured = parse_structured_training_logs(
            trainer_log_jsonl_path,
            training_loss_jsonl_path,
            log_text="",
            exit_code=None,
        )
        structured["training_summary"] = {
            **structured["training_summary"],
            "status": "failed",
            "failure_kind": "not_found",
            "failure_reason": f"log file not found: {log_path}",
        }
        return {
            "oom_detected": False,
            "oom_lines": [],
            "peak_gpu_mb": None,
            "available_gpu_mb": None,
            "current_batch_size": None,
            "current_gradient_accumulation": None,
            "exit_code": None,
            "warnings": [f"log file not found: {log_path}"],
            **structured,
        }

    text = path.read_text(encoding="utf-8", errors="replace")

    # ── OOM 检测 ──
    oom_pattern = re.compile(
        r"out of memory|OOM|CUDA error|RuntimeError.*memory|cannot allocate",
        re.IGNORECASE,
    )
    oom_lines = [
        line.strip()
        for line in text.split("\n")
        if oom_pattern.search(line)
    ]

    # ── GPU 显存信息 ──
    peak_match = re.search(r"(\d+\.?\d*)\s*(GB|MB).*peak", text, re.IGNORECASE)
    avail_match = re.search(r"(\d+\.?\d*)\s*(GB|MB).*available", text, re.IGNORECASE)

    peak_gpu_mb = None
    if peak_match:
        val = float(peak_match.group(1))
        peak_gpu_mb = val * 1024 if peak_match.group(2).upper() == "GB" else val

    available_gpu_mb = None
    if avail_match:
        val = float(avail_match.group(1))
        available_gpu_mb = val * 1024 if avail_match.group(2).upper() == "GB" else val

    # ── 当前 batch size ──
    bs_match = re.search(r"per_device_train_batch_size[=:]\s*(\d+)", text)
    ga_match = re.search(r"gradient_accumulation_steps[=:]\s*(\d+)", text)

    # ── 退出码 ──
    exit_match = re.search(r"Training exited with return code:\s*(-?\d+)", text)

    # ── 警告 ──
    warn_pattern = re.compile(r"(WARNING|warning|UserWarning|DeprecationWarning)", re.IGNORECASE)
    warnings = [
        line.strip()
        for line in text.split("\n")
        if warn_pattern.search(line)
    ][:20]

    structured = parse_structured_training_logs(
        trainer_log_jsonl_path,
        training_loss_jsonl_path,
        log_text=text,
        exit_code=int(exit_match.group(1)) if exit_match else None,
    )

    return {
        "oom_detected": len(oom_lines) > 0,
        "oom_lines": oom_lines[:10],
        "peak_gpu_mb": peak_gpu_mb,
        "available_gpu_mb": available_gpu_mb,
        "current_batch_size": int(bs_match.group(1)) if bs_match else None,
        "current_gradient_accumulation": int(ga_match.group(1)) if ga_match else None,
        "exit_code": int(exit_match.group(1)) if exit_match else None,
        "warnings": warnings,
        **structured,
    }


def parse_metrics_from_training_log(log_path: str) -> dict[str, Any]:
    """Compatibility alias for callers that use the older entrypoint name."""
    return parse_training_log(log_path)
