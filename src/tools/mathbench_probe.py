from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import (
    MATHBENCH_DATASET,
    MATHBENCH_DATASET_FILTER,
    MATHBENCH_EXTRA_ARGS,
    MATHBENCH_HF_BATCH_SIZE,
    MATHBENCH_MAX_OUT_LEN,
    MATHBENCH_MAX_SEQ_LEN,
    MATHBENCH_MODEL_KWARGS,
    MATHBENCH_NO_BATCH_PADDING,
    MATHBENCH_NUM_GPUS,
    MATHBENCH_OPENCOMPASS_PYTHON,
    MATHBENCH_OPENCOMPASS_ROOT,
    MATHBENCH_SUMMARIZER,
    MATHBENCH_TOKENIZER_KWARGS,
    MATHBENCH_WORK_DIR,
    get_session_dir,
)
from src.tools.model_runner import release_model_runner_gpu_resources
from src.utils.hf_cache import resolve_model_path


_PREFERRED_METRICS = (
    "overall.naive_average",
    "overall.weighted_average",
    "overall.accuracy",
    "overall.score",
    "perf_4",
    "perf_circular",
    "perf-circular",
    "circular_perf",
    "circular",
    "mathbench",
    "acc_4",
    "accuracy",
    "acc",
)


@dataclass(frozen=True)
class MathBenchProbeResult:
    score: float
    metric_name: str
    summary_path: Path
    work_dir: Path
    raw_metrics: dict[str, float]


def mathbench_probe_enabled(method: str | None = None) -> bool:
    normalized = str(method or "").strip().lower()
    return normalized in {"mathbench", "mathbench_ce", "mathbench_opencompass"}


def build_mathbench_probe_marker(path: str | Path) -> str:
    marker_path = Path(path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    if marker_path.exists():
        return str(marker_path)
    payload = {
        "question_id": "mathbench_opencompass_probe",
        "question_text": "MathBench probe is evaluated by OpenCompass; this marker is not used for local generation.",
        "gold_answer": "",
        "rollout_gold_answer": "",
        "evaluation_method": "mathbench_opencompass",
        "source_dataset_id": "open-compass/MathBench",
        "module": "mathbench",
        "dynamic_difficulty": "mathbench",
    }
    marker_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(marker_path)


def parse_mathbench_score(work_dir: str | Path) -> MathBenchProbeResult:
    root = Path(work_dir)
    if root.is_file():
        candidate_files = [root]
        search_root = root.parent
    else:
        search_root = root
        candidate_files = _candidate_summary_files(root)
    metrics_by_file: list[tuple[Path, dict[str, float]]] = []
    for path in candidate_files:
        metrics = _extract_metrics(path)
        if metrics:
            metrics_by_file.append((path, metrics))
    if not metrics_by_file:
        raise RuntimeError(f"no MathBench summary metrics found under {root}")

    best: tuple[int, Path, str, float, dict[str, float]] | None = None
    for path, metrics in metrics_by_file:
        _metric_key, metric_name, value, rank = _choose_metric(metrics)
        if metric_name is None:
            continue
        candidate = (rank, path, metric_name, value, metrics)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError(f"no preferred MathBench metric found under {root}")
    _rank, summary_path, metric_name, raw_value, raw_metrics = best
    return MathBenchProbeResult(
        score=_normalize_score(raw_value),
        metric_name=metric_name,
        summary_path=summary_path,
        work_dir=search_root,
        raw_metrics=raw_metrics,
    )


def run_mathbench_probe(
    model_path: str,
    *,
    trace_id: str,
    round_id: int,
    model_role: str,
    force: bool = False,
) -> MathBenchProbeResult:
    resolved_model_path = resolve_model_path(model_path)
    signature = _run_signature(model_path, resolved_model_path=resolved_model_path)
    work_dir = _probe_work_dir(trace_id, round_id, model_role, resolved_model_path, signature)
    cache_path = work_dir / "mathbench_probe_result.json"
    if not force and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                return MathBenchProbeResult(
                    score=float(cached["score"]),
                    metric_name=str(cached["metric_name"]),
                    summary_path=Path(cached["summary_path"]),
                    work_dir=Path(cached["work_dir"]),
                    raw_metrics={str(k): float(v) for k, v in dict(cached.get("raw_metrics", {})).items()},
                )
        except Exception:
            pass

    work_dir.mkdir(parents=True, exist_ok=True)
    command = _build_opencompass_command(resolved_model_path, work_dir)
    release_model_runner_gpu_resources(reason="MathBench OpenCompass probe")
    completed = subprocess.run(
        command,
        cwd=_opencompass_cwd(),
        env=_opencompass_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = work_dir / "opencompass_mathbench.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "OpenCompass MathBench evaluation failed "
            f"(exit={completed.returncode}, log={log_path})"
        )

    result = parse_mathbench_score(work_dir)
    cache_payload = {
        "signature": signature,
        "model_path": model_path,
        "resolved_model_path": resolved_model_path,
        "score": result.score,
        "metric_name": result.metric_name,
        "summary_path": str(result.summary_path),
        "work_dir": str(result.work_dir),
        "raw_metrics": result.raw_metrics,
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _candidate_summary_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv"}
    ]
    files.sort(
        key=lambda path: (
            0 if "summary" in path.name.lower() else 1,
            0 if "mathbench" in str(path).lower() else 1,
            len(path.parts),
            str(path),
        )
    )
    return files


def _extract_metrics(path: Path) -> dict[str, float]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return _flatten_json_metrics(json.loads(path.read_text(encoding="utf-8")))
        if suffix == ".jsonl":
            metrics: dict[str, float] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                metrics.update(_flatten_json_metrics(json.loads(line)))
            return metrics
        if suffix == ".csv":
            return _read_csv_metrics(path)
    except Exception:
        return {}
    return {}


def _flatten_json_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(_flatten_json_metrics(child, child_prefix))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            metrics.update(_flatten_json_metrics(child, child_prefix))
    elif isinstance(value, int | float) and not isinstance(value, bool):
        metrics[prefix] = float(value)
    return metrics


def _read_csv_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row_idx, row in enumerate(rows):
        row_name = str(row.get("dataset") or row.get("name") or row.get("abbr") or row_idx)
        metric_name = str(row.get("metric") or row.get("Metric") or "").strip()
        score_found = False
        if metric_name:
            for score_key in ("score", "Score", "value", "Value"):
                try:
                    metrics[f"{row_name}.{metric_name}"] = float(str(row.get(score_key)).strip())
                    score_found = True
                    break
                except (TypeError, ValueError):
                    continue
            if not score_found:
                numeric_model_scores: list[tuple[str, float]] = []
                metadata_keys = {"dataset", "name", "abbr", "version", "metric", "Metric", "mode"}
                for key, value in row.items():
                    if key in metadata_keys:
                        continue
                    try:
                        numeric_model_scores.append((str(key), float(str(value).strip())))
                    except (TypeError, ValueError):
                        continue
                for model_key, score in numeric_model_scores:
                    metrics[f"{row_name}.{metric_name}.{model_key}"] = score
                if len(numeric_model_scores) == 1:
                    metrics[f"{row_name}.{metric_name}"] = numeric_model_scores[0][1]
        for key, value in row.items():
            try:
                metrics[f"{row_name}.{key}"] = float(str(value).strip())
            except (TypeError, ValueError):
                continue
    return metrics


def _choose_metric(metrics: dict[str, float]) -> tuple[str | None, str | None, float, int]:
    lowered = {key.lower(): key for key in metrics}
    for rank, preferred in enumerate(_PREFERRED_METRICS):
        for lower_key, original_key in lowered.items():
            if _matches_preferred_metric(lower_key, preferred):
                return original_key, _metric_display_name(original_key, preferred), metrics[original_key], rank
    return None, None, 0.0, len(_PREFERRED_METRICS)


def _matches_preferred_metric(lower_key: str, preferred: str) -> bool:
    preferred = preferred.lower()
    if "." not in preferred:
        last_part = lower_key.rsplit(".", 1)[-1]
        return last_part == preferred or lower_key.endswith(f".{preferred}")
    lower_parts = lower_key.split(".")
    preferred_parts = preferred.split(".")
    width = len(preferred_parts)
    return any(lower_parts[idx:idx + width] == preferred_parts for idx in range(0, len(lower_parts) - width + 1))


def _metric_display_name(original_key: str, preferred: str) -> str:
    if "." not in preferred:
        return original_key.rsplit(".", 1)[-1]
    preferred_parts = preferred.lower().split(".")
    original_parts = original_key.split(".")
    lower_parts = [part.lower() for part in original_parts]
    width = len(preferred_parts)
    for idx in range(0, len(lower_parts) - width + 1):
        if lower_parts[idx:idx + width] == preferred_parts:
            return ".".join(original_parts[idx:idx + width])
    return original_key


def _normalize_score(value: float) -> float:
    score = float(value)
    if score > 1.0:
        score /= 100.0
    return max(0.0, min(1.0, score))


def _probe_work_dir(
    trace_id: str,
    round_id: int,
    model_role: str,
    model_path: str,
    run_signature: str,
) -> Path:
    base = Path(MATHBENCH_WORK_DIR) if MATHBENCH_WORK_DIR else get_session_dir(trace_id) / "mathbench_probe"
    model_hash = hashlib.sha1(str(model_path).encode("utf-8")).hexdigest()[:10]
    return base / f"model_{model_hash}" / f"run_{run_signature[:10]}"


def _run_signature(model_path: str, *, resolved_model_path: str | None = None) -> str:
    payload = {
        "model_path": model_path,
        "resolved_model_path": resolved_model_path or model_path,
        "dataset": MATHBENCH_DATASET,
        "dataset_filter": MATHBENCH_DATASET_FILTER,
        "summarizer": MATHBENCH_SUMMARIZER,
        "max_seq_len": MATHBENCH_MAX_SEQ_LEN,
        "max_out_len": MATHBENCH_MAX_OUT_LEN,
        "batch_size": MATHBENCH_HF_BATCH_SIZE,
        "num_gpus": MATHBENCH_NUM_GPUS,
        "no_batch_padding": MATHBENCH_NO_BATCH_PADDING,
        "model_kwargs": MATHBENCH_MODEL_KWARGS,
        "tokenizer_kwargs": MATHBENCH_TOKENIZER_KWARGS,
        "extra_args": MATHBENCH_EXTRA_ARGS,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _opencompass_cwd() -> str | None:
    if MATHBENCH_OPENCOMPASS_ROOT:
        return str(Path(MATHBENCH_OPENCOMPASS_ROOT))
    return None


def _build_opencompass_command(model_path: str, work_dir: Path) -> list[str]:
    runner = _opencompass_runner()
    config_dir = _prepare_opencompass_config_dir(work_dir)
    model_kwargs = _split_dict_action_args(MATHBENCH_MODEL_KWARGS)
    tokenizer_kwargs = _split_dict_action_args(MATHBENCH_TOKENIZER_KWARGS)
    command = [
        *runner,
        "--datasets",
        MATHBENCH_DATASET,
        "--hf-path",
        model_path,
        "--max-seq-len",
        str(MATHBENCH_MAX_SEQ_LEN),
        "--batch-size",
        str(MATHBENCH_HF_BATCH_SIZE),
        "--hf-num-gpus",
        str(MATHBENCH_NUM_GPUS),
        "--summarizer",
        MATHBENCH_SUMMARIZER,
        "--work-dir",
        str(work_dir),
    ]
    if model_kwargs:
        command.extend(["--model-kwargs", *model_kwargs])
    if tokenizer_kwargs:
        command.extend(["--tokenizer-kwargs", *tokenizer_kwargs])
    if config_dir is not None:
        command.extend(["--config-dir", str(config_dir)])
    if MATHBENCH_NO_BATCH_PADDING:
        command.append("--no-batch-padding")
    if MATHBENCH_EXTRA_ARGS:
        command.extend(shlex.split(MATHBENCH_EXTRA_ARGS))
    return command


def _prepare_opencompass_config_dir(work_dir: Path) -> Path | None:
    if MATHBENCH_MAX_OUT_LEN <= 0:
        return None
    if MATHBENCH_DATASET != "mathbench_gen":
        return None
    if not MATHBENCH_OPENCOMPASS_ROOT:
        return None

    source_dir = Path(MATHBENCH_OPENCOMPASS_ROOT) / "opencompass" / "configs" / "datasets" / "MathBench"
    source_config = source_dir / "mathbench_2024_gen_19e486.py"
    if not source_config.exists():
        return None

    target_dir = work_dir / "opencompass_config" / "datasets" / "MathBench"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, target_dir / source_config.name)
    selected_abbrs = _mathbench_dataset_filter_abbrs()
    lines = [
        "from mmengine.config import read_base",
        "",
        "with read_base():",
        "    from .mathbench_2024_gen_19e486 import mathbench_datasets",
        "",
    ]
    if selected_abbrs:
        lines.extend(
            [
                f"_mathbench_selected_abbrs = {selected_abbrs!r}",
                "_mathbench_selected = set(_mathbench_selected_abbrs)",
                "mathbench_datasets = [",
                "    _dataset",
                "    for _dataset in mathbench_datasets",
                "    if _dataset.get('abbr') in _mathbench_selected",
                "]",
                "if len(mathbench_datasets) != len(_mathbench_selected):",
                "    _found = {_dataset.get('abbr') for _dataset in mathbench_datasets}",
                "    _missing = sorted(_mathbench_selected - _found)",
                "    raise ValueError(f'MathBench dataset filter matched no dataset(s): {_missing}')",
                "",
            ]
        )
    lines.extend(
        [
            "for _dataset in mathbench_datasets:",
            "    _dataset['infer_cfg']['inferencer']['max_out_len'] = "
            f"{int(MATHBENCH_MAX_OUT_LEN)}",
            "",
        ]
    )
    (target_dir / "mathbench_gen.py").write_text("\n".join(lines), encoding="utf-8")
    return work_dir / "opencompass_config"


def _mathbench_dataset_filter_abbrs() -> list[str]:
    seen: set[str] = set()
    abbrs: list[str] = []
    for item in MATHBENCH_DATASET_FILTER.replace("\n", ",").split(","):
        abbr = item.strip()
        if not abbr or abbr in seen:
            continue
        seen.add(abbr)
        abbrs.append(abbr)
    return abbrs


def _split_dict_action_args(value: str) -> list[str]:
    args: list[str] = []
    for item in shlex.split(value or ""):
        if "=" not in item:
            args.append(item)
            continue
        key, raw_value = item.split("=", 1)
        args.append(f"{key}={raw_value}")
    return args


def _opencompass_env() -> dict[str, str]:
    env = os.environ.copy()
    if MATHBENCH_OPENCOMPASS_ROOT:
        existing = env.get("PYTHONPATH")
        root = str(Path(MATHBENCH_OPENCOMPASS_ROOT))
        env["PYTHONPATH"] = root if not existing else f"{root}{os.pathsep}{existing}"
    return env


def _opencompass_runner() -> list[str]:
    python = MATHBENCH_OPENCOMPASS_PYTHON or sys.executable
    if MATHBENCH_OPENCOMPASS_ROOT:
        run_py = Path(MATHBENCH_OPENCOMPASS_ROOT) / "run.py"
        if not run_py.exists():
            raise RuntimeError(f"OpenCompass run.py not found at {run_py}")
        return [python, str(run_py)]
    executable = shutil.which("opencompass")
    if executable:
        return [executable]
    module = shutil.which("python") or python
    if _module_available("opencompass"):
        return [module, "-m", "opencompass"]
    raise RuntimeError(
        "OpenCompass is required for MathBench probe. Set "
        "MATHBENCH_OPENCOMPASS_ROOT=/path/to/opencompass or install the opencompass CLI."
    )


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False
