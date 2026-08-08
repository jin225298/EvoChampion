import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

from config.settings import (
    INFERENCE_MAX_NEW_TOKENS,
    INFERENCE_TRACE_ENABLED,
    INFERENCE_TRACE_MAX_TEXT_CHARS,
    get_session_dir,
)


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _clip_text(value: Any) -> str:
    text = str(value or "")
    limit = max(0, int(INFERENCE_TRACE_MAX_TEXT_CHARS or 0))
    if limit and len(text) > limit:
        return text[:limit] + f"...<truncated {len(text) - limit} chars>"
    return text


def _last_number(text: str) -> str:
    matches = _NUMBER_RE.findall(text or "")
    return matches[-1] if matches else ""


def build_inference_trace_row(
    *,
    trace_id: str,
    round_id: int,
    stage: str,
    model_role: str,
    model_path: str,
    question_id: str = "",
    prompt: str = "",
    gold_answer: str = "",
    prediction: str = "",
    correct: bool = False,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    rollout_idx: int | None = None,
    split_role: str = "",
    module: str = "",
    dynamic_difficulty: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_max_new_tokens = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else int(INFERENCE_MAX_NEW_TOKENS)
    )
    prediction_text = str(prediction or "")
    prediction_words = len(prediction_text.split())
    prediction_chars = len(prediction_text)
    stripped_prediction = prediction_text.rstrip()
    ends_with_terminal = stripped_prediction.endswith((".", "!", "?", "。", "！", "？", "}", "]"))
    near_token_budget = prediction_words >= max(1, int(effective_max_new_tokens * 0.75))
    possible_truncation = bool(
        prediction_text
        and near_token_budget
        and not ends_with_terminal
    )
    return {
        "ts": time.time(),
        "trace_id": trace_id,
        "round_id": round_id,
        "stage": stage,
        "model_role": model_role,
        "model_path": model_path,
        "question_id": question_id,
        "split_role": split_role,
        "module": module,
        "dynamic_difficulty": dynamic_difficulty,
        "rollout_idx": rollout_idx,
        "correct": bool(correct),
        "gold_answer": _clip_text(gold_answer),
        "prediction": _clip_text(prediction_text),
        "prompt": _clip_text(prompt),
        "prediction_chars": prediction_chars,
        "prediction_words": prediction_words,
        "gold_last_number": _last_number(str(gold_answer or "")),
        "prediction_last_number": _last_number(prediction_text),
        "contains_gold_answer": bool(gold_answer) and str(gold_answer).strip().lower() in prediction_text.lower(),
        "near_token_budget_by_words": near_token_budget,
        "possible_truncation": possible_truncation,
        "max_new_tokens": effective_max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "metadata": metadata or {},
    }


def write_inference_trace_rows(
    *,
    trace_id: str,
    round_id: int,
    stage: str,
    rows: Iterable[dict[str, Any]],
) -> Path | None:
    if not INFERENCE_TRACE_ENABLED:
        return None
    session_dir = get_session_dir(trace_id)
    trace_dir = session_dir / "inference_traces" / f"round_{int(round_id)}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{stage}_{int(time.time() * 1000)}.jsonl"
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    print(f"[inference_trace] wrote {count} rows to {path}")
    return path
