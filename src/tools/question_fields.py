"""Shared helpers for preserving processed question output fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.tools.cot_format import (
    TARGET_STYLE_ANSWER,
    TARGET_STYLE_COT,
    TARGET_STYLES,
    _COT_MIN_LONG_OUTPUT_CHARS,
    _COT_RATIO_THRESHOLD,
    infer_target_style,
    normalize_target_style,
    text_or_empty,
)


PROCESSED_QUESTION_FIELDS = (
    "rollout_gold_answer",
    "train_output",
    "target_style",
    "evaluation_method",
    "needs_judge",
)


def get_question_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def processed_question_fields(item: Any) -> dict[str, Any]:
    gold_answer = text_or_empty(get_question_field(item, "gold_answer", ""))
    rollout_gold_answer = text_or_empty(
        get_question_field(item, "rollout_gold_answer", "")
    ) or gold_answer
    train_output = text_or_empty(get_question_field(item, "train_output", ""))
    target_style = infer_target_style(
        train_output,
        rollout_gold_answer,
        gold_answer,
        explicit=get_question_field(item, "target_style", None),
    )
    if target_style == TARGET_STYLE_ANSWER and not train_output:
        train_output = rollout_gold_answer
    raw_method = text_or_empty(get_question_field(item, "evaluation_method", ""))
    evaluation_method = raw_method if raw_method in {"gold", "llm_judge", "code_exec"} else ""
    raw_needs_judge = get_question_field(item, "needs_judge", False)
    needs_judge = raw_needs_judge if isinstance(raw_needs_judge, bool) else str(raw_needs_judge).lower() in {
        "1",
        "true",
        "yes",
    }
    if not evaluation_method:
        # Code domain + executable test -> code-execution judging.
        _is_code = False
        try:
            from config import settings as _settings
            _is_code = getattr(_settings, "IS_CODE_DOMAIN", False)
        except Exception:
            _is_code = False
        _has_test = bool(text_or_empty(get_question_field(item, "test", "")))
        if _is_code and _has_test:
            evaluation_method = "code_exec"
        elif rollout_gold_answer or gold_answer:
            evaluation_method = "gold"
        elif train_output:
            evaluation_method = "llm_judge"
        else:
            evaluation_method = "gold"
    needs_judge = bool(needs_judge or evaluation_method == "llm_judge")
    return {
        "rollout_gold_answer": rollout_gold_answer,
        "train_output": train_output,
        "target_style": target_style,
        "evaluation_method": evaluation_method,
        "needs_judge": needs_judge,
    }


def add_processed_question_fields(target: dict[str, Any], source: Any) -> dict[str, Any]:
    target.update(processed_question_fields(source))
    return target
