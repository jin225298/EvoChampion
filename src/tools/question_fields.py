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
    "test",
    "entry_point",
)


def _text_or_none(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_code_question(item: Any) -> bool:
    """True when the item carries an executable test set (code domain)."""
    test = _text_or_none(get_question_field(item, "test", ""))
    if not test:
        test = _text_or_none(get_question_field(item, "tests", ""))
    if not test:
        test = _text_or_none(get_question_field(item, "test_code", ""))
    if not test:
        return False
    if get_question_field(item, "entry_point", "") or get_question_field(item, "function_name", ""):
        return True
    return "def check(" in test


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
    code_question = is_code_question(item)
    raw_method = text_or_empty(get_question_field(item, "evaluation_method", ""))
    evaluation_method = raw_method if raw_method in {"gold", "llm_judge", "code_execution"} else ""
    raw_needs_judge = get_question_field(item, "needs_judge", False)
    needs_judge = raw_needs_judge if isinstance(raw_needs_judge, bool) else str(raw_needs_judge).lower() in {
        "1",
        "true",
        "yes",
    }
    if code_question:
        # Code questions are judged by executed tests; never by LLM.
        evaluation_method = "code_execution"
        needs_judge = False
    if not evaluation_method:
        evaluation_method = "gold" if rollout_gold_answer or gold_answer else "llm_judge" if train_output else "gold"
    needs_judge = bool(needs_judge or evaluation_method == "llm_judge")
    return {
        "rollout_gold_answer": rollout_gold_answer,
        "train_output": train_output,
        "target_style": target_style,
        "evaluation_method": evaluation_method,
        "needs_judge": needs_judge,
        "test": _text_or_none(get_question_field(item, "test", "")) or _text_or_none(get_question_field(item, "tests", "")) or _text_or_none(get_question_field(item, "test_code", "")),
        "entry_point": _text_or_none(get_question_field(item, "entry_point", "")) or _text_or_none(get_question_field(item, "function_name", "")),
    }


def add_processed_question_fields(target: dict[str, Any], source: Any) -> dict[str, Any]:
    target.update(processed_question_fields(source))
    return target
