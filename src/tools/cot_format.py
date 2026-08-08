"""Chain-of-thought formatting with model-specific thinking delimiters detection.

Foundational module for target-style inference and thinking/output construction.
Owns the target-style constants and small text helpers so that higher-level
modules (e.g. question_fields) can build on it without a circular import.
"""

from __future__ import annotations

import re
from typing import Any

TARGET_STYLE_ANSWER = "answer"
TARGET_STYLE_COT = "cot"
TARGET_STYLES = {TARGET_STYLE_ANSWER, TARGET_STYLE_COT}
_COT_MIN_LONG_OUTPUT_CHARS = 500
_COT_RATIO_THRESHOLD = 3.0


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_target_style(value: Any) -> str | None:
    style = text_or_empty(value).lower()
    return style if style in TARGET_STYLES else None


_MODEL_CHAT_CONFIG_CACHE: dict[str, dict] = {}

_MODEL_TYPE_TO_TEMPLATE = {
    "qwen": "qwen3",
    "qwen2": "qwen3",
    "chatglm": "chatglm3",
    "llama": "llama3",
    "mistral": "mistral",
    "yi": "yi",
    "deepseek": "deepseek",
    "internlm": "internlm2",
    "baichuan": "baichuan2",
}


def detect_model_chat_config(model_name_or_path: str) -> dict:
    """Detect thinking delimiters and LlamaFactory template name from model tokenizer.

    Returns:
        {
            "thinking_delimiters": (open_tag, close_tag) | None,
            "template_name": str (e.g., "qwen3", "llama3", "default")
        }

    Cached by model_name_or_path. Zero generation cost.
    """
    if model_name_or_path in _MODEL_CHAT_CONFIG_CACHE:
        return _MODEL_CHAT_CONFIG_CACHE[model_name_or_path]

    config = {"thinking_delimiters": None, "template_name": "default"}

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        config["template_name"] = _infer_template_name(tokenizer, model_name_or_path)
        config["thinking_delimiters"] = _probe_thinking_delimiters(tokenizer)
    except Exception:
        pass

    _MODEL_CHAT_CONFIG_CACHE[model_name_or_path] = config
    return config


def _probe_thinking_delimiters(tokenizer) -> tuple[str, str] | None:
    """Probe a tokenizer's chat template for native thinking delimiters.

    Thinking-capable templates (e.g. Qwen3) expose their delimiters in the
    *generation prompt*: with ``enable_thinking=False`` the template pre-fills an
    empty thinking block (``<think>\\n\\n</think>``) that ``enable_thinking=True``
    omits. We render the generation prompt both ways and extract the tag pair
    from whichever render carries the extra block. Returns None when the template
    has no thinking mode (renders identical) or no extractable tag pair.
    """
    messages = [{"role": "user", "content": "Test question"}]
    try:
        with_thinking = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        without_thinking = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return None

    if not isinstance(with_thinking, str) or not isinstance(without_thinking, str):
        return None
    if with_thinking == without_thinking:
        return None

    return _extract_thinking_delimiters(with_thinking, without_thinking)


def _extract_thinking_delimiters(a: str, b: str) -> tuple[str, str] | None:
    """Extract an open/close tag pair from the divergent region of two renders.

    Strips the shared prefix/suffix of the two strings and searches the larger
    divergent chunk for an HTML-like open tag and its matching close tag
    (e.g. ``<think>`` / ``</think>``).
    """
    prefix = 0
    limit = min(len(a), len(b))
    while prefix < limit and a[prefix] == b[prefix]:
        prefix += 1

    suffix = 0
    while suffix < (len(a) - prefix) and suffix < (len(b) - prefix) and a[-1 - suffix] == b[-1 - suffix]:
        suffix += 1

    diff_a = a[prefix: len(a) - suffix]
    diff_b = b[prefix: len(b) - suffix]
    blob = diff_a if len(diff_a) >= len(diff_b) else diff_b

    open_match = re.search(r"<[^/<>\s]+>", blob)
    close_match = re.search(r"</[^<>\s]+>", blob)
    if open_match and close_match:
        return (open_match.group(0).strip(), close_match.group(0).strip())
    return None


def _infer_template_name(tokenizer, model_name_or_path: str) -> str:
    """Infer LlamaFactory template name from tokenizer or model path."""
    model_type = getattr(tokenizer, "model_type", None)
    if model_type and model_type.lower() in _MODEL_TYPE_TO_TEMPLATE:
        return _MODEL_TYPE_TO_TEMPLATE[model_type.lower()]

    model_name_lower = model_name_or_path.lower()
    for key, template in _MODEL_TYPE_TO_TEMPLATE.items():
        if key in model_name_lower:
            return template

    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str):
        if "qwen" in chat_template.lower():
            return "qwen3"
        if "llama" in chat_template.lower():
            return "llama3"
        if "chatglm" in chat_template.lower():
            return "chatglm3"

    return "default"


def strip_answer_marker_tail(text: str, marker: str | None = None) -> str:
    """Remove the trailing final-answer marker segment (e.g. ``#### 72``).

    Considers the dataset-specific marker and the gsm8k-style ``####``, and
    strips from whichever appears LAST in the text (the trailing one).
    Returns the reasoning with the answer marker and everything after it
    removed. No-op when neither marker is present.
    """
    stripped = text.strip()
    cut = -1
    for sep in (marker, "####"):
        if sep:
            idx = stripped.rfind(sep)
            if idx > cut:
                cut = idx
    if cut != -1:
        return stripped[:cut].rstrip()
    return stripped


def build_train_output(
    train_output: str,
    final_answer: str,
    *answer_variants: str,
    delimiters: tuple[str, str] | None = None,
    force_box: bool = False,
) -> str:
    """Build training output with thinking delimiters and boxed final answer.

    Args:
        train_output: Reasoning/thinking content
        final_answer: Final answer
        *answer_variants: Additional answer variants to check for duplication
        delimiters: (open_tag, close_tag) from detect_model_chat_config, or None
        force_box: When True, always append \\boxed{final} unless reasoning already
            contains \\boxed{ (overrides the default substring-match suppression)

    Returns:
        Formatted output string for Alpaca "output" field with boxed final answer
    """
    reasoning = train_output.strip()
    final = final_answer.strip()

    if not reasoning:
        return _format_boxed(final) if final else final

    if force_box:
        append_box = bool(final) and "\\boxed{" not in reasoning
        if delimiters:
            open_tag, close_tag = delimiters
            body = f"{open_tag}{reasoning}{close_tag}"
        else:
            body = reasoning
        return f"{body}\n\n{_format_boxed(final)}" if append_box else body

    variants = [item.strip() for item in (final, *answer_variants) if item and item.strip()]
    answer_already_in_reasoning = not final or any(
        item in reasoning or reasoning in item for item in variants
    )

    if delimiters:
        open_tag, close_tag = delimiters
        if answer_already_in_reasoning:
            return f"{open_tag}{reasoning}{close_tag}"
        return f"{open_tag}{reasoning}{close_tag}\n\n{_format_boxed(final)}"

    if answer_already_in_reasoning:
        return reasoning
    return f"{reasoning}\n\n{_format_boxed(final)}"


def _format_boxed(answer: str) -> str:
    """Format answer in \\boxed{} notation if not already boxed."""
    if not answer:
        return answer
    answer_stripped = answer.strip()
    if answer_stripped.startswith("\\boxed{") and answer_stripped.endswith("}"):
        return answer_stripped
    return f"\\boxed{{{answer_stripped}}}"


def infer_target_style(
    train_output: Any,
    rollout_gold_answer: Any = "",
    gold_answer: Any = "",
    *,
    explicit: Any = None,
) -> str:
    """Infer target style (answer vs cot) from output characteristics.

    Migrated from question_fields.py for centralization.
    """
    train_text = "" if train_output is None else text_or_empty(train_output)
    final_text = text_or_empty(rollout_gold_answer) or text_or_empty(gold_answer)
    explicit_style = normalize_target_style(explicit)

    if explicit_style == TARGET_STYLE_ANSWER:
        return TARGET_STYLE_ANSWER
    if not train_text:
        return TARGET_STYLE_ANSWER
    if explicit_style == TARGET_STYLE_COT:
        return TARGET_STYLE_COT
    if final_text and train_text == final_text:
        if len(train_text) >= _COT_MIN_LONG_OUTPUT_CHARS:
            return TARGET_STYLE_COT
        return TARGET_STYLE_ANSWER
    if len(train_text) >= _COT_MIN_LONG_OUTPUT_CHARS:
        return TARGET_STYLE_COT
    if final_text and len(train_text) >= max(_COT_MIN_LONG_OUTPUT_CHARS // 2, int(len(final_text) * _COT_RATIO_THRESHOLD)):
        return TARGET_STYLE_COT
    return TARGET_STYLE_COT
