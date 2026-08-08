from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


_HF_TIMEOUT_MODULE = "huggingface_hub.constants"
_HF_HUB_TIMEOUT_CONSTANTS = (
    "HF_HUB_ETAG_TIMEOUT",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "DEFAULT_ETAG_TIMEOUT",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_DOWNLOAD_TIMEOUT",
)


def patch_hub_timeout(timeout: float) -> Callable[[], None]:
    try:
        module = importlib.import_module(_HF_TIMEOUT_MODULE)
    except Exception:
        return lambda: None

    saved: list[tuple[str, Any]] = []
    for attr in _HF_HUB_TIMEOUT_CONSTANTS:
        if hasattr(module, attr):
            saved.append((attr, getattr(module, attr)))
            setattr(module, attr, timeout)

    def restore() -> None:
        for attr, value in reversed(saved):
            setattr(module, attr, value)

    return restore


def load_dataset_card_summary(
    dataset_id: str,
    *,
    text_char_limit: int,
    timeout: float,
) -> dict[str, Any]:
    restore = patch_hub_timeout(timeout)
    try:
        card_cls = getattr(importlib.import_module("huggingface_hub"), "DatasetCard")
        card = card_cls.load(dataset_id)
        raw_text = str(getattr(card, "text", "") or "")
        return {
            "available": True,
            "metadata": _json_safe_value(getattr(card, "data", {}) or {}),
            "text": raw_text[:text_char_limit],
            "truncated": len(raw_text) > text_char_limit,
        }
    except Exception as exc:
        return {
            "available": False,
            "metadata": {},
            "text": "",
            "truncated": False,
            "error": _short_exception(exc),
        }
    finally:
        restore()


def _json_safe_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth >= 4:
        return str(value)[:500]
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item, depth=depth + 1)
            for key, item in list(value.items())[:80]
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item, depth=depth + 1) for item in list(value)[:80]]
    return str(value)[:500]


def _short_exception(exc: Exception, max_len: int = 240) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= max_len else text[: max_len - 3] + "..."
