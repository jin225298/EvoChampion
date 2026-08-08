"""Shared vLLM engine helpers for Ray-backed inference workers.

The old implementation exposed a Unix-socket subprocess server from this file.
The pure Ray route keeps the request/response and generation logic here, while
Ray actors own worker process lifecycle, routing, and restart behavior.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any
from uuid import uuid4


class _FallbackSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = _FallbackSamplingParams


def _count_tokens(tokenizer, text: str) -> int:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except Exception:
        return 0
    input_ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else []
    return len(input_ids)


def _truncate_text_by_tokens(tokenizer, text: str, max_prompt_tokens: int) -> tuple[str, int, int, bool]:
    original_count = _count_tokens(tokenizer, text)
    if max_prompt_tokens <= 0 or original_count <= max_prompt_tokens:
        return text, original_count, original_count, False
    try:
        encoded = tokenizer(text, truncation=False, add_special_tokens=False)
        input_ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else []
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        truncated_ids = input_ids[-max_prompt_tokens:]
        truncated = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    except Exception:
        return text, original_count, original_count, False
    final_count = _count_tokens(tokenizer, truncated)
    return truncated, original_count, final_count, final_count < original_count


def _sampling_params_supports_truncate() -> bool:
    try:
        return "truncate_prompt_tokens" in inspect.signature(SamplingParams).parameters
    except (TypeError, ValueError):
        return False


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return max(1, default)
    return max(1, value)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")


def _callable_accepts_keyword(callable_obj, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return (
        keyword in signature.parameters
        or any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    )


def _effective_inflight_per_call(max_new_tokens: int) -> int:
    configured = _positive_int_env("VLLM_RAY_INFLIGHT_PER_CALL", 8)
    output_token_budget = _positive_int_env("VLLM_RAY_MAX_INFLIGHT_OUTPUT_TOKENS", 1024)
    token_limited = max(1, output_token_budget // max(1, int(max_new_tokens or 1)))
    return max(1, min(configured, token_limited))


def _apply_chat_template(tokenizer, messages: list[dict], disable_thinking: bool = False) -> str:
    if disable_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_engine(
    model_path: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    enable_prefix_caching: bool = True,
) -> Any:
    os.environ.setdefault("XET_DISABLE", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    import torch
    from vllm import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    max_num_batched_tokens = max(
        _positive_int_env("VLLM_MAX_NUM_BATCHED_TOKENS", max_model_len),
        max_model_len,
    )
    engine_args = AsyncEngineArgs(
        **{
            **dict(
                model=model_path,
                trust_remote_code=True,
                dtype="bfloat16" if torch.cuda.is_available() else "float32",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=_positive_int_env("VLLM_MAX_NUM_SEQS", 64),
                max_num_batched_tokens=max_num_batched_tokens,
                enforce_eager=enforce_eager,
                enable_prefix_caching=enable_prefix_caching,
            ),
            **(
                {"enable_chunked_prefill": _bool_env("VLLM_ENABLE_CHUNKED_PREFILL", True)}
                if _callable_accepts_keyword(AsyncEngineArgs, "enable_chunked_prefill")
                else {}
            ),
        }
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


async def _generate_one(
    engine: Any,
    text: str,
    sampling_params: Any,
    request_id: str,
) -> str:
    final_output = None
    async for output in engine.generate(text, sampling_params, request_id=request_id):
        final_output = output
    if final_output is None or not final_output.outputs:
        return ""
    return final_output.outputs[0].text.strip()


async def _get_engine_tokenizer(engine: Any):
    tokenizer = engine.get_tokenizer()
    if inspect.isawaitable(tokenizer):
        return await tokenizer
    return tokenizer


async def _handle_generate(engine: Any, tokenizer, request: dict) -> dict:
    prompts = [str(item) for item in request.get("prompts", [])]
    disable_thinking = bool(request.get("disable_thinking", False))
    stop_after_json = bool(request.get("stop_after_json", False))
    max_prompt_tokens = int(request.get("max_prompt_tokens", 0)) or None

    texts = [
        _apply_chat_template(
            tokenizer,
            [{"role": "user", "content": prompt}],
            disable_thinking=disable_thinking,
        )
        for prompt in prompts
    ]

    original_token_counts = []
    final_token_counts = []
    truncated_flags = []
    truncation_policy = "none"
    if max_prompt_tokens and max_prompt_tokens > 0:
        truncated_texts = []
        for text in texts:
            truncated_text, original_count, final_count, was_truncated = _truncate_text_by_tokens(
                tokenizer,
                text,
                max_prompt_tokens,
            )
            truncated_texts.append(truncated_text)
            original_token_counts.append(original_count)
            final_token_counts.append(final_count)
            truncated_flags.append(was_truncated)
        texts = truncated_texts
        truncation_policy = "pre_tokenize_keep_tail" if any(truncated_flags) else "none"
    else:
        original_token_counts = [_count_tokens(tokenizer, text) for text in texts]
        final_token_counts = list(original_token_counts)

    max_new_tokens = int(request.get("max_new_tokens", 96))
    sampling_params_kwargs = dict(
        temperature=float(request.get("temperature", 0.0)),
        top_p=float(request.get("top_p", 1.0)),
        max_tokens=max_new_tokens,
        stop=["```"] if stop_after_json else None,
    )
    if max_prompt_tokens and max_prompt_tokens > 0 and _sampling_params_supports_truncate():
        sampling_params_kwargs["truncate_prompt_tokens"] = max_prompt_tokens
        if truncation_policy == "none":
            truncation_policy = "sampling_params"
    sampling_params = SamplingParams(**sampling_params_kwargs)

    request_prefix = str(request.get("request_id") or uuid4())
    inflight_per_call = _effective_inflight_per_call(max_new_tokens)

    semaphore = asyncio.Semaphore(inflight_per_call)

    async def _generate_with_semaphore(text: str, idx: int) -> str:
        async with semaphore:
            return await _generate_one(engine, text, sampling_params, f"{request_prefix}:{idx}")

    tasks = [
        asyncio.create_task(_generate_with_semaphore(text, idx))
        for idx, text in enumerate(texts)
    ]
    results = await asyncio.gather(*tasks)

    return {
        "ok": True,
        "results": results,
        "metadata": {
            "original_prompt_tokens": original_token_counts,
            "final_prompt_tokens": final_token_counts,
            "truncated": any(truncated_flags),
            "truncation_policy": truncation_policy,
            "max_prompt_tokens": max_prompt_tokens,
            "inflight_per_call": inflight_per_call,
        },
    }


async def _shutdown_engine(engine: Any) -> None:
    shutdown = getattr(engine, "shutdown_background_loop", None)
    if shutdown is not None:
        result = shutdown()
        if inspect.isawaitable(result):
            await result
        return
    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        result = shutdown()
        if inspect.isawaitable(result):
            await result
