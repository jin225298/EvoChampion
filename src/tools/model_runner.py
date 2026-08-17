"""Real model inference and answer judging for the EvoChampion system.
Uses HuggingFace transformers for model loading and generation.
Loads model lazily and caches for reuse across calls.

Loading strategy:
  1. Resolve model IDs to local snapshot paths (bypasses hf-xet crash)
  2. Always use local_files_only=True with resolved snapshot paths
  3. For LoRA adapters, deep-copy the cached base model (avoids meta-tensor errors)
  4. Thread-safe: concurrent workers share the cache via _cache_lock

Performance optimizations (2026-05-09):
  - Added run_model_batch() for batch inference
  - Reduced default max_new_tokens from 256 to 96 for eval/rollout
  - Fixed generation config warning (removed invalid top_p/top_k when do_sample=False)
  - Added timing logs to all inference functions
"""

import base64
import copy
import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import time
import gc
import threading
from typing import Any
from pathlib import Path
from collections import OrderedDict

# Disable hf-xet BEFORE any huggingface imports
os.environ.setdefault("XET_DISABLE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

    class StoppingCriteria:
        pass

    class StoppingCriteriaList(list):
        pass

from src.utils.hf_cache import resolve_model_path
from src.tools.vllm_worker_manager import RayVllmModelSwitchGuard


def _require_inference_deps() -> None:
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError("torch and transformers are required for model inference")


def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _cuda_empty_cache() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_synchronize() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


_model_cache: dict[str, tuple] = {}
_cache_lock = threading.Lock()
_cache_loading: dict[str, threading.Event] = {}
_vllm_cache: OrderedDict[str, tuple] = OrderedDict()
_vllm_generation_locks: dict[str, threading.RLock] = {}
_vllm_worker_cache: OrderedDict[str, Any] = OrderedDict()
_vllm_worker_global_start_lock = threading.Lock()
_vllm_worker_start_locks: dict[str, threading.Lock] = {}
_vllm_disabled_models: dict[str, str] = {}
_ray_init_lock = threading.Lock()


class _LocalCheckpointGpuGate:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._exclusive_active = False
        self._shared_active = 0

    def acquire_shared(self) -> None:
        with self._condition:
            while self._exclusive_active:
                self._condition.wait()
            self._shared_active += 1

    def release_shared(self) -> None:
        with self._condition:
            self._shared_active -= 1
            if self._shared_active == 0:
                self._condition.notify_all()

    def acquire_exclusive(self) -> None:
        with self._condition:
            while self._exclusive_active or self._shared_active:
                self._condition.wait()
            self._exclusive_active = True

    def release_exclusive(self) -> None:
        with self._condition:
            self._exclusive_active = False
            self._condition.notify_all()


_local_checkpoint_gpu_gate = _LocalCheckpointGpuGate()
_ray_vllm_model_switch_guard: RayVllmModelSwitchGuard | None = None


_VLLM_MEMORY_ERROR_PATTERNS = (
    "no available memory",
    "out of memory",
    "engine core initialization",
    "enginecore failed",
    "engine core failed",
    "engine is dead",
    "engine dead",
)


# ── Prompt token guard helpers ──────────────────────────────────────────────
# These functions provide safe prompt token budget computation and
# pre-tokenization / truncation for HF and vLLM generation paths.
# They avoid hard failures when tokenizers lack expected methods.
# ──────────────────────────────────────────────────────────────────────────────


def _get_safe_prompt_token_budget(
    max_model_len: int | None = None,
    max_new_tokens: int = 96,
    safety_margin: int = 16,
) -> int:
    """Compute a safe prompt token budget for HF generation.

    Uses VLLM_MAX_MODEL_LEN env var as the authoritative max model length
    when max_model_len is not explicitly provided. Falls back to 4096.
    Always reserves room for max_new_tokens plus a safety margin.

    Returns:
        Maximum number of prompt tokens to send to the model.
    """
    if max_model_len is None:
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
    budget = max_model_len - max_new_tokens - safety_margin
    return max(64, budget)


def _get_inference_batch_size(default: int = 8) -> int:
    try:
        from config.settings import INFERENCE_BATCH_SIZE
        return max(1, int(INFERENCE_BATCH_SIZE or default))
    except Exception:
        return max(1, default)


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    if not items:
        return []
    safe_batch_size = max(1, int(batch_size))
    return [items[i : i + safe_batch_size] for i in range(0, len(items), safe_batch_size)]


def _pre_tokenize_texts(
    tokenizer,
    texts: list[str],
    max_prompt_tokens: int,
) -> list[str]:
    """Pre-tokenize and truncate texts so prompt tokens fit within budget.

    For each text, tokenizes, counts tokens, and if over budget,
    truncates token ids and decodes back. Falls back gracefully when
    tokenizer lacks expected methods (e.g., no tokenizer available).

    Returns:
        Truncated texts (in-place if under budget).
    """
    if not texts or max_prompt_tokens <= 0:
        return texts

    try:
        encoded = tokenizer(texts, truncation=False, add_special_tokens=False)
    except Exception:
        # Tokenizer may not support the expected interface; return as-is
        return texts

    input_ids_list = encoded.get("input_ids", [])
    if not input_ids_list:
        return texts
    if input_ids_list and isinstance(input_ids_list[0], int):
        input_ids_list = [input_ids_list]

    result = []
    for text, ids in zip(texts, input_ids_list):
        if len(ids) <= max_prompt_tokens:
            result.append(text)
        else:
            # Keep the tail of the prompt. For agent prompts and chat templates,
            # the most recent context and output constraints are at the end.
            try:
                truncated_ids = ids[-max_prompt_tokens:]
                result.append(tokenizer.decode(truncated_ids, skip_special_tokens=True))
            except Exception:
                result.append(text)
    return result


def _count_tokens(tokenizer, text: str) -> int:
    """Safely count tokens without raising on missing methods."""
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        return len(encoded.get("input_ids", []))
    except Exception:
        return 0


def _sampling_params_supports_truncate(sampling_params_cls) -> bool:
    try:
        return "truncate_prompt_tokens" in inspect.signature(sampling_params_cls).parameters
    except (TypeError, ValueError):
        return False


def _build_sampling_params(
    SamplingParams,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop,
    prompt_budget: int,
):
    kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "stop": stop,
    }
    if prompt_budget > 0 and _sampling_params_supports_truncate(SamplingParams):
        kwargs["truncate_prompt_tokens"] = prompt_budget
    return SamplingParams(**kwargs)


def _get_instruction_prefix() -> str:
    return os.environ.get("INSTRUCTION_PREFIX", "请解答下面的题目,并在最后将最终的数值答案写在 \\boxed{} 中,例如 \\boxed{42}。\n") or "请解答下面的题目,并在最后将最终的数值答案写在 \\boxed{} 中,例如 \\boxed{42}。\n"


class VllmEngineLoadError(RuntimeError):
    """Raised after vLLM engine loading has already exhausted its retry."""


def _has_complete_json_object(text: str) -> bool:
    if not text:
        return False
    if "<think>" in text and "</think>" not in text:
        return False
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]

    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if start is None:
            if ch == "{":
                start = 0
                depth = 1
            continue
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


class _JsonObjectStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for row in input_ids:
            generated = row[self.prompt_length:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if not _has_complete_json_object(text):
                return False
        return True


class IsolatedInferenceError(RuntimeError):
    """Raised when a subprocess-isolated inference request fails."""


_BATCH_JSON_RESPONSE_START = "<<<MODEL_RUNNER_BATCH_JSON_RESPONSE_START_v1>>>"
_BATCH_JSON_RESPONSE_END = "<<<MODEL_RUNNER_BATCH_JSON_RESPONSE_END_v1>>>"
_BATCH_JSON_MAX_PAYLOAD_CHARS = int(os.environ.get("MODEL_RUNNER_BATCH_JSON_MAX_PAYLOAD_CHARS", "10485760"))
_BATCH_JSON_ERROR_SNIPPET_CHARS = 1000


def _format_batch_json_response(response: dict[str, Any]) -> str:
    payload = base64.b64encode(json.dumps(response, ensure_ascii=False).encode("utf-8")).decode("ascii")
    return f"{_BATCH_JSON_RESPONSE_START}\n{payload}\n{_BATCH_JSON_RESPONSE_END}\n"


def _write_batch_json_response(stdout_fd: int, response: dict[str, Any]) -> None:
    os.write(stdout_fd, _format_batch_json_response(response).encode("utf-8"))


def _extract_batch_json_response(stdout: str) -> dict[str, Any]:
    start = stdout.rfind(_BATCH_JSON_RESPONSE_START)
    if start < 0:
        raise IsolatedInferenceError(
            f"isolated local checkpoint inference did not return sentinel-delimited JSON: "
            f"{stdout[-1000:]}"
        )

    payload_start = start + len(_BATCH_JSON_RESPONSE_START)
    end = stdout.find(_BATCH_JSON_RESPONSE_END, payload_start)
    if end < 0:
        raise IsolatedInferenceError(
            f"isolated local checkpoint inference returned incomplete sentinel-delimited JSON: "
            f"{stdout[start:][-1000:]}"
        )

    payload = stdout[payload_start:end].strip()
    if len(payload) > _BATCH_JSON_MAX_PAYLOAD_CHARS:
        raise IsolatedInferenceError(
            f"isolated local checkpoint inference returned oversized JSON payload: "
            f"{len(payload)} chars exceeds {_BATCH_JSON_MAX_PAYLOAD_CHARS}"
        )
    try:
        raw_payload = base64.b64decode(payload.encode("ascii"), validate=True).decode("utf-8")
        response = json.loads(raw_payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise IsolatedInferenceError(
            f"isolated local checkpoint inference returned invalid JSON payload: "
            f"{payload[-_BATCH_JSON_ERROR_SNIPPET_CHARS:]}"
        ) from exc
    if not isinstance(response, dict):
        raise IsolatedInferenceError(
            f"isolated local checkpoint inference returned {type(response).__name__} instead of object"
        )
    return response


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


def _is_vllm_memory_error(error: BaseException) -> bool:
    error_msg = str(error).lower()
    return any(pattern in error_msg for pattern in _VLLM_MEMORY_ERROR_PATTERNS)


def _clear_hf_cache_unlocked() -> None:
    _model_cache.clear()


def _shutdown_vllm_engine(llm) -> None:
    shutdown_targets = [
        getattr(llm, "llm_engine", None),
        getattr(llm, "engine", None),
        getattr(llm, "engine_core", None),
    ]
    for target in shutdown_targets:
        if target is not None and hasattr(target, "shutdown"):
            try:
                target.shutdown()
            except Exception:
                pass
    if hasattr(llm, "shutdown"):
        try:
            llm.shutdown()
        except Exception:
            pass


def _evict_vllm_engine_unlocked(cache_key: str) -> None:
    item = _vllm_cache.pop(cache_key, None)
    if item is None:
        return
    llm, _tokenizer = item
    _shutdown_vllm_engine(llm)
    try:
        del llm
    except Exception as exc:
        print(f"[model_runner] warmup_model vLLM warmup failed ({type(exc).__name__}: {exc}); falling back to HF warmup")


def _clear_vllm_cache_unlocked() -> None:
    for cache_key in list(_vllm_cache.keys()):
        _evict_vllm_engine_unlocked(cache_key)


def _pop_all_vllm_workers_unlocked() -> list[Any]:
    actors = list(_vllm_worker_cache.values())
    _vllm_worker_cache.clear()
    return actors


def _pop_vllm_worker_unlocked(cache_key: str):
    return _vllm_worker_cache.pop(cache_key, None)


def _shutdown_popped_vllm_workers(actors: list[Any]) -> None:
    if not actors:
        return
    for actor in actors:
        _shutdown_ray_vllm_actor(actor)
    _collect_cuda()


def _get_vllm_generation_lock(cache_key: str) -> threading.RLock:
    with _cache_lock:
        lock = _vllm_generation_locks.get(cache_key)
        if lock is None:
            lock = threading.RLock()
            _vllm_generation_locks[cache_key] = lock
        return lock


def _get_ray_actor_name(resolved_model_path: str) -> str:
    digest = hashlib.sha256(_ray_actor_identity(resolved_model_path).encode("utf-8")).hexdigest()[:16]
    return f"agents_evolve_vllm_{digest}"


def _get_legacy_ray_actor_name(resolved_model_path: str) -> str:
    digest = hashlib.sha256(resolved_model_path.encode("utf-8")).hexdigest()[:16]
    return f"agents_evolve_vllm_{digest}"


def _callable_accepts_keyword(callable_obj, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return (
        keyword in signature.parameters
        or any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    )


def _vllm_chunked_prefill_enabled() -> bool:
    return _env_truthy("VLLM_ENABLE_CHUNKED_PREFILL", default=True)


def _vllm_scheduler_identity() -> str:
    fields = (
        "VLLM_GPU_MEMORY_UTILIZATION",
        "VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_NUM_BATCHED_TOKENS",
        "VLLM_ENFORCE_EAGER",
        "VLLM_ENABLE_PREFIX_CACHING",
        "VLLM_ENABLE_CHUNKED_PREFILL",
        "VLLM_RAY_ACTOR_NUM_GPUS",
        "VLLM_RAY_ACTOR_MAX_CONCURRENCY",
    )
    return "|".join(f"{field}={os.environ.get(field, '')}" for field in fields)


def _ray_actor_identity(resolved_model_path: str) -> str:
    return f"{resolved_model_path}|{_vllm_scheduler_identity()}"


def _ensure_ray_initialized():
    import ray
    from config.settings import RAY_ADDRESS, RAY_NAMESPACE

    if ray.is_initialized():
        return ray
    with _ray_init_lock:
        if not ray.is_initialized():
            address = (RAY_ADDRESS or "local").strip() or "local"
            init_kwargs = {
                "namespace": RAY_NAMESPACE,
                "ignore_reinit_error": True,
            }
            if address.lower() != "local":
                init_kwargs["address"] = address
            context = ray.init(**init_kwargs)
            if address.lower() == "local":
                _publish_local_ray_address(context)
    return ray


def _publish_local_ray_address(context: Any) -> None:
    """Publish a local Ray cluster address so spawned workers join this job.

    A parent process started with ``RAY_ADDRESS=local`` creates an isolated Ray
    runtime.  Multiprocessing children cannot see that in-memory runtime, and
    using ``auto`` can attach them to a different Slurm job's Ray cluster on the
    same node.  Propagate the concrete address returned by ray.init instead.
    """
    address_info = getattr(context, "address_info", None)
    if not isinstance(address_info, dict):
        return
    address = str(address_info.get("address") or address_info.get("gcs_address") or "").strip()
    if not address:
        return
    os.environ["RAY_ADDRESS"] = address
    try:
        import config.settings as settings_module
        settings_module.RAY_ADDRESS = address
    except Exception:
        pass


def _get_vllm_worker_start_lock(cache_key: str) -> threading.Lock:
    with _cache_lock:
        lock = _vllm_worker_start_locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _vllm_worker_start_locks[cache_key] = lock
        return lock


def _shutdown_ray_vllm_actor(actor) -> None:
    try:
        ray = _ensure_ray_initialized()
        try:
            ray.get(actor.shutdown.remote(), timeout=10)
        except Exception as exc:
            print(f"[model_runner] Ray vLLM actor graceful shutdown failed: {type(exc).__name__}: {exc}")
        ray.kill(actor, no_restart=True)
        _wait_for_ray_vllm_actor_exit(ray, actor)
    except Exception as exc:
        print(f"[model_runner] Ray vLLM actor kill failed: {type(exc).__name__}: {exc}")


def _wait_for_ray_vllm_actor_exit(ray, actor) -> None:
    timeout = float(os.environ.get("VLLM_RAY_ACTOR_SHUTDOWN_TIMEOUT_SECONDS", "30"))
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        try:
            ray.get(actor.ready.remote(), timeout=1)
        except Exception:
            return
        time.sleep(0.25)
    print(
        "[model_runner] Ray vLLM actor still reachable after shutdown timeout; "
        "continuing after best-effort kill"
    )


def _shutdown_vllm_worker_unlocked(cache_key: str) -> None:
    actor = _pop_vllm_worker_unlocked(cache_key)
    _shutdown_popped_vllm_workers([actor] if actor is not None else [])


def _evict_vllm_worker_unlocked(cache_key: str):
    return _pop_vllm_worker_unlocked(cache_key)


def _evict_vllm_worker(cache_key: str) -> None:
    with _cache_lock:
        actor = _evict_vllm_worker_unlocked(cache_key)
    _shutdown_popped_vllm_workers([actor] if actor is not None else [])


def _shutdown_named_vllm_worker_unlocked(cache_key: str) -> None:
    actor = _vllm_worker_cache.pop(cache_key, None)
    if actor is None:
        try:
            ray = _ensure_ray_initialized()
            actor = ray.get_actor(_get_ray_actor_name(cache_key))
        except Exception:
            return
    _shutdown_ray_vllm_actor(actor)
    _collect_cuda()


def _shutdown_legacy_named_vllm_worker(resolved_model_path: str) -> None:
    legacy_actor_name = _get_legacy_ray_actor_name(resolved_model_path)
    current_actor_name = _get_ray_actor_name(resolved_model_path)
    if legacy_actor_name == current_actor_name:
        return
    try:
        ray = _ensure_ray_initialized()
        actor = ray.get_actor(legacy_actor_name)
    except Exception:
        return
    print(
        "[model_runner] Clearing legacy Ray vLLM actor before profiled actor start: "
        f"model_path={resolved_model_path} actor_name={legacy_actor_name}"
    )
    _shutdown_popped_vllm_workers([actor])


def _clear_vllm_workers_unlocked() -> list[Any]:
    return _pop_all_vllm_workers_unlocked()


def _clear_stale_ray_vllm_workers_for_switch(target_cache_key: str, previous_cache_key: str) -> list[Any]:
    stale_keys: list[str] = []
    actors_to_shutdown: list[Any] = []
    with _cache_lock:
        for cache_key in list(_vllm_worker_cache.keys()):
            if cache_key == target_cache_key:
                continue
            stale_keys.append(cache_key)
            actor = _pop_vllm_worker_unlocked(cache_key)
            if actor is not None:
                actors_to_shutdown.append(actor)
        if _model_cache:
            print(
                "[model_runner] Clearing HF model cache before Ray vLLM model switch: "
                f"target_model={target_cache_key}"
            )
            _clear_hf_cache_unlocked()
        if _vllm_cache:
            print(
                "[model_runner] Clearing in-process vLLM engine before Ray vLLM model switch: "
                f"target_model={target_cache_key}"
            )
            _clear_vllm_cache_unlocked()

    for cache_key in stale_keys:
        print(
            "[model_runner] Clearing stale Ray vLLM actor before model switch: "
            f"model_path={cache_key} actor_name={_get_ray_actor_name(cache_key)} "
            f"target_model={target_cache_key}"
        )
    if previous_cache_key and previous_cache_key != target_cache_key and not actors_to_shutdown:
        try:
            ray = _ensure_ray_initialized()
            actor = ray.get_actor(_get_ray_actor_name(previous_cache_key))
        except Exception:
            pass
        else:
            actors_to_shutdown.append(actor)
    return actors_to_shutdown


def _shutdown_ray_vllm_actor_for_switch(actor: Any) -> None:
    _shutdown_ray_vllm_actor(actor)
    _collect_cuda()


def _get_ray_vllm_model_switch_guard() -> RayVllmModelSwitchGuard:
    global _ray_vllm_model_switch_guard
    if _ray_vllm_model_switch_guard is None:
        _ray_vllm_model_switch_guard = RayVllmModelSwitchGuard(
            clear_stale=_clear_stale_ray_vllm_workers_for_switch,
            shutdown_actors=_shutdown_ray_vllm_actor_for_switch,
        )
    return _ray_vllm_model_switch_guard


def clear_non_champion_vllm_actors(
    champion_model_path: str,
    *,
    candidate_model_path: str = "",
    previous_champion_model_path: str = "",
) -> None:
    """Kill Ray vLLM actors that are no longer the active champion.

    Detached named actors can outlive a round. After strategy inspection, keep
    only the actor for the current champion and kill known stale candidate / old
    champion actors so they do not hold GPU memory into the next round.
    """

    keep_key = _normalize_model_id(champion_model_path) if champion_model_path else ""
    stale_keys = set(_vllm_worker_cache.keys())
    for model_path in (candidate_model_path, previous_champion_model_path):
        if model_path:
            stale_keys.add(_normalize_model_id(model_path))

    stale_keys.discard(keep_key)
    if not stale_keys:
        return

    actors_to_shutdown: list[Any] = []
    with _cache_lock:
        for cache_key in sorted(stale_keys):
            print(
                "[model_runner] Clearing non-champion Ray vLLM actor after decision: "
                f"model_path={cache_key} actor_name={_get_ray_actor_name(cache_key)} "
                f"kept_champion={keep_key or '<none>'}"
            )
            actor = _pop_vllm_worker_unlocked(cache_key)
            if actor is not None:
                actors_to_shutdown.append(actor)
    _shutdown_popped_vllm_workers(actors_to_shutdown)

    for cache_key in sorted(stale_keys):
        if cache_key == keep_key:
            continue
        with _cache_lock:
            cached = cache_key in _vllm_worker_cache
        if cached:
            continue
        try:
            ray = _ensure_ray_initialized()
            actor = ray.get_actor(_get_ray_actor_name(cache_key))
        except Exception:
            continue
        _shutdown_popped_vllm_workers([actor])


def _collect_cuda() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def release_model_runner_gpu_resources(*, reason: str = "external GPU workload") -> None:
    """Release cached model_runner GPU holders before an external evaluator runs."""
    actors_to_shutdown: list[Any] = []
    with _cache_lock:
        if _model_cache:
            print(f"[model_runner] Clearing HF model cache before {reason}")
            _clear_hf_cache_unlocked()
        if _vllm_cache:
            print(f"[model_runner] Clearing in-process vLLM engine before {reason}")
            _clear_vllm_cache_unlocked()
        if _vllm_worker_cache:
            print(f"[model_runner] Clearing Ray vLLM workers before {reason}")
            actors_to_shutdown.extend(_clear_vllm_workers_unlocked())
    _shutdown_popped_vllm_workers(actors_to_shutdown)
    _collect_cuda()


def _mark_vllm_disabled(model_path: str, reason: str) -> None:
    cache_key = _normalize_model_id(model_path)
    with _cache_lock:
        _vllm_disabled_models[cache_key] = reason


def _vllm_disabled_reason(model_path: str) -> str:
    cache_key = _normalize_model_id(model_path)
    with _cache_lock:
        return _vllm_disabled_models.get(cache_key, "")


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")


def _disable_hf_fallback_after_vllm_memory_error() -> bool:
    return _env_truthy("DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR", default=True)


def _run_local_checkpoint_batch_subprocess(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    disable_thinking: bool = False,
    stop_after_json: bool = False,
    timeout_seconds: int | None = None,
) -> list[str]:
    payload = {
        "model_path": model_path,
        "prompts": prompts,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "disable_thinking": disable_thinking,
        "stop_after_json": stop_after_json,
    }
    env = os.environ.copy()
    env["MODEL_RUNNER_ISOLATED_CHILD"] = "1"
    # The child is already isolated from the training/evaluator parent process,
    # so use the simple in-process vLLM path there instead of starting/reusing a
    # detached Ray actor that may outlive the child and hold GPU memory.
    env["MODEL_RUNNER_DISABLE_RAY_VLLM"] = "1"
    env["USE_VLLM"] = "1"
    env["USE_VLLM_FOR_LOCAL_CHECKPOINTS"] = "1"
    command = [sys.executable, "-m", "src.tools.model_runner", "--batch-json"]
    timeout = timeout_seconds or int(os.environ.get("VLLM_LOCAL_CHECKPOINT_SUBPROCESS_TIMEOUT_SECONDS", "3600"))
    input_text = json.dumps(payload, ensure_ascii=False)
    start_new_session = hasattr(os, "setsid")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=start_new_session,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(process)
        raise IsolatedInferenceError(
            f"isolated local checkpoint inference timed out after {timeout}s"
        ) from exc

    if process.returncode != 0:
        stderr = (stderr or "").strip()
        stdout = (stdout or "").strip()
        signal_hint = f"signal {-process.returncode}" if process.returncode < 0 else f"exit code {process.returncode}"
        response_error = ""
        if stdout:
            try:
                response = _extract_batch_json_response(stdout)
                if not response.get("ok"):
                    response_error = str(response.get("error") or "unknown isolated inference error")
            except IsolatedInferenceError:
                response_error = ""
        detail = response_error or stderr or stdout or signal_hint
        raise IsolatedInferenceError(
            f"isolated local checkpoint inference failed ({signal_hint}): {detail[-2000:]}"
        )

    response = _extract_batch_json_response(stdout or "")

    if not response.get("ok"):
        raise IsolatedInferenceError(str(response.get("error") or "unknown isolated inference error"))
    results = response.get("results")
    if not isinstance(results, list) or len(results) != len(prompts):
        raise IsolatedInferenceError(
            f"isolated local checkpoint inference returned {type(results).__name__} "
            f"with length {len(results) if isinstance(results, list) else 'n/a'} for {len(prompts)} prompts"
        )
    return [str(item) for item in results]


def _kill_process_tree(process: subprocess.Popen) -> None:
    try:
        if process.poll() is not None:
            return
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    finally:
        try:
            process.wait(timeout=10)
        except Exception:
            pass


def _normalize_model_id(model_id: str) -> str:
    """Normalize any model ID (HuggingFace ID or local path) to resolved snapshot path.

    Ensures model/cache keys are consistent regardless of whether callers pass
    a repository ID, a snapshot directory, a symlink, or an equivalent local path.
    """
    if not model_id:
        return model_id
    model_path = Path(model_id).expanduser()
    if model_path.is_dir():
        return str(Path(os.path.realpath(model_path)).resolve())
    try:
        resolved = resolve_model_path(model_id)
        resolved_path = Path(resolved).expanduser()
        if resolved_path.is_dir():
            return str(Path(os.path.realpath(resolved_path)).resolve())
    except Exception:
        pass
    return model_id


def _is_offline_mode() -> bool:
    return (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    )


def _load_base_model(model_id: str, kwargs: dict):
    """Load a fresh base model instance from disk.

    Always calls from_pretrained to create a new model object.
    This is essential for LoRA: PeftModel.from_pretrained wraps the base
    model in-place, so we must not reuse a cached instance.
    """
    _require_inference_deps()
    resolved = resolve_model_path(model_id)
    gpu_kwargs = {k: v for k, v in kwargs.items() if k != "low_cpu_mem_usage"}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, local_files_only=True, **gpu_kwargs
        )
        print(f"[model_runner] Loaded base model: {resolved}")
        return model
    except Exception as e:
        is_meta = "meta tensor" in str(e).lower() or "to_empty" in str(e).lower()
        if not is_meta:
            print(f"[model_runner] Base model load failed ({type(e).__name__}): {e}")
        else:
            print(f"[model_runner] Meta-tensor error on first attempt, clearing cache and retrying...")
            gc.collect()
            _cuda_empty_cache()
            _cuda_synchronize()
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    resolved, local_files_only=True, **gpu_kwargs
                )
                print(f"[model_runner] Loaded base model (retry): {resolved}")
                return model
            except Exception as e_retry:
                print(f"[model_runner] Retry also failed ({type(e_retry).__name__}): {e_retry}")

    if _is_offline_mode():
        print(f"[model_runner] Cannot load {resolved} (offline mode)")
        return None

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **gpu_kwargs)
        print(f"[model_runner] Loaded base model (online): {model_id}")
        return model
    except Exception as e:
        print(f"[model_runner] Online load also failed for {model_id}: {type(e).__name__}: {e}")
        return None


def _load_tokenizer(model_id: str, local_files_only: bool = False):
    _require_inference_deps()
    resolved = resolve_model_path(model_id)
    try:
        return AutoTokenizer.from_pretrained(
            resolved, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        if local_files_only or _is_offline_mode():
            raise
    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"[model_runner] Tokenizer load failed for {model_id}: {type(e).__name__}: {e}")
        raise


def _get_lora_base_model(adapter_path: str) -> str:
    """Read base_model_name_or_path from adapter_config.json."""
    adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
    if adapter_cfg_path.exists():
        try:
            cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
            base = cfg.get("base_model_name_or_path", "")
            if base:
                return _validate_lora_base_model_ref(str(base))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[model_runner] Could not read adapter_config.json: {e}")

    from config.settings import BASE_MODEL_NAME

    base = os.environ.get("BASE_MODEL_NAME", BASE_MODEL_NAME).strip()
    if not base:
        raise RuntimeError(
            "No base_model_name_or_path in adapter_config.json and BASE_MODEL_NAME is not configured"
        )
    print("[model_runner] No base_model_name_or_path in adapter_config.json, using configured BASE_MODEL_NAME")
    return _validate_lora_base_model_ref(base)


def _validate_lora_base_model_ref(base_model_ref: str) -> str:
    """Reject adapter-provided base refs that can escape via local paths."""
    ref = base_model_ref.strip()
    if not ref or "\x00" in ref:
        raise ValueError("Unsafe LoRA base_model_name_or_path in adapter_config.json")
    normalized_parts = ref.replace("\\", "/").split("/")
    if any(part == ".." for part in normalized_parts):
        raise ValueError("Unsafe LoRA base_model_name_or_path in adapter_config.json")
    if not Path(ref).is_absolute() and ref[0] in {".", "~"}:
        raise ValueError("Unsafe LoRA base_model_name_or_path in adapter_config.json")
    return ref


def _is_lora_adapter_path(model_path: str) -> bool:
    return Path(model_path).is_dir() and (Path(model_path) / "adapter_config.json").exists()


def _load_model(model_path: str):
    """Load model and tokenizer with thread-safe caching.

    For base models: loads once, caches, reuses on subsequent calls.
    For LoRA adapters: deep-copies the cached base model so PeftModel
    can wrap it without corrupting the original.
    """
    cache_key = _normalize_model_id(model_path)

    should_wait = False

    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

        if cache_key in _cache_loading:
            event = _cache_loading[cache_key]
            should_wait = True
        else:
            event = threading.Event()
            _cache_loading[cache_key] = event

    if should_wait:
        event.wait(timeout=300)

        with _cache_lock:
            if cache_key in _model_cache:
                return _model_cache[cache_key]

    try:
        result = _load_model_inner(model_path, cache_key)
    except Exception:
        with _cache_lock:
            _cache_loading.pop(cache_key, None)
        raise

    with _cache_lock:
        _model_cache[cache_key] = result
        event = _cache_loading.pop(cache_key, None)

    if event:
        event.set()

    return result


def _load_model_inner(model_path: str, cache_key: str):
    _require_inference_deps()
    device = "cuda" if _cuda_available() else "cpu"
    dtype = torch.bfloat16 if _cuda_available() else torch.float32

    adapter_cfg = Path(model_path) / "adapter_config.json"
    is_lora = adapter_cfg.exists() if Path(model_path).is_dir() else False

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    )

    if is_lora:
        from peft import PeftModel

        base_model_id = _get_lora_base_model(model_path)
        base_cache_key = _normalize_model_id(base_model_id)

        with _cache_lock:
            cached_base = _model_cache.get(base_cache_key)

        if cached_base is not None:
            cached_model, cached_tokenizer = cached_base
            print(f"[model_runner] Deep-copying cached base model for LoRA: {base_cache_key}")
            base = copy.deepcopy(cached_model)
        else:
            print(f"[model_runner] Cache miss for base model, loading fresh: {base_model_id}")
            base = _load_base_model(base_model_id, load_kwargs)

        if base is None:
            print(f"[model_runner] LoRA base model load failed, falling back to cached base")
            with _cache_lock:
                fallback = _model_cache.get(base_cache_key)
            if fallback:
                return fallback
            raise RuntimeError(f"Cannot load base model {base_model_id} for LoRA {model_path}")

        try:
            model = PeftModel.from_pretrained(base, model_path, device_map=device)
        except Exception as e:
            print(f"[model_runner] PeftModel.from_pretrained failed: {type(e).__name__}: {e}")
            del base
            gc.collect()
            _cuda_empty_cache()
            with _cache_lock:
                fallback = _model_cache.get(base_cache_key)
            if fallback:
                print(f"[model_runner] Falling back to cached base model for inference")
                return fallback
            raise

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
        except Exception:
            tokenizer = _load_tokenizer(base_model_id)
    else:
        model = _load_base_model(model_path, load_kwargs)
        if model is None:
            raise RuntimeError(f"Cannot load model {model_path}")
        tokenizer = _load_tokenizer(model_path)

    model.eval()
    return model, tokenizer


def warmup_model(model_id: str):
    """Pre-load a model into cache before parallel workers need it.

    Call this during bootstrap to avoid concurrent loading races.
    """
    try:
        from config.settings import USE_VLLM
        if USE_VLLM:
            try:
                _get_vllm_worker(model_id)
                print(f"[model_runner] Ray vLLM warmup complete: {model_id}")
            except ModuleNotFoundError as exc:
                if exc.name not in ("vllm", "ray"):
                    raise
                print(f"[model_runner] vLLM/Ray unavailable; skipping warmup for {model_id}")
            return
    except Exception:
        pass

    cache_key = _normalize_model_id(model_id)
    with _cache_lock:
        if cache_key in _model_cache:
            return

    device = "cuda" if _cuda_available() else "cpu"
    dtype = torch.bfloat16 if _cuda_available() else torch.float32
    load_kwargs = dict(trust_remote_code=True, torch_dtype=dtype, device_map=device)

    resolved = resolve_model_path(model_id)
    model = _load_base_model(model_id, load_kwargs)
    if model is None:
        print(f"[model_runner] warmup_model failed for {model_id}")
        return

    try:
        tokenizer = _load_tokenizer(model_id)
    except Exception:
        print(f"[model_runner] warmup_model tokenizer failed for {model_id}")
        del model
        gc.collect()
        return

    model.eval()
    cache_key = _normalize_model_id(model_id)
    with _cache_lock:
        _model_cache[cache_key] = (model, tokenizer)
    print(f"[model_runner] Warmup complete: {model_id} -> {cache_key}")


def _generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    top_p: float = 1.0,
    disable_thinking: bool = False,
    stop_after_json: bool = False,
) -> list[str]:
    """Internal: batch generate for a list of prompts on the same model.

    Tokenizes all prompts together, pads to equal length, and generates
    in a single forward pass for much higher GPU utilisation.
    Falls back to sequential generation if batch fails.
    """
    if not prompts:
        return []
    _require_inference_deps()

    all_messages = [[{"role": "user", "content": p}] for p in prompts]
    texts = [
        _apply_chat_template(tokenizer, msgs, disable_thinking=disable_thinking)
        for msgs in all_messages
    ]

    # Use safe prompt token budget instead of hardcoded max_length=2048
    prompt_budget = _get_safe_prompt_token_budget(
        max_new_tokens=max_new_tokens,
    )
    texts = _pre_tokenize_texts(tokenizer, texts, prompt_budget)

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=prompt_budget
    ).to(model.device)

    input_lengths = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0
    with torch.no_grad():
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        if stop_after_json:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([
                _JsonObjectStoppingCriteria(tokenizer, input_lengths)
            ])
        outputs = model.generate(**generate_kwargs)

    results = []
    for i, output in enumerate(outputs):
        generated_ids = output[input_lengths:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(decoded.strip())

    return results


def _get_vllm_engine(model_path: str):
    from config.settings import (
        VLLM_ENABLE_PREFIX_CACHING,
        VLLM_ENFORCE_EAGER,
        VLLM_GPU_MEMORY_UTILIZATION,
        VLLM_MAX_CACHED_ENGINES,
        VLLM_MAX_NUM_BATCHED_TOKENS,
        VLLM_MAX_NUM_SEQS,
        VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION,
    )

    cache_key = _normalize_model_id(model_path)
    max_cached = max(1, int(VLLM_MAX_CACHED_ENGINES or 1))
    if VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION > 0:
        gpu_memory_utilization = min(0.95, float(VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION))
    elif max_cached > 1:
        gpu_memory_utilization = max(
            0.20,
            min(0.95, float(VLLM_GPU_MEMORY_UTILIZATION) / max_cached),
        )
    else:
        gpu_memory_utilization = min(0.95, float(VLLM_GPU_MEMORY_UTILIZATION))
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
    scheduler_kwargs = {
        "max_num_seqs": max(1, int(VLLM_MAX_NUM_SEQS)),
        "max_num_batched_tokens": max(max_model_len, int(VLLM_MAX_NUM_BATCHED_TOKENS)),
    }

    # LangGraph Send fan-out runs rollout workers concurrently in threads.
    # Protect the full cache-miss -> LLM(...) -> cache-populate path so those
    # threads share one engine instead of racing several 82%-GPU allocations.
    with _get_vllm_generation_lock(cache_key):
        workers_to_shutdown: list[Any] = []
        with _cache_lock:
            if cache_key in _vllm_cache:
                _vllm_cache.move_to_end(cache_key)
                return _vllm_cache[cache_key]
            if _model_cache:
                print(f"[model_runner] Clearing HF model cache before vLLM load")
                _clear_hf_cache_unlocked()
            if _vllm_worker_cache:
                # 机制层清理：本地 checkpoint worker 和 agent 底模引擎不能同时占用同一张 GPU。
                print("[model_runner] Clearing local vLLM workers before in-process vLLM load")
                workers_to_shutdown.extend(_clear_vllm_workers_unlocked())
            while len(_vllm_cache) >= max_cached:
                evicted_key, _ = next(iter(_vllm_cache.items()))
                print(f"[model_runner] Evicting vLLM engine by LRU: {evicted_key}")
                _evict_vllm_engine_unlocked(evicted_key)

        _shutdown_popped_vllm_workers(workers_to_shutdown)
        _collect_cuda()

        from vllm import LLM
        if _callable_accepts_keyword(LLM, "enable_chunked_prefill"):
            scheduler_kwargs["enable_chunked_prefill"] = _vllm_chunked_prefill_enabled()

        resolved = _normalize_model_id(model_path)
        print(
            f"[model_runner] Loading vLLM engine: {resolved} "
            f"(gpu_memory_utilization={gpu_memory_utilization:.3f}, "
            f"max_cached={max_cached}, enforce_eager={VLLM_ENFORCE_EAGER})"
        )

        # Try loading the vLLM engine. On GPU OOM, evict all cached engines
        # and retry with full memory budget so the new engine can fit.
        try:
            llm = LLM(
                model=resolved,
                trust_remote_code=True,
                dtype="bfloat16" if _cuda_available() else "float32",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=VLLM_ENFORCE_EAGER,
                enable_prefix_caching=VLLM_ENABLE_PREFIX_CACHING,
                **scheduler_kwargs,
            )
        except Exception as e:
            if not _is_vllm_memory_error(e):
                raise

            print(
                f"[model_runner] GPU OOM while loading vLLM engine for {resolved} "
                f"(gpu_memory_utilization={gpu_memory_utilization:.3f}), "
                f"evicting all cached engines and retrying with full budget"
            )
            workers_to_shutdown = []
            with _cache_lock:
                _clear_vllm_cache_unlocked()
                workers_to_shutdown.extend(_clear_vllm_workers_unlocked())
            _shutdown_popped_vllm_workers(workers_to_shutdown)
            _collect_cuda()

            # Use the full GPU memory budget on retry since we are now the only engine
            retry_gpu_util = min(0.95, float(VLLM_GPU_MEMORY_UTILIZATION))
            print(
                f"[model_runner] Retrying vLLM engine load: {resolved} "
                f"(gpu_memory_utilization={retry_gpu_util:.3f})"
            )
            try:
                llm = LLM(
                    model=resolved,
                    trust_remote_code=True,
                    dtype="bfloat16" if _cuda_available() else "float32",
                    gpu_memory_utilization=retry_gpu_util,
                    max_model_len=max_model_len,
                    enforce_eager=VLLM_ENFORCE_EAGER,
                    enable_prefix_caching=VLLM_ENABLE_PREFIX_CACHING,
                    **scheduler_kwargs,
                )
            except Exception as retry_error:
                raise VllmEngineLoadError(
                    f"vLLM engine load failed after retry for {resolved}: {retry_error}"
                ) from retry_error
            gpu_memory_utilization = retry_gpu_util

        tokenizer = llm.get_tokenizer()
        with _cache_lock:
            _vllm_cache[cache_key] = (llm, tokenizer)
            _vllm_cache.move_to_end(cache_key)
        print(
            f"[model_runner] Loaded vLLM engine: {resolved} "
            f"(gpu_memory_utilization={gpu_memory_utilization:.3f}, max_cached={max_cached})"
        )
        return llm, tokenizer


def _vllm_worker_request_id(model_path: str) -> str:
    return f"{os.getpid()}:{threading.get_ident()}:{time.time_ns()}:{abs(hash(model_path))}"


def _ray_get_with_timeout(object_ref, timeout: int | None = None):
    ray = _ensure_ray_initialized()
    if timeout is None:
        timeout = int(os.environ.get("VLLM_WORKER_READY_TIMEOUT", "300"))
    return ray.get(object_ref, timeout=timeout)


def _is_ray_actor_died_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    module = type(exc).__module__
    if name in {"ActorDiedError", "RayActorError"} and module.startswith("ray."):
        return True
    message = str(exc).lower()
    return (
        "actordiederror" in message
        or ("actor died" in message and "ray" in message)
        or ("actor" in message and "ray.kill" in message)
    )


def _is_ray_get_timeout_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    module = type(exc).__module__
    if name == "GetTimeoutError" and module.startswith("ray."):
        return True
    return "get timed out" in str(exc).lower()


def _is_vllm_timeout_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return _is_ray_get_timeout_error(exc) or (
        "vllm" in message
        and "timed out" in message
    )


def _is_vllm_gpu_unsafe_error(exc: BaseException) -> bool:
    return (
        _is_vllm_memory_error(exc)
        or _is_vllm_timeout_error(exc)
        or _is_ray_actor_died_error(exc)
    )


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return max(1, default)
    return max(1, value)


def _nonnegative_float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return max(0.0, default)
    return max(0.0, value)


def _vllm_worker_generate_timeout_seconds(prompt_count: int) -> int:
    base_timeout = _positive_int_env("VLLM_RAY_GENERATE_TIMEOUT_SECONDS", 1800)
    per_prompt_timeout = _nonnegative_float_env("VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS", 3.0)
    scaled_timeout = int(max(0, prompt_count) * per_prompt_timeout)
    return max(1, base_timeout, scaled_timeout)


def _get_vllm_worker(model_path: str):
    from config.settings import (
        VLLM_ENABLE_PREFIX_CACHING,
        VLLM_ENFORCE_EAGER,
        VLLM_GPU_MEMORY_UTILIZATION,
        VLLM_MAX_CACHED_ENGINES,
        VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION,
        VLLM_RAY_ACTOR_MAX_CONCURRENCY,
        VLLM_RAY_ACTOR_MAX_RESTARTS,
        VLLM_RAY_ACTOR_MAX_TASK_RETRIES,
        VLLM_RAY_ACTOR_NUM_GPUS,
    )

    cache_key = _normalize_model_id(model_path)
    max_cached = max(1, int(VLLM_MAX_CACHED_ENGINES or 1))
    gpu_memory_utilization = (
        min(0.95, float(VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION))
        if VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION > 0
        else min(0.95, float(VLLM_GPU_MEMORY_UTILIZATION))
    )

    resolved = _normalize_model_id(model_path)
    actor_name = _get_ray_actor_name(resolved)
    local_checkpoint = Path(resolved).is_dir() and (Path(resolved) / "config.json").exists()

    def cached_ready_worker():
        actor = _vllm_worker_cache.get(cache_key)
        if actor is None:
            return None
        # A cached actor can be busy serving a long generate request from another
        # rollout worker. Probing ready() here competes with that workload and a
        # timeout used to kill the shared detached actor, aborting in-flight
        # callers. Trust cached handles; generation failures evict the local
        # cache entry without destroying the named actor.
        _vllm_worker_cache.move_to_end(cache_key)
        return actor

    def get_named_actor():
        ray = _ensure_ray_initialized()
        try:
            return ray.get_actor(actor_name)
        except Exception:
            return None

    with _cache_lock:
        cached = cached_ready_worker()
        if cached is not None:
            return cached

    start_lock = _get_vllm_worker_start_lock(cache_key)
    with start_lock:
        with _cache_lock:
            cached = cached_ready_worker()
            if cached is not None:
                return cached

        # vLLM EngineCore initialization profiles and allocates GPU memory.
        # Serialize first-time starts across LangGraph worker threads so they
        # reuse one Ray actor instead of racing several GPU allocations.
        with _vllm_worker_global_start_lock:
            with _cache_lock:
                cached = cached_ready_worker()
                if cached is not None:
                    return cached
            named_actor = get_named_actor()
            if named_actor is not None:
                try:
                    response = _ray_get_with_timeout(named_actor.ready.remote(), timeout=30)
                    if not response.get("ok"):
                        raise VllmEngineLoadError(f"Ray vLLM actor readiness check failed: {response}")
                except Exception as exc:
                    if _is_ray_get_timeout_error(exc):
                        print(
                            "[model_runner] Ray vLLM named actor readiness check failed with timeout; "
                            f"model_path={resolved} actor_name={actor_name} "
                            f"error={type(exc).__name__}: {exc}; reusing named actor without killing it"
                        )
                    else:
                        print(
                            "[model_runner] Ray vLLM named actor readiness check failed; "
                            f"model_path={resolved} actor_name={actor_name} "
                            f"error={type(exc).__name__}: {exc}; not reusing named actor"
                        )
                        try:
                            _shutdown_ray_vllm_actor(named_actor)
                        finally:
                            _collect_cuda()
                        named_actor = None
            if named_actor is not None:
                with _cache_lock:
                    _vllm_worker_cache[cache_key] = named_actor
                    _vllm_worker_cache.move_to_end(cache_key)
                print(
                    "[model_runner] Ray vLLM actor ready: "
                    f"model_path={resolved} actor_name={actor_name} "
                    f"reused_named_actor=true local_checkpoint={str(local_checkpoint).lower()}"
                )
                return named_actor

            workers_to_shutdown: list[Any] = []
            with _cache_lock:
                if _model_cache:
                    print("[model_runner] Clearing HF model cache before Ray vLLM actor load")
                    _clear_hf_cache_unlocked()
                if _vllm_cache:
                    print("[model_runner] Clearing in-process vLLM engine before Ray vLLM actor load")
                    _clear_vllm_cache_unlocked()
                while len(_vllm_worker_cache) >= max_cached:
                    evicted_key, _ = next(iter(_vllm_worker_cache.items()))
                    print(f"[model_runner] Evicting Ray vLLM actor by LRU: {evicted_key}")
                    actor_to_shutdown = _pop_vllm_worker_unlocked(evicted_key)
                    if actor_to_shutdown is not None:
                        workers_to_shutdown.append(actor_to_shutdown)

            _shutdown_popped_vllm_workers(workers_to_shutdown)
            _collect_cuda()
            _shutdown_legacy_named_vllm_worker(resolved)

            print(
                f"[model_runner] Starting Ray vLLM actor: {resolved} "
                f"(gpu_memory_utilization={gpu_memory_utilization:.3f}, "
                f"max_model_len={os.environ.get('VLLM_MAX_MODEL_LEN', '4096')}, "
                f"num_gpus={VLLM_RAY_ACTOR_NUM_GPUS}, "
                f"enforce_eager={VLLM_ENFORCE_EAGER}, "
                f"enable_prefix_caching={VLLM_ENABLE_PREFIX_CACHING})"
            )
            actor = None
            try:
                from src.tools.ray_vllm_actor import get_ray_actor_class

                actor_cls = get_ray_actor_class(
                    num_gpus=float(VLLM_RAY_ACTOR_NUM_GPUS),
                    max_restarts=int(VLLM_RAY_ACTOR_MAX_RESTARTS),
                    max_task_retries=int(VLLM_RAY_ACTOR_MAX_TASK_RETRIES),
                    max_concurrency=int(VLLM_RAY_ACTOR_MAX_CONCURRENCY),
                )
                actor = actor_cls.options(
                    name=actor_name,
                    get_if_exists=True,
                    lifetime="detached",
                ).remote(
                    model_path=resolved,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")),
                    enforce_eager=VLLM_ENFORCE_EAGER,
                    enable_prefix_caching=VLLM_ENABLE_PREFIX_CACHING,
                )
                response = _ray_get_with_timeout(actor.ready.remote())
                if not response.get("ok"):
                    raise VllmEngineLoadError(f"Ray vLLM actor readiness check failed: {response}")
            except Exception as exc:
                if actor is not None:
                    try:
                        _shutdown_ray_vllm_actor(actor)
                    except Exception:
                        pass
                else:
                    named_actor = get_named_actor()
                    if named_actor is not None:
                        try:
                            _shutdown_ray_vllm_actor(named_actor)
                        except Exception:
                            pass
                _collect_cuda()
                raise VllmEngineLoadError(f"Ray vLLM actor failed to start for {resolved}: {exc}") from exc

            with _cache_lock:
                _vllm_worker_cache[cache_key] = actor
                _vllm_worker_cache.move_to_end(cache_key)
            print(
                "[model_runner] Ray vLLM actor ready: "
                f"model_path={resolved} actor_name={actor_name} "
                f"reused_named_actor=false local_checkpoint={str(local_checkpoint).lower()}"
            )
            return actor


def _run_model_batch_vllm_worker(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    disable_thinking: bool = False,
    stop_after_json: bool = False,
) -> list[str]:
    if _env_truthy("MODEL_RUNNER_DISABLE_RAY_VLLM", default=False):
        return _run_model_batch_vllm(
            model_path,
            prompts,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            disable_thinking=disable_thinking,
            stop_after_json=stop_after_json,
        )

    cache_key = _normalize_model_id(model_path)

    prompt_budget = _get_safe_prompt_token_budget(
        max_new_tokens=max_new_tokens,
    )

    def send_batch() -> dict:
        actor = _get_vllm_worker(model_path)
        timeout = _vllm_worker_generate_timeout_seconds(len(prompts))
        return _ray_get_with_timeout(
            actor.generate.remote({
                "op": "generate",
                "request_id": _vllm_worker_request_id(model_path),
                "prompts": prompts,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "disable_thinking": disable_thinking,
                "stop_after_json": stop_after_json,
                "max_prompt_tokens": prompt_budget,
            }),
            timeout=timeout,
        )

    _local_checkpoint_gpu_gate.acquire_shared()
    switch_guard = _get_ray_vllm_model_switch_guard()
    model_guard_acquired = False
    try:
        switch_guard.acquire_model(cache_key)
        model_guard_acquired = True
        try:
            response = send_batch()
        except Exception as exc:
            if _is_vllm_memory_error(exc):
                print(
                    "[model_runner] Ray vLLM actor reported memory failure during batch; "
                    f"model_path={cache_key} actor_name={_get_ray_actor_name(cache_key)} "
                    f"error={type(exc).__name__}: {exc}; shutting down actor without same-GPU retry"
                )
                _evict_vllm_worker(cache_key)
                raise
            if _is_ray_get_timeout_error(exc):
                print(
                    "[model_runner] Ray vLLM actor timed out during batch; "
                    f"model_path={cache_key} actor_name={_get_ray_actor_name(cache_key)} "
                    f"prompts={len(prompts)} timeout={_vllm_worker_generate_timeout_seconds(len(prompts))}s "
                    f"error={type(exc).__name__}: {exc}; shutting down actor without same-GPU HF fallback"
                )
                _evict_vllm_worker(cache_key)
                raise VllmEngineLoadError(
                    f"Ray vLLM actor timed out after {_vllm_worker_generate_timeout_seconds(len(prompts))}s "
                    f"for {len(prompts)} prompts: {exc}"
                ) from exc
            if not _is_ray_actor_died_error(exc):
                raise
            print(
                "[model_runner] Ray vLLM actor died during batch; "
                f"model_path={cache_key} actor_name={_get_ray_actor_name(cache_key)} "
                f"error={type(exc).__name__}: {exc}; retrying once after actor rebuild"
            )
            _evict_vllm_worker(cache_key)
            try:
                response = send_batch()
            except Exception as retry_exc:
                print(
                    "[model_runner] Ray vLLM actor retry failed after rebuild; "
                    f"model_path={cache_key} actor_name={_get_ray_actor_name(cache_key)} "
                    f"error={type(retry_exc).__name__}: {retry_exc}"
                )
                if _is_ray_actor_died_error(retry_exc) or _is_ray_get_timeout_error(retry_exc) or _is_vllm_memory_error(retry_exc):
                    _evict_vllm_worker(cache_key)
                if _is_ray_actor_died_error(retry_exc):
                    raise VllmEngineLoadError(
                        f"Ray vLLM actor died after retry for {len(prompts)} prompts: {retry_exc}"
                    ) from retry_exc
                if _is_ray_get_timeout_error(retry_exc):
                    raise VllmEngineLoadError(
                        f"Ray vLLM actor timed out after retry for {len(prompts)} prompts: {retry_exc}"
                    ) from retry_exc
                raise

        if not response.get("ok"):
            error = RuntimeError(response.get("error", "Ray vLLM actor request failed"))
            if _is_vllm_memory_error(error):
                print(
                    "[model_runner] Ray vLLM actor returned memory failure; "
                    f"model_path={cache_key} actor_name={_get_ray_actor_name(cache_key)} "
                    "shutting down actor without same-GPU retry"
                )
                _evict_vllm_worker(cache_key)
            raise error
        results: list[str] = []
        results.extend(str(item) for item in response.get("results", []))
        return results
    finally:
        try:
            if model_guard_acquired:
                switch_guard.release_model()
        finally:
            _local_checkpoint_gpu_gate.release_shared()


def _run_model_batch_vllm(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    disable_thinking: bool = False,
    stop_after_json: bool = False,
) -> list[str]:
    from vllm import SamplingParams
    cache_key = _normalize_model_id(model_path)

    def generate_once():
        # vLLM's in-process LLM.generate is not safe for concurrent calls on
        # the same engine. LangGraph Send fan-out runs rollout workers in
        # parallel threads, so serialize engine load/use while keeping the
        # graph-level worker fan-out intact.
        with _get_vllm_generation_lock(cache_key):
            llm, tokenizer = _get_vllm_engine(model_path)

            # Compute safe prompt token budget and pre-truncate if needed
            prompt_budget = _get_safe_prompt_token_budget(
                max_new_tokens=max_new_tokens,
            )
            sampling_params = _build_sampling_params(
                SamplingParams,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                stop=["```"] if stop_after_json else None,
                prompt_budget=prompt_budget,
            )
            all_outputs = []
            for batch in _batched(prompts, _get_inference_batch_size()):
                texts = [
                    _apply_chat_template(
                        tokenizer,
                        [{"role": "user", "content": p}],
                        disable_thinking=disable_thinking,
                    )
                    for p in batch
                ]
                texts = _pre_tokenize_texts(tokenizer, texts, prompt_budget)
                all_outputs.extend(llm.generate(texts, sampling_params, use_tqdm=False))
            return all_outputs

    try:
        outputs = generate_once()
    except Exception as e:
        if isinstance(e, VllmEngineLoadError):
            raise
        if not _is_vllm_memory_error(e):
            raise
        print(
            f"[model_runner] vLLM generation failed after engine load "
            f"({type(e).__name__}: {e}); clearing engines and retrying once"
        )
        with _cache_lock:
            _clear_vllm_cache_unlocked()
        _collect_cuda()
        outputs = generate_once()

    results = []
    for output in outputs:
        if output.outputs:
            results.append(output.outputs[0].text.strip())
        else:
            results.append("")
    return results


def run_model_once(
    model_path: str,
    prompt: str,
    max_new_tokens: int | None = None,
    prepend_math_instruction: bool = True,
    disable_thinking: bool = False,
    stop_after_json: bool = False,
) -> str:
    """Run model inference on a single prompt and return generated text.

    max_new_tokens defaults to INFERENCE_MAX_NEW_TOKENS from settings (96).
    """
    _require_inference_deps()
    from config.settings import INFERENCE_MAX_NEW_TOKENS, USE_VLLM

    if prepend_math_instruction:
        prompt = _get_instruction_prefix() + prompt

    if max_new_tokens is None:
        max_new_tokens = INFERENCE_MAX_NEW_TOKENS

    if not model_path:
        return ""

    t0 = time.time()
    if USE_VLLM:
        workers_to_shutdown: list[Any] = []
        with _cache_lock:
            if _vllm_cache:
                print(f"[model_runner] Clearing vLLM engine before HF fallback")
                _clear_vllm_cache_unlocked()
            if _vllm_worker_cache:
                print("[model_runner] Clearing local vLLM workers before HF fallback")
                workers_to_shutdown.extend(_clear_vllm_workers_unlocked())
        _shutdown_popped_vllm_workers(workers_to_shutdown)
        _collect_cuda()

    try:
        model, tokenizer = _load_model(model_path)
    except Exception as e:
        print(f"[model_runner] Failed to load model {model_path}: {e}")
        return ""

    try:
        messages = [{"role": "user", "content": prompt}]
        text = _apply_chat_template(tokenizer, messages, disable_thinking=disable_thinking)
        prompt_budget = _get_safe_prompt_token_budget(max_new_tokens=max_new_tokens)
        text = _pre_tokenize_texts(tokenizer, [text], prompt_budget)[0]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_budget,
        ).to(model.device)

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if stop_after_json:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([
                _JsonObjectStoppingCriteria(tokenizer, inputs["input_ids"].shape[1])
            ])

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        elapsed = time.time() - t0
        print(f"[model_runner] run_model_once completed in {elapsed:.2f}s (tokens={max_new_tokens})")
        return response.strip()
    except Exception as e:
        print(f"[model_runner] Inference error for {model_path}: {e}")
        return ""


def run_model_batch(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    prepend_math_instruction: bool = True,
    disable_thinking: bool = False,
    stop_after_json: bool = False,
) -> list[str]:
    """Run batch inference on multiple prompts efficiently.

    Tokenizes and generates in a single batch pass, utilising GPU
    parallelism instead of sequential single-prompt calls.

    Args:
        model_path: Path or ID of the model to use.
        prompts: List of prompt strings to generate answers for.
        max_new_tokens: Max tokens per generation (default from settings).

    Returns:
        List of generated strings, one per prompt, in order.
    """
    from config.settings import (
        INFERENCE_BATCH_SIZE,
        INFERENCE_MAX_NEW_TOKENS,
        USE_VLLM,
        USE_VLLM_FOR_LOCAL_CHECKPOINTS,
        USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS,
        VLLM_LOCAL_CHECKPOINT_SUBPROCESS_TIMEOUT_SECONDS,
    )

    if max_new_tokens is None:
        max_new_tokens = INFERENCE_MAX_NEW_TOKENS

    if not prompts:
        return []

    if prepend_math_instruction:
        prompts = [_get_instruction_prefix() + p for p in prompts]

    if not model_path:
        return [""] * len(prompts)

    t0 = time.time()

    is_local_checkpoint = Path(model_path).is_dir()
    is_lora_adapter = _is_lora_adapter_path(model_path)
    should_use_vllm = USE_VLLM and not (is_local_checkpoint and not USE_VLLM_FOR_LOCAL_CHECKPOINTS)
    should_hard_fail_local_vllm = should_use_vllm and is_local_checkpoint and not is_lora_adapter
    if should_use_vllm and is_lora_adapter:
        # vLLM cannot directly load an unmerged LoRA adapter directory. Keep
        # adapter inference on the HF path while full local checkpoints share
        # the in-process vLLM engine guarded by _vllm_generation_locks.
        should_use_vllm = False

    if (
        should_use_vllm
        and is_local_checkpoint
        and not is_lora_adapter
        and USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS
        and not _env_truthy("MODEL_RUNNER_ISOLATED_CHILD", default=False)
    ):
        _local_checkpoint_gpu_gate.acquire_exclusive()
        try:
            workers_to_shutdown: list[Any] = []
            with _cache_lock:
                if _model_cache:
                    print("[model_runner] Clearing HF model cache before isolated local checkpoint vLLM")
                    _clear_hf_cache_unlocked()
                if _vllm_cache:
                    print("[model_runner] Clearing in-process vLLM engine before isolated local checkpoint vLLM")
                    _clear_vllm_cache_unlocked()
                if _vllm_worker_cache:
                    print("[model_runner] Clearing Ray vLLM workers before isolated local checkpoint vLLM")
                    workers_to_shutdown.extend(_clear_vllm_workers_unlocked())
            _shutdown_popped_vllm_workers(workers_to_shutdown)
            _collect_cuda()
            try:
                results = _run_local_checkpoint_batch_subprocess(
                    model_path,
                    prompts,
                    max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    disable_thinking=disable_thinking,
                    stop_after_json=stop_after_json,
                    timeout_seconds=VLLM_LOCAL_CHECKPOINT_SUBPROCESS_TIMEOUT_SECONDS,
                )
                elapsed = time.time() - t0
                print(
                    f"[model_runner] run_local_checkpoint_batch_subprocess: {len(prompts)} prompts "
                    f"in {elapsed:.2f}s (max_new_tokens={max_new_tokens}, temperature={temperature})"
                )
                return results
            except Exception as exc:
                _mark_vllm_disabled(model_path, f"{type(exc).__name__}: {exc}")
                print(
                    f"[model_runner] isolated local checkpoint vLLM batch failed "
                    f"({type(exc).__name__}: {exc}); aborting without HF fallback"
                )
                raise
        finally:
            _local_checkpoint_gpu_gate.release_exclusive()

    if should_use_vllm:
        disabled_reason = _vllm_disabled_reason(model_path)
        if disabled_reason:
            if should_hard_fail_local_vllm:
                raise VllmEngineLoadError(
                    f"local checkpoint vLLM disabled after previous failure for {model_path}: "
                    f"{disabled_reason}"
                )
            if (
                _disable_hf_fallback_after_vllm_memory_error()
                and _is_vllm_gpu_unsafe_error(RuntimeError(disabled_reason))
            ):
                raise VllmEngineLoadError(
                    f"vLLM disabled after GPU-unsafe failure for {model_path}; "
                    "aborting without same-GPU HF fallback: "
                    f"{disabled_reason}"
                )
            print(
                f"[model_runner] Skipping vLLM for {model_path}; "
                f"disabled after previous failure: {disabled_reason}"
            )
            should_use_vllm = False

    if should_use_vllm:
        try:
            results = _run_model_batch_vllm_worker(
                model_path,
                prompts,
                max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                disable_thinking=disable_thinking,
                stop_after_json=stop_after_json,
            )
            elapsed = time.time() - t0
            print(f"[model_runner] run_model_batch_vllm_worker: {len(prompts)} prompts in {elapsed:.2f}s "
                  f"(max_new_tokens={max_new_tokens}, temperature={temperature})")
            return results
        except Exception as e:
            is_vllm_memory_failure = _is_vllm_memory_error(e)
            is_vllm_timeout_failure = _is_vllm_timeout_error(e)
            is_vllm_actor_failure = _is_ray_actor_died_error(e)
            if isinstance(e, VllmEngineLoadError) or is_vllm_memory_failure or is_vllm_timeout_failure:
                _mark_vllm_disabled(model_path, f"{type(e).__name__}: {e}")
            if should_hard_fail_local_vllm or _env_truthy("MODEL_RUNNER_ISOLATED_CHILD", default=False):
                print(
                    f"[model_runner] local checkpoint vLLM batch failed ({type(e).__name__}: {e}); "
                    "aborting without HF fallback"
                )
                raise
            if is_vllm_memory_failure and _disable_hf_fallback_after_vllm_memory_error():
                print(
                    f"[model_runner] vLLM batch failed with memory error ({type(e).__name__}: {e}); "
                    "aborting without same-GPU HF fallback"
                )
                raise
            if is_vllm_actor_failure and _disable_hf_fallback_after_vllm_memory_error():
                _mark_vllm_disabled(model_path, f"{type(e).__name__}: {e}")
                print(
                    f"[model_runner] vLLM batch failed with Ray actor failure ({type(e).__name__}: {e}); "
                    "aborting without same-GPU HF fallback"
                )
                raise
            if is_vllm_timeout_failure:
                print(
                    f"[model_runner] vLLM batch failed with Ray timeout ({type(e).__name__}: {e}); "
                    "aborting without same-GPU HF fallback"
                )
                raise
            print(f"[model_runner] vLLM batch failed ({type(e).__name__}: {e}), falling back to HF")

    _require_inference_deps()

    if USE_VLLM:
        workers_to_shutdown: list[Any] = []
        with _cache_lock:
            if _vllm_cache:
                print(f"[model_runner] Clearing vLLM engine before HF inference")
                _clear_vllm_cache_unlocked()
            if _vllm_worker_cache:
                print("[model_runner] Clearing local vLLM workers before HF inference")
                workers_to_shutdown.extend(_clear_vllm_workers_unlocked())
        _shutdown_popped_vllm_workers(workers_to_shutdown)
        _collect_cuda()

    try:
        model, tokenizer = _load_model(model_path)
    except Exception as e:
        print(f"[model_runner] Failed to load model for batch: {model_path}: {e}")
        return [""] * len(prompts)

    all_results: list[str] = []
    batch_size = INFERENCE_BATCH_SIZE

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        try:
            batch_results = _generate(
                model,
                tokenizer,
                batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                disable_thinking=disable_thinking,
                stop_after_json=stop_after_json,
            )
            all_results.extend(batch_results)
        except Exception as e:
            print(f"[model_runner] Batch generate failed ({type(e).__name__}: {e}), falling back to sequential")
            for p in batch:
                try:
                    result = run_model_once(
                        model_path,
                        p,
                        max_new_tokens=max_new_tokens,
                        prepend_math_instruction=False,
                        disable_thinking=disable_thinking,
                        stop_after_json=stop_after_json,
                    )
                    all_results.append(result)
                except Exception as e2:
                    print(f"[model_runner] Sequential fallback also failed: {e2}")
                    all_results.append("")

    elapsed = time.time() - t0
    print(f"[model_runner] run_model_batch: {len(prompts)} prompts in {elapsed:.2f}s "
          f"(batch_size={batch_size}, max_new_tokens={max_new_tokens})")
    return all_results


def judge_answer(prediction: str, gold_answer: str) -> bool:
    """Judge if a model prediction matches the gold answer for math questions.

    Uses math-verify on full text before extraction so LaTeX structures such as
    nested ``\\boxed{\\frac{...}{...}}`` are preserved. Fallbacks stay conservative:
    numeric last-number matching is allowed only for simple numeric gold answers
    to avoid false positives on intervals, multi-part answers, and equations.
    """
    if not prediction or not gold_answer:
        return False

    prediction_clean = prediction.strip()
    gold_clean = gold_answer.strip()

    if prediction_clean.lower() == gold_clean.lower():
        return True

    if _is_structured_math_answer(gold_clean):
        pred_structured = _extract_structured_answer(prediction_clean)
        gold_structured = _extract_structured_answer(gold_clean)
        if _math_text_equal(pred_structured, gold_structured):
            return True
        if _allows_structured_math_verify(gold_structured) and _math_verify_equiv(pred_structured, gold_structured):
            return True
        return False

    if _math_verify_equiv(prediction_clean, gold_clean):
        return True

    pred_final = _extract_final_answer(prediction_clean)
    gold_final = _extract_final_answer(gold_clean)

    if not pred_final:
        return False

    if not gold_final:
        gold_final = gold_clean

    if pred_final.lower() == gold_final.lower():
        return True

    if _math_equiv(pred_final, gold_final, allow_numeric_fallback=_is_simple_numeric_answer(gold_final)):
        return True

    return False


def judge_prediction_for_item(prediction: str, item, gold_answer: str = "") -> bool:
    """Domain-aware judging for one question item.

    Code domain (``USE_CODE_EXECUTION_VERIFIER``): execute the candidate code
    against the item's executable tests (``item['test']`` + ``item['entry_point']``)
    via :mod:`src.tools.code_execution`. No symbolic math, no LLM-as-judge -
    correctness is purely "did the candidate pass the tests".

    Math domain (default): symbolic math-verify / SymPy judge on
    ``(prediction, gold_answer)`` via :func:`judge_answer`.

    ``item`` may be a dict (question record) or a Pydantic question model.
    ``gold_answer`` is optional; when omitted it is read from the item.
    """
    try:
        from config import settings as _settings
        use_code = getattr(_settings, "USE_CODE_EXECUTION_VERIFIER", False)
    except Exception:
        use_code = False

    if use_code:
        from src.tools.code_execution import judge_code_prediction

        test_code = ""
        entry_point = ""
        if isinstance(item, dict):
            test_code = str(item.get("test", "") or "")
            entry_point = str(item.get("entry_point", "") or "")
        else:
            test_code = str(getattr(item, "test", "") or "")
            entry_point = str(getattr(item, "entry_point", "") or "")
        return judge_code_prediction(prediction, test_code, entry_point).passed

    if not gold_answer:
        if isinstance(item, dict):
            gold_answer = (
                item.get("rollout_gold_answer")
                or item.get("gold_answer")
                or ""
            )
        else:
            gold_answer = (
                getattr(item, "rollout_gold_answer", "")
                or getattr(item, "gold_answer", "")
            )
    return judge_answer(prediction, gold_answer)


def _extract_final_answer(text: str, marker: str | None = None) -> str:
    """Extract final answer with priority: boxed > sep-marker > regex patterns > last number.

    Returns the extracted answer string, or empty string if none found.
    """
    import re

    boxed = _extract_last_boxed(text)
    if boxed:
        return boxed

    sep = marker if (marker and marker in text) else ("####" if "####" in text else None)
    if sep:
        return text.split(sep)[-1].strip()

    patterns = [
        r"答案[：:]\s*([^\n]+)",
        r"the answer is\s*([^\n]+)",
        r"final answer[:\s]+([^\n]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    nums = _extract_numbers(text)
    if nums:
        return str(nums[-1])

    return ""


def _extract_structured_answer(text: str, marker: str | None = None) -> str:
    """Extract a candidate answer without falling back to the last number."""
    import re

    normalized = str(text or "").strip()
    boxed = _extract_last_boxed(normalized)
    if boxed:
        return boxed

    sep = marker if (marker and marker in normalized) else ("####" if "####" in normalized else None)
    if sep:
        return normalized.split(sep)[-1].strip()

    patterns = [
        r"答案[：:]\s*([^\n]+)",
        r"the answer is\s*([^\n]+)",
        r"final answer\s+is\s*([^\n]+)",
        r"final answer[:\s]+([^\n]+)",
    ]
    for pat in patterns:
        match = re.search(pat, normalized, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return normalized


def _extract_last_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...} in text, including nested braces."""
    starts: list[int] = []
    needle = r"\boxed{"
    search_pos = 0
    while True:
        idx = text.find(needle, search_pos)
        if idx < 0:
            break
        starts.append(idx + len(needle))
        search_pos = idx + len(needle)

    for start in reversed(starts):
        depth = 1
        pos = start
        while pos < len(text):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:pos].strip()
            pos += 1
    return ""


def _math_verify_equiv(pred: str, gold: str) -> bool:
    """Check mathematical equivalence with math-verify using gold-target order."""
    try:
        from math_verify import parse, verify

        pred_parsed = parse(pred)
        gold_parsed = parse(gold)
        if pred_parsed and gold_parsed:
            return bool(verify(gold_parsed, pred_parsed))
    except (ImportError, Exception):
        return False
    return False


def _math_equiv(pred: str, gold: str, *, allow_numeric_fallback: bool = True) -> bool:
    """Check mathematical equivalence using math-verify, sympy, then numeric fallback."""
    if _math_verify_equiv(pred, gold):
        return True

    try:
        import sympy
        pred_expr = sympy.sympify(pred)
        gold_expr = sympy.sympify(gold)
        diff = sympy.simplify(pred_expr - gold_expr)
        if diff == 0:
            return True
    except (ImportError, Exception):
        pass

    return bool(allow_numeric_fallback and _numeric_match(pred, gold))


def _math_text_equal(left: str, right: str) -> bool:
    """Compare math text while ignoring whitespace and harmless trailing punctuation."""
    def normalize(value: str) -> str:
        return "".join(str(value or "").strip().rstrip(".。").split()).lower()

    return bool(normalize(left) and normalize(left) == normalize(right))


def _allows_structured_math_verify(text: str) -> bool:
    """Use math-verify only for single structured expressions, not sets/equations."""
    normalized = str(text or "").strip()
    if not normalized:
        return False
    unsafe_markers = (
        r"\cup",
        r"\cap",
        r"\infty",
        r"\le",
        r"\ge",
        r"\neq",
        "∞",
        "∪",
        "∩",
        "≤",
        "≥",
        "!=",
    )
    if any(marker in normalized for marker in unsafe_markers):
        return False
    if "=" in normalized:
        return False
    if "," in normalized and len(_extract_numbers(normalized)) > 1:
        return False
    if any(ch in normalized for ch in "[](){}") and len(_extract_numbers(normalized)) > 1:
        return False
    return True


def _is_structured_math_answer(text: str) -> bool:
    """Return True for answers where last-number fallback is unsafe."""
    normalized = str(text or "").strip()
    if not normalized:
        return False
    if _is_simple_numeric_answer(normalized):
        return False
    structured_markers = (
        r"\frac",
        r"\sqrt",
        r"\cup",
        r"\cap",
        r"\infty",
        r"\text",
        r"\begin",
        r"\le",
        r"\ge",
        r"\neq",
        r"\pi",
        "∞",
        "∪",
        "∩",
        "≤",
        "≥",
        "!=",
    )
    if any(marker in normalized for marker in structured_markers):
        return True
    if "\\boxed{" in normalized and not _is_simple_numeric_answer(_extract_last_boxed(normalized)):
        return True
    if "=" in normalized:
        return True
    if "," in normalized and len(_extract_numbers(normalized)) > 1:
        return True
    if any(ch in normalized for ch in "[](){}") and len(_extract_numbers(normalized)) > 1:
        return True
    return False


def _is_simple_numeric_answer(text: str) -> bool:
    """Return True when gold is only one plain decimal/integer value."""
    import re

    normalized = str(text or "").strip()
    if not normalized:
        return False
    normalized = normalized.replace(",", "")
    return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", normalized))


def _numeric_match(prediction: str, gold: str) -> bool:
    """Check if numbers in prediction match gold answer."""
    pred_nums = _extract_numbers(prediction)
    gold_nums = _extract_numbers(gold)

    if not pred_nums or not gold_nums:
        return False

    if abs(pred_nums[-1] - gold_nums[-1]) < 1e-6:
        return True

    if len(pred_nums) == len(gold_nums) and all(abs(p - g) < 1e-6 for p, g in zip(pred_nums, gold_nums)):
        return True

    return False


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text."""
    import re
    numbers = re.findall(r"-?\d+\.?\d*", text)
    result = []
    for n in numbers:
        try:
            result.append(float(n))
        except ValueError:
            continue
    return result


def clear_model_cache():
    """Clear all cached models to free memory."""
    workers_to_shutdown: list[Any] = []
    with _cache_lock:
        _clear_hf_cache_unlocked()
        _clear_vllm_cache_unlocked()
        workers_to_shutdown.extend(_clear_vllm_workers_unlocked())
    _shutdown_popped_vllm_workers(workers_to_shutdown)
    _collect_cuda()


def get_cache_info() -> dict:
    """Return information about cached models."""
    with _cache_lock:
        return {
            "cached_models": list(_model_cache.keys()),
            "cached_vllm_models": list(_vllm_cache.keys()),
            "cached_vllm_workers": list(_vllm_worker_cache.keys()),
            "vllm_disabled_models": dict(_vllm_disabled_models),
            "cache_size": len(_model_cache),
            "vllm_cache_size": len(_vllm_cache),
            "vllm_worker_cache_size": len(_vllm_worker_cache),
        }


def _run_batch_json_cli() -> int:
    sys.stdout.flush()
    sys.stderr.flush()
    if _env_truthy("MODEL_RUNNER_ISOLATED_CHILD", default=False):
        os.environ["MODEL_RUNNER_DISABLE_RAY_VLLM"] = "1"
    original_stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    # vLLM/transformers can write directly to fd 1, bypassing Python's
    # sys.stdout. Send all runtime stdout noise to stderr and reserve the saved
    # original stdout fd for the sentinel-delimited protocol response only.
    os.dup2(stderr_fd, 1)
    os.close(stderr_fd)
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        prompts = payload.get("prompts")
        if not isinstance(prompts, list):
            raise ValueError("payload.prompts must be a list")
        model_path = str(payload.get("model_path") or "")
        max_new_tokens = int(payload.get("max_new_tokens") or 96)
        temperature = float(payload.get("temperature") or 0.0)
        top_p = float(payload.get("top_p") or 1.0)
        disable_thinking = bool(payload.get("disable_thinking") or False)
        stop_after_json = bool(payload.get("stop_after_json") or False)
        results = run_model_batch(
            model_path=model_path,
            prompts=[str(item) for item in prompts],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prepend_math_instruction=False,
            disable_thinking=disable_thinking,
            stop_after_json=stop_after_json,
        )
        clear_model_cache()
        _write_batch_json_response(original_stdout_fd, {"ok": True, "results": results})
        return 0
    except Exception as exc:
        _write_batch_json_response(original_stdout_fd, {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        return 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(original_stdout_fd, 1)
        except OSError:
            pass
        try:
            os.close(original_stdout_fd)
        except OSError:
            pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch-json":
        raise SystemExit(_run_batch_json_cli())
    raise SystemExit("Usage: python -m src.tools.model_runner --batch-json")
