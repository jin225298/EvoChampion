"""Ray Actor wrapper for vLLM inference.

Ray owns the worker process, routing, shared-memory object transport, and
restart policy.  The actor owns one vLLM AsyncLLMEngine so concurrent callers
share a single KV cache allocation and APC prefix cache per model actor.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

try:
    import ray
except ImportError:  # pragma: no cover - exercised via model_runner fallback
    ray = None

from src.tools.vllm_worker import (
    _build_engine,
    _get_engine_tokenizer,
    _handle_generate,
    _shutdown_engine,
)


class VllmRayActor:
    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        enforce_eager: bool,
        enable_prefix_caching: bool = True,
    ) -> None:
        self.model_path = model_path
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self.engine = None
        self.tokenizer = None
        init_future = asyncio.run_coroutine_threadsafe(
            self._initialize(
                model_path=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                enable_prefix_caching=enable_prefix_caching,
            ),
            self._loop,
        )
        init_future.result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _initialize(
        self,
        model_path: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        enforce_eager: bool,
        enable_prefix_caching: bool,
    ) -> None:
        self.engine = _build_engine(
            model_path=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.tokenizer = await _get_engine_tokenizer(self.engine)

    def ready(self) -> dict:
        return {"ok": True, "model_path": self.model_path}

    def generate(self, request: dict) -> dict:
        try:
            future = asyncio.run_coroutine_threadsafe(
                _handle_generate(self.engine, self.tokenizer, request),
                self._loop,
            )
            return future.result()
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def shutdown(self) -> dict:
        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown_engine(self.engine), self._loop)
            future.result()
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            return {"ok": True, "results": []}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def __ray_shutdown__(self) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown_engine(self.engine), self._loop)
            future.result(timeout=10)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass


def get_ray_actor_class(
    num_gpus: float,
    max_restarts: int,
    max_task_retries: int,
    max_concurrency: int,
) -> Any:
    global ray
    if ray is None:
        import importlib
        ray = importlib.import_module("ray")
    return ray.remote(
        num_gpus=num_gpus,
        max_restarts=max_restarts,
        max_task_retries=max_task_retries,
        max_concurrency=max(1, int(max_concurrency)),
    )(VllmRayActor)
