from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class RayVllmModelSwitchGuard:
    """Coordinate Ray vLLM model switches on a single GPU.

    The guard is intentionally small: it only owns the active-model switch and
    stale actor cleanup boundary. Actor lookup/start remains in model_runner.
    """

    def __init__(
        self,
        *,
        clear_stale: Callable[[str, str], list[Any]],
        shutdown_actors: Callable[[Any], None],
    ) -> None:
        self._condition = threading.Condition()
        self._exclusive_active = False
        self._exclusive_waiting_by_target: dict[str, int] = {}
        self._shared_active = 0
        self._active_model_key = ""
        self._clear_stale = clear_stale
        self._shutdown_actors = shutdown_actors

    @property
    def active_model_key(self) -> str:
        with self._condition:
            return self._active_model_key

    def acquire_model(self, target_model_key: str) -> None:
        target_model_key = str(target_model_key or "")
        previous_model_key = self._begin_switch_or_acquire_shared(target_model_key)
        if previous_model_key is None:
            return

        error: BaseException | None = None
        try:
            actors_to_shutdown = self._clear_stale(target_model_key, previous_model_key)
            for actor in actors_to_shutdown:
                self._shutdown_actors(actor)
        except BaseException as exc:
            error = exc

        with self._condition:
            if error is None:
                self._active_model_key = target_model_key
                self._shared_active += 1
            self._exclusive_active = False
            self._condition.notify_all()

        if error is not None:
            raise error

    def release_model(self) -> None:
        with self._condition:
            self._shared_active -= 1
            if self._shared_active == 0:
                self._condition.notify_all()

    def _begin_switch_or_acquire_shared(self, target_model_key: str) -> str | None:
        with self._condition:
            while True:
                if self._active_model_key == target_model_key:
                    while self._exclusive_active or self._has_waiting_switch_for_other_model(target_model_key):
                        self._condition.wait()
                    if self._active_model_key == target_model_key:
                        self._shared_active += 1
                        return None
                    continue

                self._exclusive_waiting_by_target[target_model_key] = (
                    self._exclusive_waiting_by_target.get(target_model_key, 0) + 1
                )
                try:
                    while self._exclusive_active or self._shared_active:
                        self._condition.wait()
                        if self._active_model_key == target_model_key:
                            break
                    if self._active_model_key == target_model_key:
                        continue
                    self._exclusive_active = True
                    return self._active_model_key
                finally:
                    waiting = self._exclusive_waiting_by_target.get(target_model_key, 0) - 1
                    if waiting > 0:
                        self._exclusive_waiting_by_target[target_model_key] = waiting
                    else:
                        self._exclusive_waiting_by_target.pop(target_model_key, None)

    def _has_waiting_switch_for_other_model(self, target_model_key: str) -> bool:
        return any(
            waiting_target != target_model_key and count > 0
            for waiting_target, count in self._exclusive_waiting_by_target.items()
        )
