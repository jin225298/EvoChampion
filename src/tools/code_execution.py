"""Code-execution judging for the code domain.

This module implements the "执行测试" (execution-based) judging required by the
code-domain self-evolution loop. Untrusted model-generated code is run inside a
sandboxed subprocess with resource limits and a timeout; correctness is decided
purely by whether the supplied ``check(candidate)`` test passes — never by LLM
subjective equivalence.

Design properties (see task spec "必须实现：代码执行判题逻辑"):

* 沙箱执行 — untrusted code runs in an isolated subprocess (``python -I -B``)
  with ``resource.setrlimit`` (address space / CPU / nproc) and a hard timeout.
* 测试构造 — a HumanEval-style harness is built from ``candidate_code`` +
  ``test`` + ``entry_point`` only. The gold/reference solution is NEVER placed in
  the harness (防污染); the test references the candidate via ``check(entry_point)``.
* 多候选聚合 — several rollout candidates for the same question are judged
  independently and aggregated; a question counts as correct only when ALL
  candidates pass (全部通过才算对).
* 失败诊断 — failures are classified into compile / assert / timeout / runtime /
  crash with a structured diagnostic.
* 难度估计 — difficulty is derived solely from execution results:
  all-pass=easy, partial=medium, all-fail=hard, no-test=unknown.
* 缓存并行 — judging results are cached (keyed by a hash of the inputs) and a
  thread pool judges candidates concurrently.

All public functions return structured dicts so the rest of the pipeline
(``difficulty_tagger``, ``evaluator``) can consume them uniformly.
"""

from __future__ import annotations

import hashlib
import os
import resource
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional

__all__ = [
    "EVALUATION_METHOD_CODE",
    "CodeJudgeResult",
    "build_harness",
    "run_sandboxed",
    "judge_code_candidate",
    "aggregate_candidates",
    "difficulty_from_results",
    "judge_predictions_code_batch",
    "extract_code_test_fields",
    "clear_code_judge_cache",
]


# Evaluation-method tag used by the dataset standardizer / difficulty tagger to
# route code items through execution judging instead of symbolic ``judge_answer``.
EVALUATION_METHOD_CODE = "code_execution"

# Failure taxonomy. ``none`` means the test passed (exit 0, no assertion error).
ERROR_TYPES = ("none", "compile", "assert", "timeout", "runtime", "crash")

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MEMORY_MB = 512
_DEFAULT_CPU_SECONDS = 15
_MAX_WORKERS = 8

# A small in-memory cache of judge results. Keyed by a sha256 of the candidate
# code + test + entry_point + timeout/memory so identical rollouts are not
# re-executed. Bounded to avoid unbounded growth.
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, "CodeJudgeResult"] = {}
_CACHE_MAX = 4096


@dataclass
class CodeJudgeResult:
    """Structured result of judging one code candidate against one test.

    ``passed`` is True only when the harness exited 0 with no assertion error.
    ``error_type`` is one of :data:`ERROR_TYPES`. ``error_message`` carries a
    short, human-readable diagnostic (last lines of stderr / signal name).
    """

    passed: bool = False
    error_type: str = "runtime"
    error_message: str = ""
    duration_ms: int = 0
    entry_point: str = ""
    test_code: str = ""
    candidate_code: str = ""
    returncode: Optional[int] = None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------
def extract_code_test_fields(item: Any) -> tuple[str, str, str]:
    """Return ``(test_code, entry_point, candidate_gold)`` from a dataset item.

    Tolerant of dict / dataclass / object items and several common field names.
    ``candidate_gold`` (the reference solution) is extracted ONLY so the caller
    can log/debug — it is never placed in the execution harness (防污染).
    """
    test_code = ""
    for name in ("test", "test_code", "check", "harness", "tests"):
        value = _get_field(item, name)
        if value:
            test_code = str(value)
            break
    entry_point = ""
    for name in ("entry_point", "entrypoint", "function", "func_name", "target_func"):
        value = _get_field(item, name)
        if value:
            entry_point = str(value)
            break
    gold = ""
    for name in ("answer", "reference_solution", "canonical_solution", "solution", "gold"):
        value = _get_field(item, name)
        if value:
            gold = str(value)
            break
    return test_code, entry_point, gold


def _get_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


# ---------------------------------------------------------------------------
# Harness construction (防污染: no gold/reference in the harness)
# ---------------------------------------------------------------------------
def build_harness(candidate_code: str, test_code: str, entry_point: str) -> str:
    """Build an executable HumanEval-style harness.

    The harness contains ONLY the candidate's code and the test's ``check``
    function, then calls ``check(entry_point)``. The reference/gold solution is
    intentionally absent so the test cannot be satisfied by reading the answer.
    """
    candidate_code = (candidate_code or "").rstrip()
    test_code = (test_code or "").rstrip()
    entry_point = (entry_point or "").strip()

    # If the test already defines check(...) and calls it, we still append our
    # own guarded call so a missing/renamed entry point surfaces as a runtime
    # NameError rather than a silent pass.
    call_line = f"check({entry_point})" if entry_point else "check()"
    harness = (
        "# === candidate code (untrusted, sandboxed) ===\n"
        f"{candidate_code}\n\n"
        "# === test (check function) ===\n"
        f"{test_code}\n\n"
        "# === entry point invocation ===\n"
        "if __name__ == '__main__':\n"
        f"    {call_line}\n"
    )
    return harness


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------
def _preexec_limits(memory_mb: int, cpu_seconds: int):
    """Build a ``preexec_fn`` that applies resource limits in the child.

    Returns None on platforms where a limit cannot be set (e.g. RLIMIT_AS on
    some macOS configs) so we degrade gracefully rather than crash the judge.
    """
    def _apply() -> None:  # pragma: no cover - runs in subprocess
        try:
            # New process group so we can kill the whole tree on timeout.
            os.setpgrp()
        except Exception:
            pass
        try:
            mem_bytes = int(memory_mb) * 1024 * 1024
            if mem_bytes > 0:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass
        try:
            cpu = int(cpu_seconds)
            if cpu > 0:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        except Exception:
            pass
        try:
            # Cap child processes to a small number to contain fork bombs.
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except Exception:
            pass

    return _apply


def _classify_error(
    returncode: Optional[int],
    stderr: str,
    timed_out: bool,
) -> tuple[str, str]:
    """Classify a failure into (error_type, error_message).

    Classification is driven by the exception type on the *last* line of the
    traceback (e.g. ``AssertionError`` vs ``ValueError``), NOT by substring
    matching anywhere in stderr — because the traceback echoes the ``assert``
    source line of the test, which would otherwise collide with runtime errors.
    """
    if timed_out:
        return "timeout", "execution exceeded time limit"
    if returncode is None:
        return "crash", "process did not return (killed)"
    if returncode == 0:
        return "none", ""
    # Killed by a signal (negative returncode).
    if returncode < 0:
        signum = -returncode
        try:
            import signal
            name = signal.Signals(signum).name
        except Exception:
            name = f"signal {signum}"
        if signum == 6:  # SIGABRT — often an assertion/abort in C code
            return "crash", f"aborted ({name})"
        return "crash", f"killed by {name}"

    stderr_l = stderr or ""
    lines = [ln for ln in stderr_l.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    # The exception type is the token before the first ':' on the last line,
    # e.g. "ValueError: boom" -> "valueerror", "AssertionError" -> "assertionerror".
    low_last = last.lower()
    exc_type = low_last.split(":", 1)[0].strip() if ":" in low_last else low_last.strip()

    if "assertionerror" in exc_type:
        return "assert", _tail(stderr_l, "assertion failed")
    if "syntaxerror" in exc_type or "indentationerror" in exc_type or "taberror" in exc_type:
        return "compile", _tail(stderr_l, "compile error")
    if "nameerror" in exc_type:
        return "runtime", _tail(stderr_l, "name error (entry point not defined?)")
    if exc_type:
        return "runtime", _tail(stderr_l, f"{exc_type} error")
    # Non-zero exit with no traceback (e.g. os._exit, sys.exit with error).
    return "runtime", _tail(stderr_l, f"exit {returncode}")


def _tail(text: str, default: str, max_lines: int = 6) -> str:
    text = (text or "").strip()
    if not text:
        return default
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return " | ".join(lines[-max_lines:])[:600]


def run_sandboxed(
    harness_code: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = _DEFAULT_MEMORY_MB,
    cpu_seconds: int = _DEFAULT_CPU_SECONDS,
) -> dict[str, Any]:
    """Execute ``harness_code`` in a sandboxed subprocess.

    Returns a dict with keys: ``passed``, ``error_type``, ``error_message``,
    ``duration_ms``, ``returncode``, ``timed_out``, ``stdout``, ``stderr``.
    """
    import time

    t0 = time.monotonic()
    tmpdir = tempfile.mkdtemp(prefix="codejudge_")
    harness_path = os.path.join(tmpdir, "harness.py")
    try:
        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(harness_code)
    except Exception as exc:  # pragma: no cover - disk failure
        return {
            "passed": False,
            "error_type": "crash",
            "error_message": f"failed to write harness: {exc}",
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "returncode": None,
            "timed_out": False,
            "stdout": "",
            "stderr": str(exc),
        }

    env = dict(os.environ)
    # Isolate: no user site-packages, no bytecode, quiet, isolated mode.
    cmd = [sys.executable, "-I", "-B", "-X", "faulthandler", harness_path]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_preexec_limits(memory_mb, cpu_seconds),
            close_fds=True,
        )
    except Exception as exc:  # pragma: no cover - spawn failure
        return {
            "passed": False,
            "error_type": "crash",
            "error_message": f"failed to spawn judge process: {exc}",
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "returncode": None,
            "timed_out": False,
            "stdout": "",
            "stderr": str(exc),
        }

    timed_out = False
    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        try:
            stdout_b, stderr_b = proc.communicate(timeout=5)
        except Exception:
            stdout_b, stderr_b = b"", b""
        returncode = proc.returncode
    except Exception as exc:  # pragma: no cover
        return {
            "passed": False,
            "error_type": "crash",
            "error_message": f"judge process error: {exc}",
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "returncode": None,
            "timed_out": False,
            "stdout": "",
            "stderr": str(exc),
        }

    stdout = _decode(stdout_b)
    stderr = _decode(stderr_b)
    error_type, error_message = _classify_error(returncode, stderr, timed_out)
    passed = (not timed_out) and returncode == 0 and error_type == "none"
    return {
        "passed": passed,
        "error_type": error_type,
        "error_message": error_message,
        "duration_ms": int((time.monotonic() - t0) * 1000),
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": stdout[-2000:],
        "stderr": stderr[-4000:],
    }


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _decode(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return repr(data)


# ---------------------------------------------------------------------------
# Single-candidate judging (with cache)
# ---------------------------------------------------------------------------
def _cache_key(candidate_code: str, test_code: str, entry_point: str,
               timeout: float, memory_mb: int) -> str:
    raw = f"{candidate_code}\x00{test_code}\x00{entry_point}\x00{timeout}\x00{memory_mb}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def judge_code_candidate(
    candidate_code: str,
    test_code: str,
    entry_point: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = _DEFAULT_MEMORY_MB,
    cpu_seconds: int = _DEFAULT_CPU_SECONDS,
    use_cache: bool = True,
) -> CodeJudgeResult:
    """Judge one code candidate against one test. Cached by input hash."""
    if not (candidate_code or "").strip():
        return CodeJudgeResult(
            passed=False, error_type="runtime",
            error_message="empty candidate code", entry_point=entry_point,
            test_code=test_code, candidate_code=candidate_code,
        )
    if not (test_code or "").strip() or not (entry_point or "").strip():
        # No executable test → cannot judge by execution (unknown / unavailable).
        return CodeJudgeResult(
            passed=False, error_type="runtime",
            error_message="missing test or entry_point", entry_point=entry_point,
            test_code=test_code, candidate_code=candidate_code,
        )

    key = _cache_key(candidate_code, test_code, entry_point, timeout, memory_mb)
    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            return cached

    harness = build_harness(candidate_code, test_code, entry_point)
    out = run_sandboxed(harness, timeout=timeout, memory_mb=memory_mb, cpu_seconds=cpu_seconds)
    result = CodeJudgeResult(
        passed=bool(out["passed"]),
        error_type=out["error_type"],
        error_message=out["error_message"],
        duration_ms=out["duration_ms"],
        entry_point=entry_point,
        test_code=test_code,
        candidate_code=candidate_code,
        returncode=out["returncode"],
        timed_out=out["timed_out"],
        stdout=out["stdout"],
        stderr=out["stderr"],
    )
    if use_cache:
        with _CACHE_LOCK:
            if len(_CACHE) >= _CACHE_MAX:
                # Drop a quarter of the oldest-ish entries to bound memory.
                for k in list(_CACHE.keys())[: _CACHE_MAX // 4]:
                    _CACHE.pop(k, None)
            _CACHE[key] = result
    return result


def clear_code_judge_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# Multi-candidate aggregation + difficulty
# ---------------------------------------------------------------------------
def aggregate_candidates(results: Iterable[CodeJudgeResult]) -> dict[str, Any]:
    """Aggregate several rollout candidates for the same question.

    A question is correct only when ALL candidates pass (全部通过才算对).
    Returns structured summary with pass counts / rates / per-candidate errors.
    """
    results = [r for r in results if r is not None]
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "all_pass": False,
            "any_pass": False,
            "error_types": {},
            "results": [],
        }
    pass_count = sum(1 for r in results if r.passed)
    error_types: dict[str, int] = {}
    for r in results:
        if not r.passed:
            error_types[r.error_type] = error_types.get(r.error_type, 0) + 1
    return {
        "total": total,
        "pass_count": pass_count,
        "pass_rate": pass_count / total,
        "all_pass": pass_count == total,
        "any_pass": pass_count > 0,
        "error_types": error_types,
        "results": [r.to_dict() for r in results],
    }


def difficulty_from_results(results: Iterable[CodeJudgeResult]) -> str:
    """Estimate difficulty from execution results only.

    all-pass=easy, partial=medium, all-fail=hard, no-test/unknown=unknown.
    """
    results = [r for r in results if r is not None]
    total = len(results)
    if total == 0:
        return "unknown"
    # If every candidate was "missing test", treat as unknown (no executable test).
    if all(r.error_type == "runtime" and "missing test" in (r.error_message or "")
           for r in results):
        return "unknown"
    pass_count = sum(1 for r in results if r.passed)
    if pass_count == total:
        return "easy"
    if pass_count == 0:
        return "hard"
    return "medium"


# ---------------------------------------------------------------------------
# Batch judging — drop-in for difficulty_tagger._judge_predictions_batch
# ---------------------------------------------------------------------------
def judge_predictions_code_batch(
    predictions: list[str],
    items: list[Any],
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = _DEFAULT_MEMORY_MB,
    max_workers: int = _MAX_WORKERS,
) -> list[dict[str, Any]]:
    """Judge a batch of (prediction, item) pairs by execution.

    ``predictions[i]`` is the model-generated code for ``items[i]``. Returns a
    list of judgement dicts shaped like ``_judge_predictions_batch`` output so
    the rollout pipeline can consume them uniformly::

        {"correct": bool, "score": float, "reason": str, "source": "code_execution",
         "judge_raw_text": str, "fallback_used": False, "schema_errors": []}

    ``correct`` follows the "all pass" rule; with a single candidate per
    question (the common rollout case) it is simply that candidate's pass state.
    """
    n = len(predictions)
    judgements: list[dict[str, Any]] = [
        {
            "correct": False,
            "score": 0.0,
            "reason": "not judged",
            "source": "code_execution",
            "judge_raw_text": "",
            "fallback_used": False,
            "schema_errors": [],
        }
        for _ in range(n)
    ]
    if n == 0:
        return judgements

    workers = max(1, min(int(max_workers), n, _MAX_WORKERS))

    def _judge_one(idx: int) -> tuple[int, CodeJudgeResult]:
        pred = predictions[idx] if idx < len(predictions) else ""
        item = items[idx] if idx < len(items) else {}
        test_code, entry_point, _gold = extract_code_test_fields(item)
        # HumanEval-style completion tasks: the model generates the function
        # body given the signature (question). The executable candidate is
        # completion_prompt + prediction; without this, the entry_point would
        # be undefined and the test would always fail.
        completion_prompt = ""
        if isinstance(item, dict):
            completion_prompt = str(item.get("completion_prompt") or "")
        candidate_code = f"{completion_prompt}{pred}"
        result = judge_code_candidate(
            candidate_code, test_code, entry_point,
            timeout=timeout, memory_mb=memory_mb,
        )
        import os as _os
        if _os.getenv("CODE_JUDGE_DEBUG", "").strip():
            print(f"[code_judge_debug] idx={idx} ep={entry_point!r} "
                  f"cp_len={len(completion_prompt)} pred_len={len(pred)} "
                  f"test_len={len(test_code or '')} passed={result.passed} "
                  f"err={result.error_type} pred_head={pred[:60]!r}", flush=True)
        return idx, result

    if workers == 1:
        for idx in range(n):
            i, result = _judge_one(idx)
            judgements[i] = _judgement_from_result(result)
        return judgements

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_judge_one, idx) for idx in range(n)]
        for fut in as_completed(futures):
            idx, result = fut.result()
            judgements[idx] = _judgement_from_result(result)
    return judgements


def _judgement_from_result(result: CodeJudgeResult) -> dict[str, Any]:
    correct = bool(result.passed)
    if correct:
        reason = "code execution: test passed"
    else:
        reason = f"code execution: {result.error_type} — {result.error_message}"
    return {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "reason": reason,
        "source": "code_execution",
        "judge_raw_text": result.error_message,
        "fallback_used": False,
        "schema_errors": [],
        # Extra structured fields for downstream difficulty / diagnostics.
        "error_type": result.error_type,
        "duration_ms": result.duration_ms,
        "passed": correct,
    }
