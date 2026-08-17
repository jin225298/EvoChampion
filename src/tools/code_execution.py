"""Code-execution based answer verification for the code domain.

The math domain judges rollouts with ``math-verify`` + SymPy symbolic checks.
The code domain cannot do that: a candidate is "correct" only when its
definition actually passes the executable tests that ship with the problem.

This module implements that contract with strict execution-only semantics:

* 沙箱执行：untrusted candidate code runs in an isolated subprocess with
  resource limits (CPU + wall-clock timeout) so a runaway or malicious
  candidate cannot affect the judging process.
* 测试构造：the problem's ``test`` field (a ``def check(candidate):`` block)
  plus ``entry_point`` are assembled into a single harness that defines
  ``check``, defines the candidate function, then calls ``check(fn)``.
* 多候选聚合：for one question with N rollout candidates, every candidate is
  judged independently; the question is only "correct" when ALL candidates
  pass ("全部通过才算对").
* 失败诊断：compile errors / assertion failures / timeouts / runtime
  exceptions / process crashes are distinguished and reported structurally.
* 难度估计：difficulty is derived purely from execution results — all pass =
  easy, partial = medium, all fail = hard, no test = unknown. No LLM reading.
* 防污染：the harness never imports the gold answer; only the candidate code
  and the test block are executed.
* 缓存并行：results are cached by content hash; judging uses a thread pool.

No model weights, no network, no GPU — pure stdlib subprocess.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import textwrap
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

# Status constants — the single source of truth for failure modes.
STATUS_PASSED = "passed"
STATUS_COMPILE_ERROR = "compile_error"
STATUS_ASSERTION_ERROR = "assertion_error"
STATUS_TIMEOUT = "timeout"
STATUS_RUNTIME_ERROR = "runtime_error"
STATUS_CRASH = "crash"
STATUS_MISSING_ENTRY_POINT = "missing_entry_point"
STATUS_NO_TEST = "no_test"

# Difficulty labels (must match the math domain's easy/medium/hard/unknown).
DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"
DIFFICULTY_UNKNOWN = "unknown"

# Exit codes used by the harness to communicate status back to the parent.
_EXIT_PASSED = 0
_EXIT_ASSERTION = 1
_EXIT_COMPILE = 2
_EXIT_MISSING_ENTRY = 3
_EXIT_RUNTIME = 4
# 124 is conventionally used by `timeout`; we map it to timeout ourselves.

# Defaults. Overridable per-call and via env.
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("CODE_JUDGE_TIMEOUT_SECONDS", "10"))
DEFAULT_MAX_WORKERS = int(os.getenv("CODE_JUDGE_MAX_WORKERS", "8"))
# Hard CPU-second cap so a candidate cannot spin forever even if wall-clock
# timeout fails to fire (e.g. stuck in a C extension).
DEFAULT_CPU_SECONDS = int(os.getenv("CODE_JUDGE_CPU_SECONDS", "30"))

_MARKDOWN_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n?(.*?)```", re.DOTALL)
# Qwen3 thinking-mode output: <think>...</think> then the answer. We want the
# answer (after the closing tag), not the reasoning trace.
_THINK_BLOCK_RE = re.compile(r"<think[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


# =============================================================================
# Result structure
# =============================================================================
@dataclass
class CodeJudgeResult:
    """Structured result of judging one candidate against one problem's tests.

    ``passed`` is the only boolean callers need for a 0/1 verdict; ``status``
    gives the diagnostic category; ``error_type`` / ``error_message`` carry the
    exception detail when available.
    """

    passed: bool = False
    status: str = STATUS_CRASH
    error_type: str = ""
    error_message: str = ""
    exit_code: int = -1
    duration_ms: float = 0.0
    candidate_code: str = ""
    test_code: str = ""
    entry_point: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# Code extraction — turn a raw model rollout into runnable Python source
# =============================================================================
def extract_code(text: str) -> str:
    """Best-effort extraction of runnable Python code from a model rollout.

    The model may wrap code in markdown fences or prepend prose. We prefer the
    longest fenced block; if there is none, we fall back to the whole text
    (the harness will report a compile error if it is not valid Python).
    """
    if not text:
        return ""
    stripped = text.strip()
    # Drop thinking-mode traces; keep only the answer that follows them.
    no_think = _THINK_BLOCK_RE.sub("", stripped).strip()
    if no_think:
        stripped = no_think
    matches = _MARKDOWN_FENCE_RE.findall(stripped)
    if matches:
        # Prefer the longest fenced block — usually the real solution.
        return max(matches, key=len).strip()
    return stripped


# =============================================================================
# Harness assembly
# =============================================================================
_HARNESS_TEMPLATE = textwrap.dedent(
    """\
    # Auto-generated code-judging harness. Do not edit.
    # Runs in an isolated subprocess with resource limits.
    import sys as _sys

    _ENTRY_POINT = {entry_point!r}
    _status = None
    _error_type = ""
    _error_msg = ""

    # --- Define the test (check) block -----------------------------------
    try:
        _test_ns = {{}}
        exec(_TEST_CODE, _test_ns)
    except Exception as _e:
        # A broken test block is an infra failure, not a candidate failure.
        _sys.stderr.write("TEST_DEF_ERROR:" + repr(_e))
        _sys.exit(5)

    _check = _test_ns.get("check")
    if not callable(_check):
        _sys.stderr.write("TEST_NO_CHECK")
        _sys.exit(6)

    # --- Define the candidate --------------------------------------------
    try:
        _cand_ns = {{}}
        exec(_CANDIDATE_CODE, _cand_ns)
    except Exception as _e:
        _status = "compile_error"
        _error_type = type(_e).__name__
        _error_msg = _truncate(str(_e))
        _emit(_status, _error_type, _error_msg, _EXIT_COMPILE)

    _fn = _cand_ns.get(_ENTRY_POINT)
    if not callable(_fn):
        _status = "missing_entry_point"
        _error_msg = "entry_point {entry_point!r} not defined or not callable"
        _emit(_status, _error_type, _error_msg, _EXIT_MISSING_ENTRY)

    # --- Run the tests ---------------------------------------------------
    try:
        _check(_fn)
    except AssertionError as _e:
        _status = "assertion_error"
        _error_type = "AssertionError"
        _error_msg = _truncate(str(_e))
        _emit(_status, _error_type, _error_msg, _EXIT_ASSERTION)
    except Exception as _e:
        _status = "runtime_error"
        _error_type = type(_e).__name__
        _error_msg = _truncate(str(_e))
        _emit(_status, _error_type, _error_msg, _EXIT_RUNTIME)

    _status = "passed"
    _emit(_status, _error_type, _error_msg, _EXIT_PASSED)
    """
)


_EXIT_CODE_MAP = {
    _EXIT_PASSED: STATUS_PASSED,
    _EXIT_ASSERTION: STATUS_ASSERTION_ERROR,
    _EXIT_COMPILE: STATUS_COMPILE_ERROR,
    _EXIT_MISSING_ENTRY: STATUS_MISSING_ENTRY_POINT,
    _EXIT_RUNTIME: STATUS_RUNTIME_ERROR,
    5: "test_def_error",
    6: "test_no_check",
}


def _truncate(s: str, limit: int = 2000) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= limit else s[:limit] + "...[truncated]"


_EXIT_CONSTANTS_SRC = (
    f"_EXIT_PASSED = {_EXIT_PASSED}\n"
    f"_EXIT_ASSERTION = {_EXIT_ASSERTION}\n"
    f"_EXIT_COMPILE = {_EXIT_COMPILE}\n"
    f"_EXIT_MISSING_ENTRY = {_EXIT_MISSING_ENTRY}\n"
    f"_EXIT_RUNTIME = {_EXIT_RUNTIME}\n"
)


def build_harness(candidate_code: str, test_code: str, entry_point: str) -> str:
    """Assemble the full harness source: test + candidate + check call."""
    # Inline the two code blobs as repr-quoted strings inside the harness.
    cand_blob = _safe_blob(candidate_code)
    test_blob = _safe_blob(test_code)
    harness = _HARNESS_TEMPLATE.format(entry_point=entry_point)
    # Inject constants + code blobs right after the imports.
    inject = (
        _EXIT_CONSTANTS_SRC
        + f"_CANDIDATE_CODE = {cand_blob}\n"
        + f"_TEST_CODE = {test_blob}\n\n"
    )
    return inject + harness


def _safe_blob(code: str) -> str:
    """Render a code string as a Python string literal that survives exec."""
    # Use repr to get a properly quoted single-line string; newlines become \n.
    return repr(code)


def _emit_helper_src() -> str:
    return textwrap.dedent(
        """\
        def _emit(status, error_type, error_msg, exit_code):
            # Write a single line the parent parses: STATUS:TYPE:MSG
            line = "%s:%s:%s" % (status, error_type, _truncate(error_msg))
            _sys.stdout.write(line + "\\n")
            _sys.stdout.flush()
            _sys.exit(exit_code)
        def _truncate(s, limit=2000):
            s = "" if s is None else str(s)
            return s if len(s) <= limit else s[:limit] + "...[truncated]"
        """
    )


# =============================================================================
# Resource limits
# =============================================================================
def _preexec_limits(cpu_seconds: int):
    """Return a preexec_fn that installs CPU/memory limits (Linux only)."""
    def _limits():  # pragma: no cover - runs in subprocess
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            # Address space cap (bytes). 1 GiB is generous for code judging.
            try:
                mem = 1024 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except (ValueError, OSError):
                pass
            # Prevent fork bombs.
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            except (ValueError, OSError):
                pass
        except Exception:
            pass

    return _limits


# =============================================================================
# Core judging
# =============================================================================
def _run_harness(
    candidate_code: str,
    test_code: str,
    entry_point: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    python_exe: str | None = None,
) -> CodeJudgeResult:
    """Execute the harness in an isolated subprocess and parse the result."""
    import time

    harness_src = build_harness(candidate_code, test_code, entry_point)
    # Prepend the _emit helper (used by the template).
    harness_src = _emit_helper_src() + "\n" + harness_src

    py = python_exe or sys.executable
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [py, "-I", "-c", harness_src],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_preexec_limits(cpu_seconds) if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired:
        duration_ms = (time.monotonic() - start) * 1000
        return CodeJudgeResult(
            passed=False,
            status=STATUS_TIMEOUT,
            error_type="TimeoutExpired",
            error_message=f"exceeded {timeout}s",
            exit_code=124,
            duration_ms=round(duration_ms, 2),
            candidate_code=candidate_code,
            test_code=test_code,
            entry_point=entry_point,
        )
    except Exception as e:  # pragma: no cover - subprocess launch failure
        duration_ms = (time.monotonic() - start) * 1000
        return CodeJudgeResult(
            passed=False,
            status=STATUS_CRASH,
            error_type=type(e).__name__,
            error_message=_truncate(str(e)),
            exit_code=-1,
            duration_ms=round(duration_ms, 2),
            candidate_code=candidate_code,
            test_code=test_code,
            entry_point=entry_point,
        )

    duration_ms = (time.monotonic() - start) * 1000
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    exit_code = proc.returncode

    status = _EXIT_CODE_MAP.get(exit_code, STATUS_CRASH)
    error_type = ""
    error_msg = ""

    # Parse the STATUS:TYPE:MSG line from stdout.
    if stdout:
        first_line = stdout.strip().splitlines()[0] if stdout.strip() else ""
        if first_line:
            parts = first_line.split(":", 2)
            if parts:
                status = parts[0] or status
                if len(parts) >= 2:
                    error_type = parts[1]
                if len(parts) >= 3:
                    error_msg = parts[2]

    # Fall back to stderr for infra-style failures.
    if not error_msg and stderr:
        if status == STATUS_CRASH:
            error_type = "Stderr"
            error_msg = _truncate(stderr.strip().splitlines()[-1] if stderr.strip() else "")
        elif status in ("test_def_error", "test_no_check"):
            error_msg = _truncate(stderr)

    passed = status == STATUS_PASSED
    return CodeJudgeResult(
        passed=passed,
        status=status,
        error_type=error_type,
        error_message=error_msg,
        exit_code=exit_code,
        duration_ms=round(duration_ms, 2),
        candidate_code=candidate_code,
        test_code=test_code,
        entry_point=entry_point,
    )


# =============================================================================
# Public API
# =============================================================================
_RESULT_CACHE: dict[str, CodeJudgeResult] = {}
_CACHE_LOCK_NEEDED = False  # dict ops are GIL-atomic; ThreadPoolExecutor is safe.


def _cache_key(candidate_code: str, test_code: str, entry_point: str) -> str:
    h = hashlib.sha256()
    h.update(candidate_code.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(test_code.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(entry_point.encode("utf-8", "replace"))
    return h.hexdigest()


def judge_code_prediction(
    candidate_code: str,
    test_code: str,
    entry_point: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    use_cache: bool = True,
    python_exe: str | None = None,
) -> CodeJudgeResult:
    """Judge one candidate against one problem's tests.

    ``candidate_code`` is the model's raw rollout (markdown fences stripped).
    ``test_code`` is the problem's ``def check(candidate):`` block.
    ``entry_point`` is the function name the test calls.
    """
    candidate_code = extract_code(candidate_code) if candidate_code else ""
    if not test_code or not test_code.strip():
        return CodeJudgeResult(
            passed=False,
            status=STATUS_NO_TEST,
            error_type="NoTest",
            error_message="problem has no executable test",
            candidate_code=candidate_code,
            test_code=test_code,
            entry_point=entry_point,
        )
    if not candidate_code.strip():
        return CodeJudgeResult(
            passed=False,
            status=STATUS_COMPILE_ERROR,
            error_type="EmptyCode",
            error_message="empty candidate code",
            candidate_code=candidate_code,
            test_code=test_code,
            entry_point=entry_point,
        )

    if use_cache:
        key = _cache_key(candidate_code, test_code, entry_point)
        cached = _RESULT_CACHE.get(key)
        if cached is not None:
            return cached

    result = _run_harness(
        candidate_code, test_code, entry_point,
        timeout=timeout, cpu_seconds=cpu_seconds, python_exe=python_exe,
    )
    if use_cache:
        _RESULT_CACHE[key] = result
    return result


def judge_code_predictions_batch(
    items: Iterable[dict[str, Any]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    use_cache: bool = True,
) -> list[CodeJudgeResult]:
    """Judge many (candidate, test, entry_point) triples in a thread pool.

    Each item is a dict with keys ``candidate_code``, ``test_code``,
    ``entry_point``. Returns a list of ``CodeJudgeResult`` in input order.
    """
    items = list(items)
    if not items:
        return []
    workers = max(1, min(max_workers, len(items)))
    if workers == 1:
        return [
            judge_code_prediction(
                it.get("candidate_code", ""),
                it.get("test_code", ""),
                it.get("entry_point", ""),
                timeout=timeout, cpu_seconds=cpu_seconds, use_cache=use_cache,
            )
            for it in items
        ]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                judge_code_prediction,
                it.get("candidate_code", ""),
                it.get("test_code", ""),
                it.get("entry_point", ""),
                timeout=timeout, cpu_seconds=cpu_seconds, use_cache=use_cache,
            )
            for it in items
        ]
        return [f.result() for f in futures]


# =============================================================================
# Multi-candidate aggregation + difficulty
# =============================================================================
@dataclass
class CodeAggregation:
    """Aggregate verdict for one question across its rollout candidates."""

    question_id: str = ""
    total: int = 0
    passed_count: int = 0
    failed_count: int = 0
    pass_rate: float = 0.0
    # "全部通过才算对": correct iff every candidate passed.
    correct: bool = False
    difficulty: str = DIFFICULTY_UNKNOWN
    # Per-candidate status summary for diagnostics.
    statuses: list[str] = field(default_factory=list)
    first_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_code_results(
    question_id: str,
    results: list[CodeJudgeResult],
    *,
    has_test: bool = True,
) -> CodeAggregation:
    """Aggregate per-candidate results into a single question verdict.

    Difficulty (execution-only, no LLM):
      * all candidates pass  -> easy
      * some pass             -> medium
      * none pass             -> hard
      * no test               -> unknown

    Correctness: ALL candidates must pass ("全部通过才算对").
    """
    total = len(results)
    if not has_test or total == 0:
        return CodeAggregation(
            question_id=question_id,
            total=total,
            difficulty=DIFFICULTY_UNKNOWN if not has_test else DIFFICULTY_HARD,
            statuses=[r.status for r in results],
        )

    passed = [r for r in results if r.passed]
    passed_count = len(passed)
    failed_count = total - passed_count
    pass_rate = passed_count / total if total else 0.0
    correct = passed_count == total  # all pass

    if passed_count == total:
        difficulty = DIFFICULTY_EASY
    elif passed_count > 0:
        difficulty = DIFFICULTY_MEDIUM
    else:
        difficulty = DIFFICULTY_HARD

    first_error = ""
    for r in results:
        if not r.passed:
            first_error = f"{r.status}:{r.error_type}"
            break

    return CodeAggregation(
        question_id=question_id,
        total=total,
        passed_count=passed_count,
        failed_count=failed_count,
        pass_rate=round(pass_rate, 4),
        correct=correct,
        difficulty=difficulty,
        statuses=[r.status for r in results],
        first_error=first_error,
    )


def code_difficulty_from_results(
    results: list[CodeJudgeResult], *, has_test: bool = True
) -> str:
    """Return easy/medium/hard/unknown from per-candidate execution results."""
    if not has_test or not results:
        return DIFFICULTY_UNKNOWN if not has_test else DIFFICULTY_HARD
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    if passed == total:
        return DIFFICULTY_EASY
    if passed > 0:
        return DIFFICULTY_MEDIUM
    return DIFFICULTY_HARD


def clear_cache() -> None:
    """Drop the in-memory judge cache (e.g. between rounds)."""
    _RESULT_CACHE.clear()


def cache_info() -> dict[str, int]:
    return {"size": len(_RESULT_CACHE)}
