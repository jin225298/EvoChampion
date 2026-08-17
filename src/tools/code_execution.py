"""Code execution judging for the code-domain self-evolution loop.

Correctness is decided ONLY by executing candidate code against executable
tests in a sandboxed subprocess — never by LLM subjective judgment and never
by reading the code. All verdicts are 0/1 derived from real test execution.

Design contract (hard requirements from the task):
  * Sandbox execution: untrusted candidate code runs in an isolated subprocess
    with CPU/memory resource limits and a wall-clock timeout, so a runaway or
    malicious candidate cannot affect the judge process.
  * Test construction: a harness is built from (candidate code + test +
    entry_point), supporting HumanEval-style ``def check(candidate): assert ...``
    tests. The harness never embeds the gold/reference answer.
  * Multi-candidate aggregation: several rollout candidates for one question
    are judged uniformly; pass count / pass rate are aggregated and the
    "all tests pass" rule decides each candidate's verdict.
  * Failure diagnosis: compile error / assertion failure / timeout / runtime
    error / process crash are distinguished and reported.
  * Difficulty estimation: defined ONLY by test execution results —
    all pass = easy, partial = medium, all fail = hard, no test = unknown.
  * Anti-pollution: test construction contains no gold answer; no LLM judges
    behavioral equivalence.
  * Caching + parallelism: judge results are cached and a thread pool judges
    multiple candidates concurrently.

Public API
----------
``judge_code_candidate(prediction, *, test, entry_point, ...) -> dict``
    Judge one model rollout. Returns a structured judgement dict whose core
    keys (``correct``, ``score``, ``reason``, ``source``) match the shape used
    by ``difficulty_tagger`` / ``evaluator`` so it is a drop-in for the math
    ``judge_answer`` path, plus extra code-specific fields.
``judge_code_batch(predictions, *, tests, entry_points, ...) -> list[dict]``
    Thread-pool parallel judging with result caching.
``aggregate_code_candidates(results) -> dict``
    Aggregate multiple candidates for one question.
``code_difficulty_from_results(passed, total) -> str``
    easy / medium / hard / unknown from test pass counts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SandboxResult",
    "CodeJudgeResult",
    "extract_code",
    "build_harness",
    "run_sandboxed",
    "judge_code_candidate",
    "judge_code_batch",
    "aggregate_code_candidates",
    "code_difficulty_from_results",
    "is_code_exec_item",
    "EVALUATION_METHOD_CODE_EXEC",
    "FAILURE_KIND_NONE",
    "FAILURE_KIND_NO_TEST",
    "FAILURE_KIND_COMPILE",
    "FAILURE_KIND_ASSERTION",
    "FAILURE_KIND_TIMEOUT",
    "FAILURE_KIND_RUNTIME",
    "FAILURE_KIND_CRASH",
]


# Evaluation-method tag attached to code-domain questions so the rollout and
# evaluator judging paths route them to execution-based judging instead of the
# symbolic math ``judge_answer`` path.
EVALUATION_METHOD_CODE_EXEC = "code_exec"


def is_code_exec_item(item: Any) -> bool:
    """True when a question item should be judged by executing its code.

    A code item carries an executable ``test`` and ``entry_point``. Items
    tagged ``evaluation_method == "code_exec"`` are always code items.
    """
    if not isinstance(item, dict):
        return False
    if str(item.get("evaluation_method") or "").strip().lower() == EVALUATION_METHOD_CODE_EXEC:
        return True
    test = item.get("test") or item.get("code_test")
    entry_point = item.get("entry_point") or item.get("entry_point_func")
    return bool(test) and bool(entry_point)


# ---------------------------------------------------------------------------
# Failure-kind constants (single source of truth for structured diagnosis)
# ---------------------------------------------------------------------------
FAILURE_KIND_NONE = "none"            # all tests passed
FAILURE_KIND_NO_TEST = "no_test"      # no executable test available → unknown
FAILURE_KIND_COMPILE = "compile_error"
FAILURE_KIND_ASSERTION = "assertion_failed"
FAILURE_KIND_TIMEOUT = "timeout"
FAILURE_KIND_RUNTIME = "runtime_error"
FAILURE_KIND_CRASH = "crash"          # killed by signal / non-zero exit w/o known cause

_DEFAULT_TIMEOUT_SECONDS = float(os.getenv("CODE_EXEC_TIMEOUT_SECONDS", "10"))
_DEFAULT_MAX_MEMORY_MB = int(os.getenv("CODE_EXEC_MAX_MEMORY_MB", "512"))
_DEFAULT_CPU_SECONDS = int(os.getenv("CODE_EXEC_CPU_SECONDS", "15"))
_JUDGE_SOURCE = "code_execution"

# Marks stderr lines that indicate a specific failure class. Order matters:
# compile errors are detected first (SyntaxError/IndentationError), then the
# subprocess outcome classifies assertion / runtime / timeout / crash.
_COMPILE_MARKERS = ("SyntaxError", "IndentationError", "TabError")
_ASSERTION_MARKERS = ("AssertionError", "AssertionError:")
_TIMEOUT_MARKERS = ("TimeoutExpired",)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SandboxResult:
    """Outcome of running one harness in the sandbox subprocess."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    execution_time_ms: int = 0
    signal: int | None = None
    parsed_counts: dict[str, int] | None = None  # {"passes": p, "failures": f} when counted

    @property
    def crashed(self) -> bool:
        # Negative returncode means the process was killed by a signal.
        return self.returncode < 0 or self.signal is not None


@dataclass
class CodeJudgeResult:
    """Structured judgement for one candidate. ``to_dict`` yields the shape
    consumed by the math-domain judge pipeline (``correct``/``score``/``reason``/
    ``source``) plus code-specific fields."""

    correct: bool
    score: float
    reason: str
    source: str = _JUDGE_SOURCE
    passed_tests: int = 0
    total_tests: int = 0
    pass_rate: float = 0.0
    difficulty: str = "unknown"
    failure_kind: str = FAILURE_KIND_NONE
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    execution_time_ms: int = 0
    # Fields kept for parity with the llm_judge / difficulty_tagger judgement
    # dict shape so the result flows through the existing pipeline unchanged.
    judge_raw_text: str = ""
    fallback_used: bool = False
    schema_errors: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": bool(self.correct),
            "score": float(self.score),
            "reason": self.reason,
            "source": self.source,
            "passed_tests": int(self.passed_tests),
            "total_tests": int(self.total_tests),
            "pass_rate": float(self.pass_rate),
            "difficulty": self.difficulty,
            "failure_kind": self.failure_kind,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": int(self.returncode),
            "execution_time_ms": int(self.execution_time_ms),
            "judge_raw_text": self.judge_raw_text,
            "fallback_used": bool(self.fallback_used),
            "schema_errors": list(self.schema_errors),
        }


# ---------------------------------------------------------------------------
# Difficulty — defined ONLY by test execution results
# ---------------------------------------------------------------------------
def code_difficulty_from_results(passed: int, total: int) -> str:
    """Map test pass counts to difficulty.

    all pass → easy, partial → medium, all fail → hard, no test → unknown.
    """
    try:
        passed = int(passed)
        total = int(total)
    except (TypeError, ValueError):
        return "unknown"
    if total <= 0:
        return "unknown"
    if passed >= total:
        return "easy"
    if passed <= 0:
        return "hard"
    return "medium"


# ---------------------------------------------------------------------------
# Code extraction from model rollouts
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(
    r"```(?:python|py|Python)?\s*\n(.*?)```",
    re.DOTALL,
)


def _strip_fences(text: str) -> str:
    """Return code from the first ```python ... ``` fenced block, if any."""
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_code(prediction: str, entry_point: str | None = None) -> str:
    """Extract the candidate Python code from a model rollout.

    Handles markdown fences and, when ``entry_point`` is known, prefers the
    block that actually defines ``def <entry_point>(``. Falls back to the full
    (de-fenced) text so partial / unformatted outputs are still judged rather
    than silently dropped.
    """
    if not prediction:
        return ""
    text = _strip_fences(prediction)

    if entry_point:
        # Prefer a fenced block that defines the entry point.
        for match in _FENCE_RE.finditer(prediction):
            block = match.group(1)
            if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", block, re.MULTILINE):
                return block.strip()
        # Otherwise try to slice from the def line to the end of the text.
        def_match = re.search(
            rf"^[ \t]*def\s+{re.escape(entry_point)}\s*\(",
            text,
            re.MULTILINE,
        )
        if def_match:
            return text[def_match.start():].strip()

    return text


# ---------------------------------------------------------------------------
# Harness construction (gold-free — never embeds the reference answer)
# ---------------------------------------------------------------------------
def build_harness(candidate_code: str, test_code: str, entry_point: str) -> str:
    """Build a runnable harness from candidate code + test + entry_point.

    The harness defines the candidate, defines the test's ``check`` function,
    and calls ``check(<entry_point>)``. It contains NO gold/reference answer.
    """
    parts: list[str] = []
    if candidate_code:
        parts.append(candidate_code.strip())
    if test_code:
        parts.append(test_code.strip())
    call = f"check({entry_point})" if entry_point else "check()"
    parts.append(call)
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# AST-based per-assert counting harness
# ---------------------------------------------------------------------------
class _AssertCounter(ast.NodeTransformer):
    """Wrap each top-level ``assert`` in the check function so per-test
    pass/fail can be counted in a single sandbox run."""

    def __init__(self) -> None:
        self.assert_count = 0

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        self.assert_count += 1
        idx = self.assert_count
        # try: <assert>; _passes += 1
        # except AssertionError: _failures += 1
        return ast.Try(
            body=[
                node,
                ast.AugAssign(
                    target=ast.Name(id="_passes", ctx=ast.Store()),
                    op=ast.Add(),
                    value=ast.Constant(value=1),
                ),
            ],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id="AssertionError", ctx=ast.Load()),
                    name=None,
                    body=[
                        ast.AugAssign(
                            target=ast.Name(id="_failures", ctx=ast.Store()),
                            op=ast.Add(),
                            value=ast.Constant(value=1),
                        ),
                    ],
                )
            ],
            orelse=[],
            finalbody=[],
        )


def _build_counting_harness(
    candidate_code: str,
    test_code: str,
    entry_point: str,
) -> tuple[str, int]:
    """Build a harness that counts per-assert pass/fail and prints JSON.

    Returns (harness_source, total_asserts). If the check function has no
    top-level asserts, returns ("", 0) so the caller falls back to the
    whole-check harness.
    """
    try:
        test_tree = ast.parse(test_code or "")
    except SyntaxError:
        return "", 0

    check_fn: ast.FunctionDef | None = None
    for node in test_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_fn = node
            break
    if check_fn is None:
        # Fall back to the first function defined in the test (some datasets
        # name it differently); still require it to take a `candidate` arg.
        for node in test_tree.body:
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args]
                if "candidate" in args or len(args) >= 1:
                    check_fn = node
                    break
    if check_fn is None:
        return "", 0

    counter = _AssertCounter()
    counter.visit(check_fn)
    total = counter.assert_count
    if total <= 0:
        return "", 0

    # Rebuild the check function with counters initialized before its body and
    # a JSON report printed after.
    new_body: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id="_passes", ctx=ast.Store())],
            value=ast.Constant(value=0),
        ),
        ast.Assign(
            targets=[ast.Name(id="_failures", ctx=ast.Store())],
            value=ast.Constant(value=0),
        ),
    ]
    new_body.extend(check_fn.body)
    new_body.append(
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="__codejudge_report__", ctx=ast.Load()),
                    attr="report",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Name(id="_passes", ctx=ast.Load()),
                    ast.Name(id="_failures", ctx=ast.Load()),
                ],
                keywords=[],
            )
        )
    )
    check_fn.body = new_body
    ast.fix_missing_locations(test_tree)

    preamble = (
        "import json as _json\n"
        "class __codejudge_report__:\n"
        "    @staticmethod\n"
        "    def report(passes, failures):\n"
        "        print('__CODEJUDGE_RESULT__' + _json.dumps({'passes': passes, 'failures': failures}))\n"
    )
    try:
        test_src = ast.unparse(test_tree)
    except Exception:
        return "", 0
    harness = "\n\n".join(
        s for s in [candidate_code.strip(), preamble, test_src] if s
    )
    harness += f"\n\ncheck({entry_point})\n" if entry_point else "\n\ncheck()\n"
    return harness, total


_RESULT_LINE_RE = re.compile(r"__CODEJUDGE_RESULT__(\{.*\})")


def _parse_counted_result(stdout: str) -> dict[str, int] | None:
    match = _RESULT_LINE_RE.search(stdout or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return {
            "passes": int(data.get("passes", 0)),
            "failures": int(data.get("failures", 0)),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Sandbox subprocess execution
# ---------------------------------------------------------------------------
def _preexec_limits(max_memory_mb: int, cpu_seconds: int):
    """Build a preexec_fn that applies resource limits (Linux only)."""
    if sys.platform == "win32":
        return None
    try:
        import resource  # POSIX-only
    except ImportError:
        return None

    def _limit() -> None:  # pragma: no cover - exercised on the server
        # New process group so we can kill the whole tree on timeout.
        try:
            os.setsid()
        except Exception:
            pass
        # Memory cap (address space). Bytes.
        mem_bytes = int(max_memory_mb) * 1024 * 1024
        if mem_bytes > 0:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass
        # CPU time cap (seconds).
        if cpu_seconds > 0:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            except (ValueError, OSError):
                pass
        # No core dumps.
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass

    return _limit


def run_sandboxed(
    code: str,
    *,
    timeout: float | None = None,
    max_memory_mb: int | None = None,
    cpu_seconds: int | None = None,
    python_executable: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> SandboxResult:
    """Run ``code`` in an isolated subprocess with resource limits + timeout.

    The judge process never imports or executes the candidate code directly;
    it only spawns a subprocess and reads its stdout/stderr/returncode.
    """
    timeout = float(_DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout)
    max_memory_mb = int(_DEFAULT_MAX_MEMORY_MB if max_memory_mb is None else max_memory_mb)
    cpu_seconds = int(_DEFAULT_CPU_SECONDS if cpu_seconds is None else cpu_seconds)
    python_executable = python_executable or sys.executable

    tmp_dir = Path(tempfile.mkdtemp(prefix="codejudge_"))
    script_path = tmp_dir / "harness.py"
    try:
        script_path.write_text(code, encoding="utf-8")
    except OSError:
        return SandboxResult(returncode=-1, stderr="failed to write harness to temp dir")

    env = dict(os.environ)
    # Keep the sandbox minimal and deterministic.
    env["PYTHONPATH"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    preexec = _preexec_limits(max_memory_mb, cpu_seconds)
    import time

    start = time.monotonic()
    proc = subprocess.Popen(
        [python_executable, "-I", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(tmp_dir),
        env=env,
        preexec_fn=preexec,
    )
    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return SandboxResult(
            returncode=proc.returncode,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            timed_out=False,
            execution_time_ms=elapsed_ms,
            signal=None,
        )
    except subprocess.TimeoutExpired:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        # Kill the whole process group (preexec did setsid, so the child's
        # pgid equals its pid). Never use pgid 0 — that targets our own group.
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
        try:
            stdout_b, stderr_b = proc.communicate(timeout=5)
        except Exception:
            stdout_b, stderr_b = b"", b""
        return SandboxResult(
            returncode=-9,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            timed_out=True,
            execution_time_ms=elapsed_ms,
            signal=9,
        )
    except Exception as exc:  # pragma: no cover - defensive
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return SandboxResult(
            returncode=-1,
            stderr=f"sandbox launch failed: {exc}",
            timed_out=False,
            execution_time_ms=elapsed_ms,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _classify_failure(sb: SandboxResult) -> str:
    """Classify a non-passing sandbox result into a failure kind."""
    if sb.timed_out:
        return FAILURE_KIND_TIMEOUT
    stderr = sb.stderr or ""
    stdout = sb.stdout or ""
    combined = stderr + "\n" + stdout
    if any(marker in combined for marker in _COMPILE_MARKERS):
        return FAILURE_KIND_COMPILE
    if any(marker in combined for marker in _ASSERTION_MARKERS):
        return FAILURE_KIND_ASSERTION
    if sb.returncode < 0 or sb.signal is not None:
        return FAILURE_KIND_CRASH
    if sb.returncode != 0:
        return FAILURE_KIND_RUNTIME
    # returncode 0 but no counted result and not all-pass → treat as runtime.
    return FAILURE_KIND_RUNTIME


# ---------------------------------------------------------------------------
# Single-candidate judging
# ---------------------------------------------------------------------------
def _has_executable_test(test: str, entry_point: str | None) -> bool:
    if not test or not str(test).strip():
        return False
    if not entry_point or not str(entry_point).strip():
        return False
    return True


def judge_code_candidate(
    prediction: str,
    *,
    test: str,
    entry_point: str,
    gold_answer: str = "",
    question_text: str = "",
    timeout: float | None = None,
    max_memory_mb: int | None = None,
    cpu_seconds: int | None = None,
) -> dict[str, Any]:
    """Judge one model rollout by executing it against the test.

    Returns a structured dict (see ``CodeJudgeResult.to_dict``). When there is
    no executable test, returns ``correct=False`` with ``failure_kind=no_test``
    and ``difficulty=unknown`` — never a pass.
    """
    if not _has_executable_test(test, entry_point):
        return CodeJudgeResult(
            correct=False,
            score=0.0,
            reason="no executable test; cannot judge by execution",
            difficulty="unknown",
            failure_kind=FAILURE_KIND_NO_TEST,
        ).to_dict()

    candidate_code = extract_code(prediction, entry_point)
    if not candidate_code:
        return CodeJudgeResult(
            correct=False,
            score=0.0,
            reason="empty candidate code after extraction",
            difficulty="hard",
            failure_kind=FAILURE_KIND_COMPILE,
        ).to_dict()

    # 1) Try the counting harness (per-assert pass/fail in one run).
    counting_harness, total = _build_counting_harness(candidate_code, test, entry_point)
    if counting_harness and total > 0:
        sb = run_sandboxed(
            counting_harness,
            timeout=timeout,
            max_memory_mb=max_memory_mb,
            cpu_seconds=cpu_seconds,
        )
        counts = _parse_counted_result(sb.stdout)
        if counts is not None:
            passed = int(counts.get("passes", 0))
            failures = int(counts.get("failures", 0))
            total_tests = passed + failures if (passed + failures) > 0 else total
            correct = total_tests > 0 and passed >= total_tests
            difficulty = code_difficulty_from_results(passed, total_tests)
            if correct:
                failure_kind = FAILURE_KIND_NONE
                reason = f"all {total_tests} test(s) passed"
            else:
                # The counting harness only catches AssertionError per assert,
                # so any counted failure is an assertion failure. A non-assert
                # exception would have crashed the harness (counts is None) and
                # we would have fallen through to the whole-check path.
                failure_kind = FAILURE_KIND_ASSERTION
                reason = f"{passed}/{total_tests} test(s) passed"
            return CodeJudgeResult(
                correct=correct,
                score=1.0 if correct else 0.0,
                reason=reason,
                passed_tests=passed,
                total_tests=total_tests,
                pass_rate=(passed / total_tests) if total_tests else 0.0,
                difficulty=difficulty,
                failure_kind=failure_kind,
                stdout=sb.stdout,
                stderr=sb.stderr,
                returncode=sb.returncode,
                execution_time_ms=sb.execution_time_ms,
            ).to_dict()
        # Counting harness produced no parseable result (crash/timeout/compile).
        # Fall through to whole-check classification.

    # 2) Whole-check harness (covers tests without top-level asserts, or when
    #    the counting harness could not parse a result).
    harness = build_harness(candidate_code, test, entry_point)
    sb = run_sandboxed(
        harness,
        timeout=timeout,
        max_memory_mb=max_memory_mb,
        cpu_seconds=cpu_seconds,
    )

    if sb.returncode == 0 and not sb.timed_out:
        # All tests passed (check() ran without raising).
        return CodeJudgeResult(
            correct=True,
            score=1.0,
            reason="all tests passed",
            passed_tests=1,
            total_tests=1,
            pass_rate=1.0,
            difficulty="easy",
            failure_kind=FAILURE_KIND_NONE,
            stdout=sb.stdout,
            stderr=sb.stderr,
            returncode=sb.returncode,
            execution_time_ms=sb.execution_time_ms,
        ).to_dict()

    failure_kind = _classify_failure(sb)
    # No-test is impossible here (guarded above); map crash/timeout/etc.
    if failure_kind == FAILURE_KIND_NONE:
        failure_kind = FAILURE_KIND_RUNTIME
    difficulty = "hard"  # all tests failed → hard (no partial info available)
    reason = {
        FAILURE_KIND_COMPILE: "candidate code failed to compile",
        FAILURE_KIND_ASSERTION: "at least one assertion failed",
        FAILURE_KIND_TIMEOUT: "execution timed out",
        FAILURE_KIND_RUNTIME: "runtime error during execution",
        FAILURE_KIND_CRASH: "process crashed (killed by signal)",
    }.get(failure_kind, "execution failed")
    return CodeJudgeResult(
        correct=False,
        score=0.0,
        reason=reason,
        passed_tests=0,
        total_tests=total if total else 1,
        pass_rate=0.0,
        difficulty=difficulty,
        failure_kind=failure_kind,
        stdout=sb.stdout,
        stderr=sb.stderr,
        returncode=sb.returncode,
        execution_time_ms=sb.execution_time_ms,
    ).to_dict()


# ---------------------------------------------------------------------------
# Batch judging with caching + thread pool
# ---------------------------------------------------------------------------
_JUDGE_CACHE: dict[str, dict[str, Any]] = {}
_JUDGE_CACHE_LOCK = threading.Lock()
_JUDGE_EXECUTOR: ThreadPoolExecutor | None = None
_JUDGE_EXECUTOR_LOCK = threading.Lock()


def _cache_key(prediction: str, test: str, entry_point: str) -> str:
    raw = "\x00".join([prediction or "", test or "", entry_point or ""])
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    global _JUDGE_EXECUTOR
    with _JUDGE_EXECUTOR_LOCK:
        if _JUDGE_EXECUTOR is None:
            _JUDGE_EXECUTOR = ThreadPoolExecutor(
                max_workers=max(1, int(max_workers)),
                thread_name_prefix="codejudge",
            )
        return _JUDGE_EXECUTOR


def judge_code_batch(
    predictions: list[str],
    *,
    tests: list[str],
    entry_points: list[str],
    gold_answers: list[str] | None = None,
    question_texts: list[str] | None = None,
    timeout: float | None = None,
    max_memory_mb: int | None = None,
    cpu_seconds: int | None = None,
    max_workers: int | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Judge many candidates concurrently with result caching.

    Returns a list of judgement dicts aligned with ``predictions``.
    """
    if not predictions:
        return []
    n = len(predictions)
    gold_answers = gold_answers or [""] * n
    question_texts = question_texts or [""] * n
    if max_workers is None:
        max_workers = min(8, max(1, (os.cpu_count() or 4)))

    def _one(idx: int) -> dict[str, Any]:
        pred = predictions[idx] if idx < len(predictions) else ""
        test = tests[idx] if idx < len(tests) else ""
        entry = entry_points[idx] if idx < len(entry_points) else ""
        gold = gold_answers[idx] if idx < len(gold_answers) else ""
        qtext = question_texts[idx] if idx < len(question_texts) else ""
        key = _cache_key(pred, test, entry)
        if use_cache:
            with _JUDGE_CACHE_LOCK:
                cached = _JUDGE_CACHE.get(key)
            if cached is not None:
                return dict(cached)
        result = judge_code_candidate(
            pred,
            test=test,
            entry_point=entry,
            gold_answer=gold,
            question_text=qtext,
            timeout=timeout,
            max_memory_mb=max_memory_mb,
            cpu_seconds=cpu_seconds,
        )
        if use_cache:
            with _JUDGE_CACHE_LOCK:
                _JUDGE_CACHE[key] = dict(result)
        return result

    executor = _get_executor(max_workers)
    return list(executor.map(_one, range(n)))


def clear_code_judge_cache() -> None:
    """Drop all cached judge results (e.g. between rounds)."""
    with _JUDGE_CACHE_LOCK:
        _JUDGE_CACHE.clear()


# ---------------------------------------------------------------------------
# Multi-candidate aggregation (one question, several rollouts)
# ---------------------------------------------------------------------------
def aggregate_code_candidates(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-candidate judgements for one question.

    Each candidate is correct only if ALL its tests pass. The aggregation
    reports how many candidates passed, the pass rate, and the worst-case
    difficulty across candidates (all-fail dominates).
    """
    if not results:
        return {
            "correct": False,
            "pass_count": 0,
            "candidate_count": 0,
            "pass_rate": 0.0,
            "difficulty": "unknown",
            "failure_kind": FAILURE_KIND_NO_TEST,
        }
    pass_count = sum(1 for r in results if bool(r.get("correct")))
    total = len(results)
    # Difficulty: take the hardest among candidates so a question with any
    # all-fail candidate is not mislabeled easy.
    diff_order = {"easy": 0, "medium": 1, "hard": 2, "unknown": 3}
    diffs = [str(r.get("difficulty", "unknown")) for r in results]
    worst = max(diffs, key=lambda d: diff_order.get(d, 3))
    if pass_count == 0:
        # No candidate fully passed → difficulty is hard unless tests missing.
        if any(str(r.get("failure_kind")) == FAILURE_KIND_NO_TEST for r in results):
            worst = "unknown"
        else:
            worst = "hard"
    return {
        "correct": pass_count == total,
        "pass_count": pass_count,
        "candidate_count": total,
        "pass_rate": (pass_count / total) if total else 0.0,
        "difficulty": worst,
        "failure_kind": results[0].get("failure_kind", FAILURE_KIND_NONE),
    }


# ---------------------------------------------------------------------------
# CLI for local smoke testing (no model needed)
# ---------------------------------------------------------------------------
def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Code execution judge smoke test")
    parser.add_argument("--prediction", help="Candidate code (or @file)")
    parser.add_argument("--test", required=True, help="Test code with check(candidate) (or @file)")
    parser.add_argument("--entry-point", required=True)
    args = parser.parse_args()

    def _read(val: str) -> str:
        if val and val.startswith("@"):
            return Path(val[1:]).read_text(encoding="utf-8")
        return val or ""

    result = judge_code_candidate(
        _read(args.prediction),
        test=_read(args.test),
        entry_point=args.entry_point,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("correct") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
