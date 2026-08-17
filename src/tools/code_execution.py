"""Code-domain judging: correctness is decided by *executing tests*, never by an
LLM reading the code and judging behavioral equivalence.

Principles:
- All tests pass                      -> correct
- Any failure / timeout / compile
  error / runtime exception / crash  -> incorrect
- No executable test                 -> not judged; reported as unknown/unavailable

The judge pipeline:  sandbox execution + test harness construction + multi-candidate
aggregation + failure diagnosis + difficulty estimation + anti-contamination
(no gold answer injected) + memoised result cache + thread-pool parallelism.

Module layout:
- extract_code          strip fences / think-tags / surrounding prose
- build_harness         solution + test + entry_point -> executable script
- run_isolated          subprocess sandbox (cpu/mem/fs limits + timeout + tmp cwd)
- _diagnose_failure     classify stderr/returncode into failure kinds
- judge_single          one candidate -> CodeJudgeResult
- judge_multi           N candidates -> aggregated verdict + difficulty
- code_judge_answer     boolean wrapper compatible with judge_answer()
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
DOMAIN = os.getenv("DOMAIN", "").strip().lower()
IS_CODE_DOMAIN = DOMAIN == "code"

CODE_EXEC_TIMEOUT_SECONDS = float(os.getenv("CODE_EXEC_TIMEOUT_SECONDS", "10"))
CODE_EXEC_MAX_MEMORY_MB = int(os.getenv("CODE_EXEC_MAX_MEMORY_MB", "512"))
CODE_EXEC_MAX_OUTPUT_BYTES = int(os.getenv("CODE_EXEC_MAX_OUTPUT_BYTES", "8192"))
CODE_EXEC_MAX_WORKERS = int(os.getenv("CODE_EXEC_MAX_WORKERS", "8"))
CODE_EXEC_CACHE_ENABLED = os.getenv("CODE_EXEC_CACHE_ENABLED", "1") in ("1", "true", "yes")
CODE_EXEC_CACHE_DIR = os.getenv("CODE_EXEC_CACHE_DIR", "")

# Importing these modules from a candidate solution could read files, spawn
# processes, or hit the network inside our sandbox. Blocked before execution.
IMPORT_BLOCKLIST = {
    "os", "sys", "subprocess", "pathlib", "shutil", "socket", "urllib",
    "requests", "http", "ftplib", "telnetlib", "paramiko", "ctypes",
    "multiprocessing", "threading", "signal", "resource",
}

# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------
KIND_PASS = "pass"
KIND_FAIL = "fail"                 # assertion-level test failure
KIND_COMPILE_ERROR = "compile_error"
KIND_TIMEOUT = "timeout"
KIND_RUNTIME_ERROR = "runtime_error"
KIND_CRASH = "crash"
KIND_NO_TESTS = "no_tests"
KIND_FORBIDDEN_IMPORT = "forbidden_import"
KIND_UNKNOWN = "unknown"


@dataclass
class CodeJudgeResult:
    """Structured result for a single candidate against one test set."""
    correct: bool
    status: str                      # KIND_* value
    passed: int = 0
    failed: int = 0
    total: int = 0
    failure_kind: str = ""
    failure_message: str = ""
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    cached: bool = False
    reason: str = ""


@dataclass
class CodeAggregate:
    """Aggregated verdict over multiple candidates for the same question."""
    correct: bool
    n_candidates: int
    n_passed: int
    n_failed: int
    pass_rate: float
    status: str
    difficulty: str                  # easy/medium/hard/unknown
    reason: str = ""
    per_candidate: list[CodeJudgeResult] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Code extraction
# -----------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_THINK_RE = re.compile(
    r"<\s*(?:think|final|reasoning)\s*>(.*?)<\s*/\s*(?:think|final|reasoning)\s*>",
    re.DOTALL,
)


def _looks_like_python(body: str) -> bool:
    """Best-effort 'is this code and not prose' check.

    Any def/class/import/from line marks it as code even when it has a syntax
    error (the error is then diagnosed at execution time as a compile error,
    rather than the whole candidate being discarded as 'no code').
    """
    if not body:
        return False
    if re.search(r"^\s*(?:def|class|import|from)\s", body, re.MULTILINE):
        return True
    try:
        ast.parse(body)
        return True
    except SyntaxError:
        return False


def extract_code(text: str) -> str:
    """Pull the first Python code block out of a model response.

    Prefers an explicit ```python ... ``` fence; otherwise drops think/reasoning
    tags and trims leading prose up to the first code construct. Fenced bodies
    are returned verbatim; unfenced bodies need def/class/import signs or valid
    python to be treated as code.
    """
    if not text:
        return ""
    stripped = str(text).strip()
    for match in _FENCE_RE.finditer(stripped):
        body = match.group(1).strip()
        if body:
            return body
    body = _THINK_RE.sub("", stripped)
    lines = body.splitlines()
    code_start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("def ", "class ", "import ", "from ")) or s.startswith("@"):
            code_start = i
            break
        if s.startswith("#") or not s:
            continue
    body = "\n".join(lines[code_start:]).strip()
    if _looks_like_python(body):
        return body
    return ""


# -----------------------------------------------------------------------------
# Entry-point resolution + harness construction
# -----------------------------------------------------------------------------
def _resolve_entry_point(test_code: str, item: dict) -> str | None:
    """Find the callable the test drives.

    Priority: item['entry_point'] (function name); else None. A `def check(`
    in the test with no entry_point cannot be driven (unknown candidate fn), so
    we return None and rely on the test referencing the function directly.
    """
    ep = item.get("entry_point") or item.get("function_name")
    if ep and isinstance(ep, str) and ep.strip():
        return ep.strip()
    return None


def build_harness(solution: str, test_code: str, entry_point: str | None) -> str:
    """Assemble one executable script: solution + tests (+ check call).

    The harness contains *only* solution + test. The gold answer is never part
    of the harness (anti-contamination). When the test defines `def check(`
    (HumanEval convention) and an entry_point is known, append `check(entry)`.
    """
    harness = f"{solution}\n\n# ===== tests =====\n{test_code}\n"
    if entry_point and "def check(" in test_code:
        harness += f"\ncheck({entry_point})\n"
    return harness


# -----------------------------------------------------------------------------
# Sandboxed execution
# -----------------------------------------------------------------------------
def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def run_isolated(
    script: str,
    *,
    timeout: float | None = None,
    max_memory_mb: int | None = None,
    max_output_bytes: int | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Run untrusted code in a subprocess with resource limits and timeout.

    Resource limits (CPU time, address space, file size) are applied in the
    child; the parent enforces wall-clock timeout and SIGKILLs the whole process
    group so stray children cannot leak. Runs in a throwaway temp cwd.
    """
    timeout = timeout if timeout is not None else CODE_EXEC_TIMEOUT_SECONDS
    max_memory_mb = max_memory_mb if max_memory_mb is not None else CODE_EXEC_MAX_MEMORY_MB
    max_output_bytes = max_output_bytes if max_output_bytes is not None else CODE_EXEC_MAX_OUTPUT_BYTES

    own_tmp = cwd is None
    cwd = Path(cwd) if cwd is not None else Path(tempfile.mkdtemp(prefix="codejudge_"))
    (cwd / "solution.py").write_text(script, encoding="utf-8")

    # Minimal env: no proxy / HF secrets / site-packages so candidate code cannot
    # silently reach the network or read credentials.
    env = {
        "PATH": os.getenv("PATH", "/usr/bin:/bin"),
        "HOME": str(cwd),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }

    # Limits are best-effort: some platforms (e.g. macOS) refuse to shrink the
    # RLIMIT_AS hard limit, which must not take down the whole judge. On Linux
    # (the training cluster) all limits apply and the sandbox is enforced.
    preamble = (
        f"import resource\n"
        f"def _setr(name, value):\n"
        f"    try:\n"
        f"        cur_soft, cur_hard = resource.getrlimit(name)\n"
        f"        if cur_hard == resource.RLIM_INFINITY or value <= cur_hard:\n"
        f"            resource.setrlimit(name, (value, value))\n"
        f"    except (ValueError, OSError):\n"
        f"        pass\n"
        f"_setr(resource.RLIMIT_CPU, {int(timeout * 2)})\n"
        f"_setr(resource.RLIMIT_AS, {max_memory_mb * 1024 * 1024})\n"
        f"_setr(resource.RLIMIT_FSIZE, {max_output_bytes * 2})\n"
        f"_setr(resource.RLIMIT_NOFILE, 128)\n"
        f"import runpy; runpy.run_path('solution.py', run_name='__main__')\n"
    )
    cmd = [sys.executable, "-I", "-c", preamble]
    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,   # own process group -> killpg on timeout
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc.pid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            stdout, stderr = proc.communicate()
        returncode = proc.returncode
    except Exception as e:  # subprocess spawn failure etc.
        return {
            "returncode": 2,
            "timed_out": False,
            "stdout": "",
            "stderr": f"judge framework error: {e}",
            "wall_ms": (time.monotonic() - t0) * 1000.0,
        }
    finally:
        if own_tmp:
            shutil.rmtree(cwd, ignore_errors=True)
    return {
        "returncode": returncode if not timed_out else -signal.SIGKILL,
        "timed_out": timed_out,
        "stdout": stdout[-max_output_bytes:],
        "stderr": stderr[-max_output_bytes:],
        "wall_ms": (time.monotonic() - t0) * 1000.0,
    }


# -----------------------------------------------------------------------------
# Failure diagnosis
# -----------------------------------------------------------------------------
def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:300]
    return ""


def _count_asserts(test_code: str) -> int:
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return 0
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))


def _diagnose_failure(out: dict[str, Any], test_count: int) -> tuple[str, str]:
    """Classify a failed run into a failure kind + one-line message."""
    rc = out["returncode"]
    stderr = out["stderr"] or ""
    stdout = out["stdout"] or ""
    combined = stderr + "\n" + stdout

    if out["timed_out"] or rc == -signal.SIGKILL:
        return KIND_TIMEOUT, "execution timed out"
    if rc < 0:
        return KIND_CRASH, f"process killed by signal {-rc}"
    if rc == 0:
        return KIND_PASS, "all tests passed"
    if re.search(r"\b(SyntaxError|IndentationError|TabError)\b", stderr):
        return KIND_COMPILE_ERROR, _first_line(stderr)
    if re.search(r"\b(ModuleNotFoundError|ImportError)\b", stderr):
        return KIND_COMPILE_ERROR, _first_line(stderr)
    if re.search(r"\bAssertionError\b", stderr):
        return KIND_FAIL, _first_line(stderr)
    if test_count == 0:
        return KIND_NO_TESTS, "no executable tests present"
    if re.search(r"\b(Error|Exception|Traceback)\b", combined):
        return KIND_RUNTIME_ERROR, _first_line(stderr) or "runtime error"
    return KIND_FAIL, _first_line(stderr) or f"exit code {rc}"


# -----------------------------------------------------------------------------
# Cache (result memoisation)
# -----------------------------------------------------------------------------
def _cache_key(solution: str, test_code: str, entry_point: str | None) -> str:
    h = hashlib.sha256()
    for part in (solution, test_code, entry_point or "", sys.version):
        h.update(part.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _cache_load(key: str) -> dict | None:
    if not CODE_EXEC_CACHE_ENABLED or not CODE_EXEC_CACHE_DIR:
        return None
    try:
        path = Path(CODE_EXEC_CACHE_DIR) / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _cache_store(key: str, payload: dict) -> None:
    if not CODE_EXEC_CACHE_ENABLED or not CODE_EXEC_CACHE_DIR:
        return
    try:
        cache_dir = Path(CODE_EXEC_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Single-candidate judge
# -----------------------------------------------------------------------------
def _solution_forbidden_imports(solution: str) -> str | None:
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in IMPORT_BLOCKLIST:
                    return alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in IMPORT_BLOCKLIST:
                return top
    return None


def judge_single(
    prediction: str,
    test_code: str,
    *,
    entry_point: str | None = None,
    timeout: float | None = None,
) -> CodeJudgeResult:
    """Judge one candidate against one test set by executing it."""
    if not test_code or not str(test_code).strip():
        return CodeJudgeResult(correct=False, status=KIND_NO_TESTS,
                               failure_kind=KIND_NO_TESTS,
                               reason="no executable test")
    solution = extract_code(prediction)
    if not solution:
        return CodeJudgeResult(correct=False, status=KIND_UNKNOWN,
                               failure_kind=KIND_UNKNOWN,
                               reason="no python code extracted")
    forbidden = _solution_forbidden_imports(solution)
    if forbidden:
        return CodeJudgeResult(correct=False, status=KIND_FORBIDDEN_IMPORT,
                               failure_kind=KIND_FORBIDDEN_IMPORT,
                               reason=f"forbidden import: {forbidden}")

    key = _cache_key(solution, test_code, entry_point)
    cached = _cache_load(key)
    if cached:
        return CodeJudgeResult(
            correct=cached["correct"], status=cached["status"],
            passed=cached["passed"], failed=cached["failed"], total=cached["total"],
            failure_kind=cached.get("failure_kind", ""),
            failure_message=cached.get("failure_message", ""),
            duration_ms=cached.get("duration_ms", 0.0), cached=True,
        )

    harness = build_harness(solution, test_code, entry_point)
    test_count = _count_asserts(test_code)
    out = run_isolated(harness, timeout=timeout)

    if out["returncode"] == 0 and not out["timed_out"]:
        result = CodeJudgeResult(
            correct=True, status=KIND_PASS,
            passed=test_count, total=test_count,
            failure_kind=KIND_PASS,
            stdout=out["stdout"], stderr=out["stderr"],
            duration_ms=out["wall_ms"],
        )
    else:
        kind, message = _diagnose_failure(out, test_count)
        result = CodeJudgeResult(
            correct=False, status=kind,
            passed=0, total=test_count,
            failure_kind=kind, failure_message=message,
            stdout=out["stdout"], stderr=out["stderr"],
            duration_ms=out["wall_ms"],
        )
    _cache_store(key, {
        "correct": result.correct, "status": result.status,
        "passed": result.passed, "failed": result.failed, "total": result.total,
        "failure_kind": result.failure_kind, "failure_message": result.failure_message,
        "duration_ms": result.duration_ms,
    })
    return result


# -----------------------------------------------------------------------------
# Multi-candidate aggregation + difficulty estimation
# -----------------------------------------------------------------------------
def estimate_difficulty(pass_rate: float, n_candidates: int) -> str:
    """Difficulty defined strictly by test-execution pass rate."""
    if n_candidates <= 0:
        return "unknown"
    if pass_rate >= 1.0:
        return "easy"
    if pass_rate > 0.0:
        return "medium"
    return "hard"


def judge_multi(
    predictions: list[str],
    test_code: str,
    *,
    entry_point: str | None = None,
    timeout: float | None = None,
    max_workers: int | None = None,
) -> CodeAggregate:
    """Judge N candidate solutions, aggregate, and estimate difficulty.

    Correctness rule: a question counts correct only when *all* candidates pass
    every test (conservative). pass_rate/difficulty come from execution only.
    """
    if not predictions:
        return CodeAggregate(correct=False, n_candidates=0, n_passed=0, n_failed=0,
                             pass_rate=0.0, status=KIND_NO_TESTS, difficulty="unknown",
                             reason="no candidate")
    max_workers = max_workers or CODE_EXEC_MAX_WORKERS
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(judge_single, p, test_code, entry_point=entry_point, timeout=timeout)
            for p in predictions
        ]
        results = [f.result() for f in futures]

    n_passed = sum(1 for r in results if r.correct)
    n_failed = len(results) - n_passed
    pass_rate = n_passed / len(results) if results else 0.0
    all_pass = n_failed == 0 and all(r.status == KIND_PASS for r in results)
    if all_pass:
        status = KIND_PASS
    elif any(r.status == KIND_FAIL for r in results):
        status = KIND_FAIL
    elif any(r.status == KIND_TIMEOUT for r in results):
        status = KIND_TIMEOUT
    elif any(r.status in (KIND_COMPILE_ERROR, KIND_FORBIDDEN_IMPORT) for r in results):
        status = KIND_COMPILE_ERROR
    elif any(r.status == KIND_RUNTIME_ERROR for r in results):
        status = KIND_RUNTIME_ERROR
    elif any(r.status == KIND_CRASH for r in results):
        status = KIND_CRASH
    else:
        status = KIND_UNKNOWN
    first_fail = next((r for r in results if not r.correct), None)
    reason = "all tests passed" if all_pass else (first_fail.reason if first_fail else "no test")
    return CodeAggregate(
        correct=all_pass,
        n_candidates=len(results),
        n_passed=n_passed,
        n_failed=n_failed,
        pass_rate=pass_rate,
        status=status,
        difficulty=estimate_difficulty(pass_rate, len(results)),
        reason=reason,
        per_candidate=results,
    )


# -----------------------------------------------------------------------------
# Public API (drop-in for judge_answer)
# -----------------------------------------------------------------------------
def is_code_item(item: Any) -> bool:
    """True when the item carries an executable test set."""
    if not isinstance(item, dict):
        return False
    test = item.get("test") or item.get("tests") or item.get("test_code")
    if not test or not str(test).strip():
        return False
    if item.get("entry_point") or item.get("function_name"):
        return True
    return "def check(" in str(test)


def test_code_for_item(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("test") or item.get("tests") or item.get("test_code") or "").strip()


def code_judge_answer(
    prediction: str,
    gold_answer: str = "",
    item: dict | None = None,
) -> CodeJudgeResult:
    """Judge a code candidate, extracting the test set from the item.

    gold_answer is accepted for signature compatibility but intentionally ignored
    for judging: correctness is defined by executed tests only.
    """
    item = item or {}
    test_code = test_code_for_item(item)
    entry_point = _resolve_entry_point(test_code, item)
    return judge_single(prediction, test_code, entry_point=entry_point)


def code_judge_multi(predictions: list[str], item: dict | None = None) -> CodeAggregate:
    item = item or {}
    test_code = test_code_for_item(item)
    entry_point = _resolve_entry_point(test_code, item)
    return judge_multi(predictions, test_code, entry_point=entry_point)


def code_judge_answer_bool(
    prediction: str,
    gold_answer: str = "",
    item: dict | None = None,
) -> bool:
    """Boolean verdict: True only when the executed test set fully passes."""
    return code_judge_answer(prediction, gold_answer, item).correct


# Compatible alias used by other nodes.
judge_code_answer = code_judge_answer_bool


# -----------------------------------------------------------------------------
# Domain dispatcher — the single chokepoint other nodes call.
# -----------------------------------------------------------------------------
def _item_is_code(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return is_code_item(item) or (IS_CODE_DOMAIN and bool(test_code_for_item(item)))


def judge_solution(
    prediction: str,
    gold_answer: str = "",
    item: dict | None = None,
    math_judge=None,
) -> bool:
    """Domain-aware boolean judgement used by evaluator / bootstrap / tagger.

    - Code items are judged by executing tests (never by LLM, never by gold-string
      comparison).
    - Under DOMAIN=code with no executable test, the item is not judged: it never
      counts as correct (unknown), satisfying the 'no test = not a pass' rule.
    - Otherwise defers to the supplied math judge (math-verify symbolic matching).
    """
    if _item_is_code(item):
        return code_judge_answer_bool(prediction, gold_answer, item)
    if IS_CODE_DOMAIN:
        return False
    if math_judge is not None:
        return bool(math_judge(prediction, gold_answer))
    return False


def judge_chunk_solutions(
    predictions: list[str],
    gold_answer: str,
    item: dict | None = None,
    math_judge=None,
) -> list[bool]:
    """Judge a batch of candidate solutions for one question (multi-candidate).

    Code items use judge_multi (executed tests + aggregation); math items use the
    symbolic judge per candidate.
    """
    if _item_is_code(item):
        agg = code_judge_multi(predictions, item)
        return [r.correct for r in agg.per_candidate]
    if IS_CODE_DOMAIN:
        return [False] * len(predictions)
    if math_judge is not None:
        return [bool(math_judge(p, gold_answer)) for p in predictions]
    return [False] * len(predictions)
