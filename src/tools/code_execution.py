"""Code execution judge for the EvoChampion system.

Provides sandboxed execution of candidate code against test cases,
with structured diagnostics, difficulty estimation, caching, and
parallel execution.

Design principles:
  - All correctness judgments come from actual test execution, never LLM
  - Test harness construction never includes the gold/reference solution
  - Every judgment returns a structured CodeExecutionResult
  - Results are cacheable and parallelizable via ThreadPoolExecutor
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Result type ──────────────────────────────────────────────────────────────

# Diagnosis constants
DIAG_COMPILE_ERROR = "compile_error"
DIAG_ASSERTION_FAILURE = "assertion_failure"
DIAG_TIMEOUT = "timeout"
DIAG_RUNTIME_ERROR = "runtime_error"
DIAG_CRASH = "crash"
DIAG_ALL_PASS = "all_pass"
DIAG_NO_TEST = "no_test"
DIAG_NO_CANDIDATE = "no_candidate"

# Difficulty constants
DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"
DIFFICULTY_UNKNOWN = "unknown"


@dataclass
class CodeExecutionResult:
    """Structured result of code execution judging."""

    correct: bool = False
    pass_count: int = 0
    total_count: int = 0
    pass_rate: float = 0.0
    diagnosis: str = DIAG_NO_CANDIDATE
    error_message: str = ""
    execution_time: float = 0.0
    entry_point: str = ""
    test_code: str = ""
    candidate_code: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Cache ────────────────────────────────────────────────────────────────────

_cache: dict[str, CodeExecutionResult] = {}
_cache_lock = threading.Lock()


def _cache_key(candidate_code: str, test_code: str, entry_point: str, timeout: int) -> str:
    payload = json.dumps(
        {
            "candidate": candidate_code,
            "test": test_code,
            "entry_point": entry_point,
            "timeout": timeout,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clear_code_execution_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ── Code extraction ──────────────────────────────────────────────────────────

_MARKDOWN_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL,
)


def extract_python_code(text: str, entry_point: str = "") -> str:
    """Extract Python code from a model response.

    Handles:
      - Markdown code fences (```python ... ```)
      - Raw function definitions
      - Code with explanations (extracts the code block)

    If entry_point is provided, validates that the extracted code
    contains a definition for that function.
    """
    if not text or not text.strip():
        return ""

    # Try markdown code fences first
    fences = _MARKDOWN_FENCE_RE.findall(text)
    if fences:
        # Use the last fence (models often put the final answer last)
        code = fences[-1].strip()
        if code:
            if not entry_point or _code_defines_function(code, entry_point):
                return code

    # Try to find a function definition directly
    lines = text.strip().split("\n")
    code_lines: list[str] = []
    in_function = False
    func_indent = 0

    for line in lines:
        stripped = line.strip()
        # Detect function/class definition start
        if re.match(r"^(def |class )", stripped):
            in_function = True
            func_indent = len(line) - len(line.lstrip())
            code_lines.append(line)
            continue
        if in_function:
            # End of function when we hit a non-empty, less-indented line
            if stripped and not line.startswith(" " * (func_indent + 1)) and not line.startswith("\t"):
                if not re.match(r"^(def |class |@)", stripped):
                    in_function = False
                    # Don't append this line - it's outside the function
                    continue
            code_lines.append(line)

    if code_lines:
        code = "\n".join(code_lines).strip()
        if code:
            if not entry_point or _code_defines_function(code, entry_point):
                return code

    # Fallback: return the entire text if it looks like Python code
    if "def " in text or "return " in text:
        return text.strip()

    return ""


def _code_defines_function(code: str, func_name: str) -> bool:
    """Check if the code defines a function with the given name."""
    pattern = re.compile(rf"^\s*def\s+{re.escape(func_name)}\s*\(", re.MULTILINE)
    return bool(pattern.search(code))


# ── Harness construction ────────────────────────────────────────────────────

def build_test_harness(candidate_code: str, test_code: str, entry_point: str) -> str:
    """Build an executable test harness.

    The harness:
      1. Defines the candidate function (from model output)
      2. Defines the test/check function (from dataset, NO gold answer)
      3. Calls check(entry_point) to run the tests

    Anti-contamination: the test_code must NOT contain the gold solution.
    The harness only includes the candidate code and the test code.
    """
    parts: list[str] = []

    # Candidate code (from model prediction)
    if candidate_code.strip():
        parts.append("# ── Candidate code ──")
        parts.append(candidate_code.strip())
        parts.append("")

    # Test code (from dataset - contains check function, no gold answer)
    if test_code.strip():
        parts.append("# ── Test code ──")
        parts.append(test_code.strip())
        parts.append("")

    # Runner
    if entry_point.strip():
        parts.append("# ── Runner ──")
        parts.append(f"check({entry_point.strip()})")
        parts.append("")

    return "\n".join(parts)


# ── Sandbox execution ───────────────────────────────────────────────────────

_DEFAULT_TIMEOUT = int(os.getenv("CODE_EXEC_TIMEOUT", "10"))
_DEFAULT_MEMORY_LIMIT_MB = int(os.getenv("CODE_EXEC_MEMORY_MB", "512"))


def _set_resource_limits(memory_mb: int) -> None:
    """Set resource limits for the child process (Linux only)."""
    try:
        # Memory limit
        mem_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass
    try:
        # CPU time limit (seconds) - slightly more than timeout
        cpu_limit = _DEFAULT_TIMEOUT + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except (ValueError, OSError):
        pass


def execute_code_sandbox(
    code: str,
    timeout: int = _DEFAULT_TIMEOUT,
    memory_mb: int = _DEFAULT_MEMORY_LIMIT_MB,
) -> tuple[int, str, str, float]:
    """Execute code in a sandboxed subprocess.

    Returns:
        (returncode, stdout, stderr, elapsed_seconds)
    """
    start = time.time()

    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="code_exec_", dir="/tmp"
    ) as f:
        f.write(code)
        f.flush()
        script_path = f.name

    try:
        # Build the execution command with resource limits
        # Use a wrapper script that sets limits before executing
        wrapper_code = f"""
import resource
import sys
import os

# Set memory limit
mem_bytes = {memory_mb} * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except (ValueError, OSError):
    pass

# Set CPU time limit
cpu_limit = {timeout} + 5
try:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
except (ValueError, OSError):
    pass

# Execute the target code
exec(open("{script_path}").read())
"""
        wrapper_path = script_path + "_wrapper.py"
        with open(wrapper_path, "w") as wf:
            wf.write(wrapper_code)

        result = subprocess.run(
            [sys.executable, wrapper_path],
            capture_output=True,
            text=True,
            timeout=timeout + 10,  # Extra buffer for process startup
            env={
                "PATH": os.getenv("PATH", "/usr/bin:/bin"),
                "HOME": "/tmp",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": "",
            },
        )
        elapsed = time.time() - start
        return result.returncode, result.stdout, result.stderr, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return -1, "", "TIMEOUT", elapsed
    except Exception as exc:
        elapsed = time.time() - start
        return -2, "", str(exc), elapsed
    finally:
        # Clean up temp files
        for p in (script_path, script_path + "_wrapper.py"):
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Diagnosis ───────────────────────────────────────────────────────────────

def _diagnose_failure(returncode: int, stderr: str, stdout: str) -> str:
    """Diagnose the type of failure from execution output."""
    stderr_lower = (stderr or "").lower()

    if "TIMEOUT" in stderr:
        return DIAG_TIMEOUT

    if returncode == -1:
        return DIAG_TIMEOUT

    if returncode < 0:
        # Killed by signal
        return DIAG_CRASH

    # Check for compile/syntax errors
    if "syntaxerror" in stderr_lower or "indentationerror" in stderr_lower:
        return DIAG_COMPILE_ERROR

    if "importerror" in stderr_lower or "modulenotfounderror" in stderr_lower:
        return DIAG_COMPILE_ERROR

    # Check for assertion failures
    if "assertionerror" in stderr_lower or "assert " in stderr_lower:
        return DIAG_ASSERTION_FAILURE

    # Check for NameError (function not defined)
    if "nameerror" in stderr_lower:
        return DIAG_RUNTIME_ERROR

    # Check for TypeError
    if "typeerror" in stderr_lower:
        return DIAG_RUNTIME_ERROR

    # Check for other runtime errors
    if "error" in stderr_lower or "exception" in stderr_lower:
        return DIAG_RUNTIME_ERROR

    # Non-zero exit without error message
    if returncode != 0:
        return DIAG_CRASH

    return DIAG_RUNTIME_ERROR


# ── Core judge function ─────────────────────────────────────────────────────

def judge_code_answer(
    candidate_code: str,
    test_code: str,
    entry_point: str,
    timeout: int = _DEFAULT_TIMEOUT,
    use_cache: bool = True,
) -> CodeExecutionResult:
    """Judge a code answer by executing it against test cases.

    This is the main entry point for code execution judging.

    Args:
        candidate_code: The Python code generated by the model
        test_code: The test function code (HumanEval-style check)
        entry_point: The function name to test
        timeout: Execution timeout in seconds
        use_cache: Whether to use the result cache

    Returns:
        CodeExecutionResult with structured diagnosis
    """
    # Check cache
    if use_cache:
        key = _cache_key(candidate_code, test_code, entry_point, timeout)
        with _cache_lock:
            cached = _cache.get(key)
        if cached is not None:
            result = CodeExecutionResult(**asdict(cached))
            result.cached = True
            return result

    # Validate inputs
    if not candidate_code or not candidate_code.strip():
        return CodeExecutionResult(
            diagnosis=DIAG_NO_CANDIDATE,
            error_message="No candidate code provided",
        )

    if not test_code or not test_code.strip():
        return CodeExecutionResult(
            diagnosis=DIAG_NO_TEST,
            error_message="No test code provided",
            candidate_code=candidate_code,
        )

    if not entry_point or not entry_point.strip():
        return CodeExecutionResult(
            diagnosis=DIAG_NO_TEST,
            error_message="No entry_point provided",
            candidate_code=candidate_code,
        )

    # Build harness
    harness = build_test_harness(candidate_code, test_code, entry_point)

    # Execute
    returncode, stdout, stderr, elapsed = execute_code_sandbox(
        harness, timeout=timeout
    )

    # Diagnose
    if returncode == 0:
        diagnosis = DIAG_ALL_PASS
        correct = True
        pass_count = 1
        total_count = 1
    else:
        diagnosis = _diagnose_failure(returncode, stderr, stdout)
        correct = False
        pass_count = 0
        total_count = 1

    # Extract error message (last few lines of stderr)
    error_lines = (stderr or "").strip().split("\n")
    error_message = "\n".join(error_lines[-5:]) if error_lines else ""

    result = CodeExecutionResult(
        correct=correct,
        pass_count=pass_count,
        total_count=total_count,
        pass_rate=pass_count / total_count if total_count else 0.0,
        diagnosis=diagnosis,
        error_message=error_message,
        execution_time=elapsed,
        entry_point=entry_point,
        test_code=test_code,
        candidate_code=candidate_code,
    )

    # Cache result
    if use_cache:
        key = _cache_key(candidate_code, test_code, entry_point, timeout)
        with _cache_lock:
            _cache[key] = result

    return result


def judge_code_from_prediction(
    prediction: str,
    test_code: str,
    entry_point: str,
    timeout: int = _DEFAULT_TIMEOUT,
    use_cache: bool = True,
) -> CodeExecutionResult:
    """Judge a model prediction by extracting code and executing tests.

    This is the high-level entry point that:
      1. Extracts Python code from the model's prediction
      2. Judges it against the test cases

    Args:
        prediction: The raw model output (may contain markdown, explanations)
        test_code: The test function code
        entry_point: The function name to test
        timeout: Execution timeout
        use_cache: Whether to use caching

    Returns:
        CodeExecutionResult
    """
    candidate_code = extract_python_code(prediction, entry_point)
    return judge_code_answer(
        candidate_code=candidate_code,
        test_code=test_code,
        entry_point=entry_point,
        timeout=timeout,
        use_cache=use_cache,
    )


# ── Difficulty estimation ───────────────────────────────────────────────────

def estimate_code_difficulty(
    pass_count: int,
    total_count: int,
    has_tests: bool = True,
) -> str:
    """Estimate difficulty from test execution results.

    Rules:
      - All tests pass → easy
      - Some tests pass → medium
      - No tests pass → hard
      - No tests available → unknown
    """
    if not has_tests or total_count <= 0:
        return DIFFICULTY_UNKNOWN

    if pass_count >= total_count:
        return DIFFICULTY_EASY
    if pass_count > 0:
        return DIFFICULTY_MEDIUM
    return DIFFICULTY_HARD


# ── Multi-candidate aggregation ─────────────────────────────────────────────

@dataclass
class MultiCandidateResult:
    """Aggregated result for multiple candidates on the same question."""

    question_id: str = ""
    total_candidates: int = 0
    pass_count: int = 0
    pass_rate: float = 0.0
    all_pass: bool = False
    any_pass: bool = False
    difficulty: str = DIFFICULTY_UNKNOWN
    per_candidate_results: list[CodeExecutionResult] = field(default_factory=list)


def judge_multiple_candidates(
    question_id: str,
    candidates: list[str],
    test_code: str,
    entry_point: str,
    timeout: int = _DEFAULT_TIMEOUT,
    max_workers: int = 4,
    use_cache: bool = True,
) -> MultiCandidateResult:
    """Judge multiple candidates for the same question and aggregate results.

    Aggregation rule: "全部通过才算对" (all must pass for correct).

    Args:
        question_id: Question identifier
        candidates: List of model predictions (raw text)
        test_code: The test function code
        entry_point: The function name
        timeout: Execution timeout per candidate
        max_workers: Thread pool size
        use_cache: Whether to use caching

    Returns:
        MultiCandidateResult with aggregated stats
    """
    if not candidates:
        return MultiCandidateResult(
            question_id=question_id,
            difficulty=DIFFICULTY_UNKNOWN,
        )

    has_tests = bool(test_code and test_code.strip() and entry_point and entry_point.strip())

    # Judge candidates in parallel
    results: list[CodeExecutionResult] = [None] * len(candidates)  # type: ignore[list-item]

    def _judge_one(idx: int, prediction: str) -> tuple[int, CodeExecutionResult]:
        result = judge_code_from_prediction(
            prediction=prediction,
            test_code=test_code,
            entry_point=entry_point,
            timeout=timeout,
            use_cache=use_cache,
        )
        return idx, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_judge_one, idx, pred): idx
            for idx, pred in enumerate(candidates)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    # Aggregate
    pass_count = sum(1 for r in results if r and r.correct)
    total = len(results)
    all_pass = pass_count == total and total > 0
    any_pass = pass_count > 0

    difficulty = estimate_code_difficulty(pass_count, total, has_tests=has_tests)

    return MultiCandidateResult(
        question_id=question_id,
        total_candidates=total,
        pass_count=pass_count,
        pass_rate=pass_count / total if total else 0.0,
        all_pass=all_pass,
        any_pass=any_pass,
        difficulty=difficulty,
        per_candidate_results=results,
    )


# ── Batch judging ───────────────────────────────────────────────────────────

def judge_code_batch(
    predictions: list[str],
    test_codes: list[str],
    entry_points: list[str],
    timeout: int = _DEFAULT_TIMEOUT,
    max_workers: int = 4,
    use_cache: bool = True,
) -> list[CodeExecutionResult]:
    """Judge a batch of code predictions in parallel.

    Args:
        predictions: List of model predictions
        test_codes: List of test codes (one per prediction)
        entry_points: List of entry points (one per prediction)
        timeout: Execution timeout per item
        max_workers: Thread pool size
        use_cache: Whether to use caching

    Returns:
        List of CodeExecutionResult, one per prediction
    """
    if not predictions:
        return []

    results: list[CodeExecutionResult] = [None] * len(predictions)  # type: ignore[list-item]

    def _judge_one(idx: int) -> tuple[int, CodeExecutionResult]:
        result = judge_code_from_prediction(
            prediction=predictions[idx],
            test_code=test_codes[idx] if idx < len(test_codes) else "",
            entry_point=entry_points[idx] if idx < len(entry_points) else "",
            timeout=timeout,
            use_cache=use_cache,
        )
        return idx, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_judge_one, idx): idx for idx in range(len(predictions))}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


# ── Compatibility shim for evaluator integration ────────────────────────────

def is_code_domain() -> bool:
    """Check if the current run is in code domain."""
    return os.getenv("DOMAIN", "").strip().lower() == "code"


def get_code_judge_func():
    """Return the appropriate judge function based on DOMAIN setting.

    When DOMAIN=code, returns a function that uses code execution.
    Otherwise, returns the standard judge_answer function.
    """
    if is_code_domain():
        def _code_judge(prediction: str, gold_answer: str, item: dict | None = None) -> bool:
            """Code execution judge - uses test execution, not text matching."""
            if item is None:
                return False
            test_code = str(item.get("test", "") or "")
            entry_point = str(item.get("entry_point", "") or "")
            if not test_code or not entry_point:
                return False
            result = judge_code_from_prediction(
                prediction=prediction,
                test_code=test_code,
                entry_point=entry_point,
            )
            return result.correct
        return _code_judge
    else:
        from src.tools.model_runner import judge_answer
        return judge_answer


# ── CLI for testing ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick self-test
    test_prediction = """
Here's the solution:
```python
def add(a, b):
    return a + b
```
"""
    test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0
"""
    entry_point = "add"

    result = judge_code_from_prediction(test_prediction, test_code, entry_point)
    print(f"Correct: {result.correct}")
    print(f"Diagnosis: {result.diagnosis}")
    print(f"Pass rate: {result.pass_rate}")
    print(f"Time: {result.execution_time:.3f}s")

    # Test difficulty estimation
    print(f"\nDifficulty (all pass): {estimate_code_difficulty(3, 3)}")
    print(f"Difficulty (partial): {estimate_code_difficulty(1, 3)}")
    print(f"Difficulty (none pass): {estimate_code_difficulty(0, 3)}")
    print(f"Difficulty (no tests): {estimate_code_difficulty(0, 0, has_tests=False)}")
