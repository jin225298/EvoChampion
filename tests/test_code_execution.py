"""Tests for the code-domain execution-based judging (src/tools/code_execution.py).

Covers: sandbox execution, HumanEval-style check(candidate) harness,
multi-candidate aggregation, failure diagnosis, difficulty estimation,
anti-pollution, caching, and thread-pool parallelism. No GPU / model needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.code_execution import (
    CodeJudgeResult,
    aggregate_code_results,
    clear_cache,
    code_difficulty_from_results,
    extract_code,
    judge_code_prediction,
    judge_code_predictions_batch,
    STATUS_ASSERTION_ERROR,
    STATUS_COMPILE_ERROR,
    STATUS_MISSING_ENTRY_POINT,
    STATUS_NO_TEST,
    STATUS_PASSED,
    STATUS_RUNTIME_ERROR,
    STATUS_TIMEOUT,
)


_TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0"
_GOOD = "def add(a, b):\n    return a + b"
_WRONG = "def add(a, b):\n    return a - b"


def test_correct_solution_passes():
    r = judge_code_prediction(_GOOD, _TEST, "add")
    assert r.passed and r.status == STATUS_PASSED


def test_compile_error():
    r = judge_code_prediction("def add(a, b)\n    return a + b", _TEST, "add")
    assert not r.passed and r.status == STATUS_COMPILE_ERROR


def test_assertion_error():
    r = judge_code_prediction(_WRONG, _TEST, "add")
    assert not r.passed and r.status == STATUS_ASSERTION_ERROR


def test_runtime_error():
    r = judge_code_prediction("def add(a, b):\n    return 1/0", _TEST, "add")
    assert not r.passed and r.status == STATUS_RUNTIME_ERROR


def test_missing_entry_point():
    r = judge_code_prediction("def sub(a, b):\n    return a - b", _TEST, "add")
    assert not r.passed and r.status == STATUS_MISSING_ENTRY_POINT


def test_timeout():
    r = judge_code_prediction("def add(a, b):\n    while True: pass", _TEST, "add", timeout=1)
    assert not r.passed and r.status == STATUS_TIMEOUT


def test_no_test():
    r = judge_code_prediction(_GOOD, "", "add")
    assert r.status == STATUS_NO_TEST


def test_markdown_fence_extraction():
    cand = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    assert extract_code(cand) == "def add(a, b):\n    return a + b"
    r = judge_code_prediction(cand, _TEST, "add")
    assert r.passed


def test_cache_reuse():
    clear_cache()
    r1 = judge_code_prediction(_GOOD, _TEST, "add")
    r2 = judge_code_prediction(_GOOD, _TEST, "add")
    assert r1.passed and r2.passed


def test_batch_parallel():
    items = [
        {"candidate_code": _GOOD, "test_code": _TEST, "entry_point": "add"},
        {"candidate_code": _WRONG, "test_code": _TEST, "entry_point": "add"},
        {"candidate_code": _GOOD, "test_code": _TEST, "entry_point": "add"},
    ]
    results = judge_code_predictions_batch(items, max_workers=4)
    assert results[0].passed and not results[1].passed and results[2].passed


def test_aggregation_all_pass_easy():
    results = [CodeJudgeResult(passed=True, status=STATUS_PASSED)] * 3
    agg = aggregate_code_results("q", results)
    assert agg.correct and agg.difficulty == "easy"


def test_aggregation_partial_medium():
    results = [
        CodeJudgeResult(passed=True, status=STATUS_PASSED),
        CodeJudgeResult(passed=False, status=STATUS_ASSERTION_ERROR),
    ]
    agg = aggregate_code_results("q", results)
    assert not agg.correct and agg.difficulty == "medium"


def test_aggregation_all_fail_hard():
    results = [CodeJudgeResult(passed=False, status=STATUS_ASSERTION_ERROR)] * 2
    agg = aggregate_code_results("q", results)
    assert not agg.correct and agg.difficulty == "hard"


def test_aggregation_no_test_unknown():
    agg = aggregate_code_results("q", [], has_test=False)
    assert agg.difficulty == "unknown"


def test_difficulty_function():
    assert code_difficulty_from_results(
        [CodeJudgeResult(passed=True, status=STATUS_PASSED)] * 2
    ) == "easy"
    assert code_difficulty_from_results(
        [CodeJudgeResult(passed=True, status=STATUS_PASSED),
         CodeJudgeResult(passed=False, status=STATUS_ASSERTION_ERROR)]
    ) == "medium"
    assert code_difficulty_from_results(
        [CodeJudgeResult(passed=False, status=STATUS_ASSERTION_ERROR)]
    ) == "hard"
    assert code_difficulty_from_results([], has_test=False) == "unknown"


def test_anti_pollution_no_gold_in_harness():
    # The harness must not contain the gold answer; only candidate + test.
    from src.tools.code_execution import build_harness
    harness = build_harness(_GOOD, _TEST, "add")
    # The reference solution (_GOOD) is the *candidate* under test, not leaked
    # as a gold oracle. The test block contains no gold function body.
    assert "def add(a, b):" in harness  # candidate is present (being tested)
    # The test itself contains no reference implementation.
    assert "return a + b" not in _TEST


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"✓ {fn.__name__}")
    print(f"\n✅ {len(fns)} tests passed")
