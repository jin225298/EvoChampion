"""Local (CPU) tests for the code-domain judge. No GPU / training involved."""
import pytest

from src.tools import code_execution as ce


def _item(solution_test: str, entry_point: str | None = "add"):
    return {"question": "add two numbers", "test": solution_test, "entry_point": entry_point}


ITEM = _item(
    "def check(candidate):\n"
    "    assert candidate(1, 2) == 3\n"
    "    assert candidate(-1, 1) == 0\n"
    "    assert candidate(0, 0) == 0\n",
    entry_point="add",
)


def test_correct_solution_passes():
    sol = "def add(a, b):\n    return a + b"
    r = ce.judge_single(sol, ITEM["test"], entry_point="add")
    assert r.correct is True
    assert r.status == ce.KIND_PASS
    assert r.total == 3


def test_wrong_solution_fails_assertion():
    sol = "def add(a, b):\n    return a - b"
    r = ce.judge_single(sol, ITEM["test"], entry_point="add")
    assert r.correct is False
    assert r.failure_kind == ce.KIND_FAIL


def test_syntax_error_classified_compile_error():
    sol = "def add(a, b):\n  return a +"
    r = ce.judge_single(sol, ITEM["test"], entry_point="add")
    assert r.correct is False
    assert r.failure_kind == ce.KIND_COMPILE_ERROR


def test_timeout_killed():
    sol = "def add(a, b):\n    while True:\n        pass"
    r = ce.judge_single(sol, ITEM["test"], entry_point="add", timeout=1.0)
    assert r.correct is False
    assert r.failure_kind == ce.KIND_TIMEOUT


def test_no_test_unknown():
    r = ce.judge_single("def add(a, b):\n    return a + b", "", entry_point="add")
    assert r.correct is False
    assert r.failure_kind == ce.KIND_NO_TESTS


def test_no_code_extracted_unknown():
    r = ce.judge_single("This is just prose, no code at all.", ITEM["test"], entry_point="add")
    assert r.correct is False
    assert r.failure_kind == ce.KIND_UNKNOWN


def test_fenced_code_extraction():
    raw = "Here is my solution:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    assert ce.extract_code(raw).strip() == "def add(a, b):\n    return a + b"
    r = ce.judge_single(raw, ITEM["test"], entry_point="add")
    assert r.correct is True


def test_think_tag_stripped():
    raw = "<think>I should just return the sum.</think>\ndef add(a, b):\n    return a + b"
    r = ce.judge_single(raw, ITEM["test"], entry_point="add")
    assert r.correct is True


def test_runtime_error():
    sol = "def add(a, b):\n    return a[0] + b"  # TypeError: 'int' not subscriptable
    r = ce.judge_single(sol, ITEM["test"], entry_point="add")
    assert r.correct is False
    assert r.failure_kind == ce.KIND_RUNTIME_ERROR


def test_forbidden_import_blocked():
    sol = "import os\n\ndef add(a, b):\n    return a + b"
    r = ce.judge_single(sol, ITEM["test"], entry_point="add")
    assert r.correct is False
    assert r.failure_kind == ce.KIND_FORBIDDEN_IMPORT


def test_multi_candidate_aggregation_and_difficulty():
    good = "def add(a, b):\n    return a + b"
    bad = "def add(a, b):\n    return a - b"
    agg = ce.judge_multi([good, bad], ITEM["test"], entry_point="add")
    assert agg.correct is False          # not all candidates pass
    assert agg.n_passed == 1
    assert agg.n_failed == 1
    assert agg.pass_rate == 0.5
    assert agg.difficulty == "medium"

    agg_all = ce.judge_multi([good, good], ITEM["test"], entry_point="add")
    assert agg_all.correct is True
    assert agg_all.difficulty == "easy"


def test_is_code_item():
    assert ce.is_code_item(ITEM) is True
    assert ce.is_code_item({"question": "no test here"}) is False
    assert ce.is_code_item({"question": "x", "test": "assert 1 == 1", "entry_point": "f"}) is True


def test_humaneval_style_with_check():
    humaneval_item = {
        "question": "Return the sum of two ints.",
        "test": "def check(candidate):\n"
                "    assert candidate(1, 2) == 3\n"
                "    assert candidate(0, 0) == 0\n",
        "entry_point": "add",
    }
    r = ce.judge_single("def add(a, b):\n    return a + b", humaneval_item["test"],
                        entry_point="add")
    assert r.correct is True


def test_standalone_assert_style():
    item = {"question": "write add", "test": "assert add(2, 3) == 5", "entry_point": "add"}
    r = ce.judge_single("def add(a, b):\n    return a + b", item["test"], entry_point="add")
    assert r.correct is True


# ---------------------------------------------------------------------------
# Dispatch path: the functions evaluator / bootstrap / difficulty_tagger call.
# ---------------------------------------------------------------------------
def test_judge_solution_code_item_executes_tests():
    item = {"test": "def check(candidate):\n    assert candidate(2, 3) == 5\n", "entry_point": "add"}
    assert ce.judge_solution("def add(a, b):\n    return a + b", "ignored gold", item) is True
    assert ce.judge_solution("def add(a, b):\n    return a - b", "ignored gold", item) is False


def test_judge_solution_no_test_not_a_pass():
    # Code domain + no executable test must never count as correct.
    item = {"question": "write add", "answer": "def add(a,b): return a+b"}
    assert ce.judge_solution("def add(a, b):\n    return a + b", "gold", item) is False


def test_judge_chunk_solutions_aggregation():
    item = {"test": "def check(candidate):\n    assert candidate(1, 2) == 3\n", "entry_point": "add"}
    good = "def add(a, b):\n    return a + b"
    bad = "def add(a, b):\n    return a - b"
    verdicts = ce.judge_chunk_solutions([good, bad], "gold", item)
    assert verdicts == [True, False]


def test_code_judge_answer_ignores_gold_string():
    # gold_answer must not influence the verdict; only the executed test does.
    item = {"test": "def check(candidate):\n    assert candidate(1, 2) == 3\n", "entry_point": "add"}
    r = ce.code_judge_answer("def add(a, b):\n    return a + b", gold_answer="WRONG GOLD", item=item)
    assert r.correct is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
