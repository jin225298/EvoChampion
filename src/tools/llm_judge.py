from __future__ import annotations

from typing import Any, Callable

from config.settings import LLM_JUDGE_BATCH_SIZE, LLM_JUDGE_MAX_NEW_TOKENS, USE_LLM_AS_JUDGE
from src.tools.agent_prompts import EVALUATOR_JUDGE_PROMPT
from src.tools.llm_decision import decide_json_with_audit, decide_json_with_audit_batch, prompt_for_agent
from src.tools.model_runner import judge_answer


LLM_JUDGE_CORRECT_THRESHOLD = 1.0


def _clamped_score(value: Any, fallback: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return float(fallback)


def _binary_score(value: Any, fallback: float = 0.0) -> float:
    score = _clamped_score(value, fallback)
    return 1.0 if score >= 1.0 else 0.0


def llm_judge_decision(
    prediction: str,
    gold_answer: str = "",
    question_text: str = "",
    reference_solution: str = "",
    state: dict[str, Any] | None = None,
    *,
    use_llm: bool | None = None,
    judge_answer_func: Callable[[str, str], bool] | None = None,
    decide_json_func: Callable[..., dict[str, Any]] | None = None,
    prompt_for_agent_func: Callable[[dict[str, Any], str, str], str] | None = None,
) -> dict[str, Any]:
    """Return structured LLM-as-judge scores for a prediction.

    This mirrors the evaluator's existing rubric while allowing judge-only
    datasets to provide a reference solution instead of a concise gold answer.
    """

    hard_judge = judge_answer_func or judge_answer
    hard_correct = bool(gold_answer and hard_judge(prediction, gold_answer))
    fallback_score = 1.0 if hard_correct else 0.0
    fallback = {
        "result_score": fallback_score,
        "step_score": fallback_score,
        "total_score": fallback_score,
        "reason": "symbolic fallback judge",
        "symbolic_answer_match": hard_correct,
        "source": "symbolic_fallback",
        "judge_raw_text": "",
        "fallback_used": True,
        "schema_errors": [],
    }
    if not (USE_LLM_AS_JUDGE if use_llm is None else bool(use_llm)):
        return dict(fallback)

    prompt_state = dict(state) if isinstance(state, dict) else {}
    prompt = (prompt_for_agent_func or prompt_for_agent)(
        prompt_state,
        "evaluator_judge",
        EVALUATOR_JUDGE_PROMPT,
    )
    context = {
        "question": question_text,
        "gold_solution": gold_answer,
        "reference_solution": reference_solution,
        "student_solution": prediction,
        "symbolic_answer_match": hard_correct,
        "rubric": {
            "binary": True,
            "correct_value": 1.0,
            "incorrect_value": 0.0,
            "judge_only": "Compare only final answer equivalence between reference/gold and student.",
        },
    }
    audit: dict[str, Any] = {}
    if decide_json_func is not None:
        decision = decide_json_func(
            agent_name="evaluator_judge",
            prompt=prompt,
            context=context,
            fallback=fallback,
            trace_id=prompt_state.get("trace_id", ""),
            round_id=prompt_state.get("round_id", 0),
            max_new_tokens=LLM_JUDGE_MAX_NEW_TOKENS,
            disable_thinking=False,
            stop_after_json=False,
        )
    else:
        decision, audit = decide_json_with_audit(
            agent_name="evaluator_judge",
            prompt=prompt,
            context=context,
            fallback=fallback,
            trace_id=prompt_state.get("trace_id", ""),
            round_id=prompt_state.get("round_id", 0),
            max_new_tokens=LLM_JUDGE_MAX_NEW_TOKENS,
            disable_thinking=False,
            stop_after_json=False,
        )
    result_score = _binary_score(decision.get("result_score"), fallback["result_score"])
    total_score = result_score
    return {
        "result_score": result_score,
        "step_score": result_score,
        "total_score": total_score,
        "reason": str(decision.get("reason") or fallback["reason"]),
        "symbolic_answer_match": hard_correct,
        "source": "llm_judge",
        "judge_raw_text": str(audit.get("raw_text") or ""),
        "fallback_used": bool(audit.get("fallback_used", False)),
        "schema_errors": list(audit.get("schema_errors") or []),
    }


def _judge_fallback(
    *,
    prediction: str,
    gold_answer: str,
    judge_answer_func: Callable[[str, str], bool] | None = None,
) -> dict[str, Any]:
    hard_judge = judge_answer_func or judge_answer
    hard_correct = bool(gold_answer and hard_judge(prediction, gold_answer))
    fallback_score = 1.0 if hard_correct else 0.0
    return {
        "result_score": fallback_score,
        "step_score": fallback_score,
        "total_score": fallback_score,
        "reason": "symbolic fallback judge",
        "symbolic_answer_match": hard_correct,
        "source": "symbolic_fallback",
        "judge_raw_text": "",
        "fallback_used": True,
        "schema_errors": [],
    }


def _judge_context(
    *,
    prediction: str,
    gold_answer: str,
    question_text: str,
    reference_solution: str,
    symbolic_answer_match: bool,
) -> dict[str, Any]:
    return {
        "question": question_text,
        "gold_solution": gold_answer,
        "reference_solution": reference_solution,
        "student_solution": prediction,
        "symbolic_answer_match": symbolic_answer_match,
        "rubric": {
            "binary": True,
            "correct_value": 1.0,
            "incorrect_value": 0.0,
            "judge_only": "Compare only final answer equivalence between reference/gold and student.",
        },
    }


def _decision_to_judgement(
    decision: dict[str, Any],
    audit: dict[str, Any],
    *,
    symbolic_answer_match: bool,
) -> dict[str, Any]:
    result_score = _binary_score(decision.get("result_score"), 1.0 if symbolic_answer_match else 0.0)
    score = result_score
    return {
        "score": score,
        "correct": is_correct_by_score(score),
        "reason": str(decision.get("reason") or ""),
        "result_score": score,
        "step_score": score,
        "symbolic_answer_match": symbolic_answer_match,
        "source": "llm_judge",
        "judge_raw_text": str(audit.get("raw_text") or ""),
        "fallback_used": bool(audit.get("fallback_used", False)),
        "schema_errors": list(audit.get("schema_errors") or []),
    }


def judge_predictions_with_llm_batch(
    *,
    predictions: list[str],
    question_texts: list[str],
    gold_answers: list[str] | None = None,
    reference_solutions: list[str] | None = None,
    state: dict[str, Any] | None = None,
    use_llm: bool | None = None,
    judge_answer_func: Callable[[str, str], bool] | None = None,
    prompt_for_agent_func: Callable[[dict[str, Any], str, str], str] | None = None,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    if not predictions:
        return []

    prompt_state = dict(state) if isinstance(state, dict) else {}
    use_llm_enabled = USE_LLM_AS_JUDGE if use_llm is None else bool(use_llm)
    hard_judge = judge_answer_func or judge_answer
    golds = list(gold_answers or [])
    refs = list(reference_solutions or [])
    questions = list(question_texts or [])
    judgements = [
        {
            "score": 0.0,
            "correct": False,
            "reason": "not judged",
            "result_score": 0.0,
            "step_score": 0.0,
            "symbolic_answer_match": False,
            "source": "not_judged",
            "judge_raw_text": "",
            "fallback_used": True,
            "schema_errors": [],
        }
        for _ in predictions
    ]
    requests: list[dict[str, Any]] = []
    request_indices: list[int] = []
    for idx, prediction in enumerate(predictions):
        gold = golds[idx] if idx < len(golds) else ""
        reference_solution = refs[idx] if idx < len(refs) else ""
        question_text = questions[idx] if idx < len(questions) else ""
        fallback = _judge_fallback(
            prediction=prediction,
            gold_answer=gold,
            judge_answer_func=hard_judge,
        )
        if not use_llm_enabled:
            score = float(fallback.get("total_score", 0.0))
            judgements[idx] = {
                "score": score,
                "correct": is_correct_by_score(score),
                "reason": str(fallback.get("reason") or ""),
                "result_score": score,
                "step_score": score,
                "symbolic_answer_match": bool(fallback.get("symbolic_answer_match", False)),
                "source": str(fallback.get("source") or "symbolic_fallback"),
                "judge_raw_text": "",
                "fallback_used": True,
                "schema_errors": [],
            }
            continue
        requests.append(
            {
                "context": _judge_context(
                    prediction=prediction,
                    gold_answer=gold,
                    question_text=question_text,
                    reference_solution=reference_solution,
                    symbolic_answer_match=bool(fallback.get("symbolic_answer_match", False)),
                ),
                "fallback": fallback,
                "trace_id": prompt_state.get("trace_id", ""),
                "round_id": prompt_state.get("round_id", 0),
            }
        )
        request_indices.append(idx)

    if requests:
        prompt = (prompt_for_agent_func or prompt_for_agent)(
            prompt_state,
            "evaluator_judge",
            EVALUATOR_JUDGE_PROMPT,
        )
        chunk_size = max(1, int(batch_size or LLM_JUDGE_BATCH_SIZE or len(requests)))
        cursor = 0
        while cursor < len(requests):
            chunk = requests[cursor:cursor + chunk_size]
            decisions = decide_json_with_audit_batch(
                agent_name="evaluator_judge",
                prompt=prompt,
                requests=chunk,
                max_new_tokens=LLM_JUDGE_MAX_NEW_TOKENS,
                disable_thinking=False,
                stop_after_json=False,
            )
            for offset, packed in enumerate(decisions):
                original_idx = request_indices[cursor + offset]
                fallback = chunk[offset]["fallback"]
                judgements[original_idx] = _decision_to_judgement(
                    packed.get("result") or fallback,
                    packed.get("audit") or {},
                    symbolic_answer_match=bool(fallback.get("symbolic_answer_match", False)),
                )
            cursor += chunk_size

    return judgements


def llm_judge_score(
    prediction: str,
    gold_answer: str = "",
    question_text: str = "",
    reference_solution: str = "",
    state: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    decision = llm_judge_decision(
        prediction,
        gold_answer,
        question_text,
        reference_solution,
        state,
        **kwargs,
    )
    return float(decision.get("total_score", 0.0))


def is_correct_by_score(score: float, threshold: float = LLM_JUDGE_CORRECT_THRESHOLD) -> bool:
    return float(score) >= float(threshold)


def judge_prediction_with_llm(
    *,
    prediction: str,
    question_text: str,
    gold_answer: str = "",
    reference_solution: str = "",
    state: dict[str, Any] | None = None,
    threshold: float = LLM_JUDGE_CORRECT_THRESHOLD,
    **kwargs: Any,
) -> dict[str, Any]:
    decision = llm_judge_decision(
        prediction=prediction,
        gold_answer=gold_answer,
        question_text=question_text,
        reference_solution=reference_solution,
        state=state,
        **kwargs,
    )
    score = float(decision.get("total_score", 0.0))
    return {
        "score": score,
        "correct": is_correct_by_score(score, threshold=threshold),
        "reason": str(decision.get("reason") or ""),
        "result_score": float(decision.get("result_score", score)),
        "step_score": float(decision.get("step_score", score)),
        "symbolic_answer_match": bool(decision.get("symbolic_answer_match", False)),
        "source": str(decision.get("source") or "llm_judge"),
        "judge_raw_text": str(decision.get("judge_raw_text") or ""),
        "fallback_used": bool(decision.get("fallback_used", False)),
        "schema_errors": list(decision.get("schema_errors") or []),
    }
