import json
from pathlib import Path
from typing import Any

from config.settings import (
    AGENT_BASE_MODEL_NAME,
    CONTEXT_COMPACTION_ENABLED,
    CONTEXT_COMPACTION_TRIGGER_CHARS,
    LLM_AGENT_LEAF_MAX_NEW_TOKENS,
    LLM_AGENT_MAX_NEW_TOKENS,
    USE_LLM_AGENTS,
    get_session_dir,
)
from src.tools.context_compactor import compact_context
from src.tools.model_runner import run_model_batch


_PLACEHOLDER_VALUES = {
    "action_key",
    "action_type",
    "branch_parent_node_id",
    "selected_action",
    "template",
    "string",
    "reason",
}


_CRITICAL_PROMPT_AGENTS = {
    "parameter_master",
    "inspection_agent.decision",
    "inspection_agent.confidence",
    "inspection_agent.replay",
    "evaluator_judge",
}


_REQUIRED_FIELDS_BY_AGENT = {
    "evaluator_judge": {"result_score", "step_score", "total_score", "reason"},
    "parameter_master": {"action_key", "replay_sample_ratio", "dataset_selection_mode", "reason"},
    "parameter_master.mutation": {"lr_scale", "replay_scale", "reason"},
    "parameter_master.retry_same_data": {"retry_same_data"},
    "parameter_master.training_hyperparams": {"reason"},
    "teaching_teacher.search_query": {"search_query"},
    "teaching_teacher.target_difficulty": {"target_difficulty"},
    "teaching_teacher.difficulty_weights": {"difficulty_weights"},
    "teaching_teacher.dataset_policy_hint": {"dataset_policy_hint"},
    "replay_teacher.replay_sample_ratio": {"replay_sample_ratio"},
    "dataset_reviewer": {"verdict", "reason", "suitability_score"},
}


def _required_fields_for_agent(agent_name: str) -> set[str]:
    return set(_REQUIRED_FIELDS_BY_AGENT.get(agent_name, set()))


def _estimate_token_count(text: str) -> int:
    """Cheap diagnostic token estimate used only for logging.

    Model-specific tokenization is intentionally avoided here so decision
    logging remains lightweight and cannot fail the decision path.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, int((ascii_chars / 4.0) + (non_ascii_chars / 1.8)))


def _prompt_budget_for(max_new_tokens: int) -> int:
    try:
        from src.tools.model_runner import _get_safe_prompt_token_budget

        return int(_get_safe_prompt_token_budget(max_new_tokens=max_new_tokens))
    except Exception:
        return max(64, 4096 - int(max_new_tokens) - 16)


def _extract_json(text: str) -> dict[str, Any]:
    """从模型输出文本中提取第一个完整的 JSON 对象。

    Small local models often continue with prose after producing complete JSON,
    或者在 JSON 前包含闭合的 <think> 块。旧的「第一个 { 到最后一个 }」
    切片方法在后续文本包含花括号或尾部被截断时会失败。
    本扫描器改为在第一个平衡匹配的 JSON 对象处停止，更鲁棒。
    """
    if not text:
        return {}

    candidates = [text]
    if "</think>" in text:
        candidates.insert(0, text.rsplit("</think>", 1)[-1])

    for candidate in candidates:
        for start, end in _json_object_spans(candidate):
            try:
                loaded = json.loads(candidate[start:end])
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                return loaded
    return {}


def _json_object_spans(text: str):
    """扫描文本中所有平衡匹配的 JSON 对象起止位置。

    逐字符遍历，跟踪字符串状态（引号内/外、转义）和花括号嵌套深度，
    每当深度归零时 yield 一个 (start, end) 区间。
    """
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = idx
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield start, idx + 1
                start = None
        elif depth < 0:
            start = None
            depth = 0


def _persist_decision(
    trace_id: str,
    round_id: int,
    agent_name: str,
    context: dict[str, Any],
    result: dict[str, Any],
    raw_text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """将 Agent 决策结果持久化到 session 目录的 JSONL 日志中。

    每个决策记录包含：轮次、Agent 名称、输入上下文键列表、最终结果和原始输出文本，
    便于后续回放和调试。
    """
    if not trace_id:
        return
    try:
        path = get_session_dir(trace_id) / "llm_agent_decisions.jsonl"
        entry = {
            "round": round_id,
            "agent": agent_name,
            "context_keys": sorted(context.keys()),
            "result": result,
            "raw_text": raw_text[:2000],
        }
        if metadata:
            entry.update(metadata)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _build_request(
    prompt: str,
    context: dict[str, Any],
    trace_id: str = "",
    round_id: int = 0,
    agent_name: str = "",
) -> str:
    """拼接最终发给模型的请求文本：系统提示词 + 压缩后 JSON 上下文 + 输出格式约束。

    Uses structured context compaction when the context is large, replacing
    huge raw fields with context_summary/full_context_ref entries.
    """
    # Use compaction for large contexts; small contexts stay as-is
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    if CONTEXT_COMPACTION_ENABLED and len(context_json) > CONTEXT_COMPACTION_TRIGGER_CHARS:
        compacted = compact_context(
            context,
            trace_id=trace_id,
            round_id=round_id,
            agent_name=agent_name,
        )
        context_json = json.dumps(compacted, ensure_ascii=False, indent=2)

    return (
        prompt.strip()
        + "\n\n输入上下文 JSON:\n"
        + context_json
        + "\n\n输出要求: 只输出一个单行 JSON 对象，不要输出 <think>、markdown 或解释。"
    )


def _is_placeholder(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in _PLACEHOLDER_VALUES


def _validate_decision(
    agent_name: str,
    parsed: dict[str, Any],
    fallback: dict[str, Any],
    context: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not parsed:
        return ["no_json_object"]

    required_fields = _required_fields_for_agent(str(agent_name))
    missing = sorted(field for field in required_fields if field not in parsed)
    errors.extend(f"missing_required_field:{field}" for field in missing)

    for key, value in parsed.items():
        if isinstance(value, str) and _is_placeholder(value):
            errors.append(f"placeholder_value:{key}")

    if "action_key" in parsed:
        action_key = str(parsed.get("action_key") or "")
        candidate_actions = context.get("candidate_actions")
        allowed = set(candidate_actions.keys()) if isinstance(candidate_actions, dict) else set()
        candidate_action_keys = context.get("candidate_action_keys")
        if isinstance(candidate_action_keys, list):
            allowed.update(str(key) for key in candidate_action_keys if str(key))
        fallback_action = fallback.get("action_key")
        if fallback_action:
            allowed.add(str(fallback_action))
        if allowed and action_key not in allowed:
            errors.append(f"invalid_action_key:{action_key}")
        if _is_placeholder(action_key):
            errors.append("placeholder_action_key")

    if "replay_sample_ratio" in parsed:
        raw_ratio = parsed.get("replay_sample_ratio")
        try:
            if raw_ratio is None:
                raise TypeError("missing replay_sample_ratio")
            ratio = float(raw_ratio)
            if ratio < 0.0 or ratio > 1.0:
                errors.append("replay_sample_ratio_out_of_range")
        except (TypeError, ValueError):
            errors.append("replay_sample_ratio_not_number")

    hyperparams = parsed.get("training_hyperparams", parsed.get("hyperparameters"))
    if "training_hyperparams" in parsed or "hyperparameters" in parsed:
        if not isinstance(hyperparams, dict):
            errors.append("training_hyperparams_must_be_object")

    if str(agent_name).startswith("inspection_agent.decision"):
        decision = parsed.get("decision")
        if decision is not None and str(decision) not in {
            "promote",
            "provisional_promote",
            "keep_branch",
            "prune",
            "rollback",
        }:
            errors.append(f"invalid_inspection_decision:{decision}")

    if str(agent_name).endswith("confidence") or "confidence" in parsed:
        if "confidence" in parsed:
            raw_confidence = parsed.get("confidence")
            try:
                if raw_confidence is None:
                    raise TypeError("missing confidence")
                confidence = float(raw_confidence)
                if confidence < 0.0 or confidence > 1.0:
                    errors.append("confidence_out_of_range")
            except (TypeError, ValueError):
                errors.append("confidence_not_number")

    if str(agent_name) == "evaluator_judge":
        for score_key in ("result_score", "step_score", "total_score"):
            if score_key in parsed:
                raw_score = parsed.get(score_key)
                try:
                    if raw_score is None:
                        raise TypeError(f"missing {score_key}")
                    score = float(raw_score)
                    if score < 0.0 or score > 1.0:
                        errors.append(f"{score_key}_out_of_range")
                    elif score not in {0.0, 1.0}:
                        errors.append(f"{score_key}_not_binary")
                except (TypeError, ValueError):
                    errors.append(f"{score_key}_not_number")

    return errors


def _apply_optional_defaults(agent_name: str, parsed: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(parsed)
    return normalized


def _build_retry_request(
    original_request: str,
    schema_errors: list[str],
) -> str:
    return (
        original_request
        + "\n\nYour previous output was invalid."
        + "\nValidation errors: "
        + json.dumps(schema_errors, ensure_ascii=False)
        + "\nReturn only one corrected JSON object."
        + "\nDo not output markdown, prose, <think>, placeholders, or fenced code blocks."
    )


def _run_json_decision(
    agent_name: str,
    prompt: str,
    context: dict[str, Any],
    fallback: dict[str, Any],
    trace_id: str,
    round_id: int,
    max_new_tokens: int | None = None,
    disable_thinking: bool = True,
    stop_after_json: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], str, bool, dict[str, Any]]:
    """执行一次 LLM Agent JSON 决策的核心流程。

    1. 若 USE_LLM_AGENTS 关闭，直接返回 fallback（硬编码模式）
    2. 拼接 prompt + 上下文，调用 run_model_batch 获取模型输出
    3. 从原始输出中提取 JSON，与 fallback 合并（LLM 覆盖 fallback 中的同名字段）
    4. 持久化决策记录并写入 A/B 对比日志
    返回 (合并后的结果, 原始解析 JSON, 原始文本, schema 是否有效, 审计信息)。
    """
    if not USE_LLM_AGENTS:
        return (
            dict(fallback),
            {},
            "",
            False,
            {
                "json_parse_ok": False,
                "schema_valid": False,
                "schema_errors": ["llm_agents_disabled"],
                "retry_count": 0,
                "fallback_used": True,
                "final_result_source": "fallback",
                "exception": "",
                "raw_text": "",
                "raw_parsed": {},
            },
        )

    effective_max_new_tokens = max_new_tokens or LLM_AGENT_MAX_NEW_TOKENS
    request = _build_request(prompt, context, trace_id=trace_id, round_id=round_id, agent_name=agent_name)
    request_char_length = len(request)
    estimated_prompt_tokens = _estimate_token_count(request)
    prompt_budget = _prompt_budget_for(effective_max_new_tokens)
    prompt_truncated_estimate = estimated_prompt_tokens > prompt_budget
    raw_text = ""
    raw_parsed: dict[str, Any] = {}
    result = dict(fallback)
    schema_errors: list[str] = ["not_run"]
    retry_count = 0
    exception_text = ""
    attempts: list[dict[str, Any]] = []
    try:
        for attempt in range(2):
            retry_count = attempt
            attempt_request = request if attempt == 0 else _build_retry_request(request, schema_errors)
            raw_text = run_model_batch(
                AGENT_BASE_MODEL_NAME,
                [attempt_request],
                max_new_tokens=effective_max_new_tokens,
                prepend_math_instruction=False,
                disable_thinking=disable_thinking,
                stop_after_json=stop_after_json,
            )[0]
            raw_parsed = _extract_json(raw_text)
            schema_errors = _validate_decision(agent_name, raw_parsed, fallback, context)
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_text_len": len(raw_text or ""),
                    "raw_text": (raw_text or "")[:1000],
                    "json_parse_ok": bool(raw_parsed),
                    "schema_errors": list(schema_errors),
                }
            )
            if not schema_errors:
                result.update(_apply_optional_defaults(agent_name, raw_parsed))
                break
    except Exception as exc:
        exception_text = f"{type(exc).__name__}: {exc}"
        raw_text = exception_text

    json_parse_ok = bool(raw_parsed)
    schema_valid = bool(raw_parsed) and not schema_errors
    fallback_used = not schema_valid
    final_result_source = "llm_retried" if schema_valid and retry_count > 0 else "llm" if schema_valid else "fallback"
    _persist_decision(
        trace_id,
        round_id,
        agent_name,
        context,
        result,
        raw_text,
        metadata={
            "json_parse_ok": json_parse_ok,
            "schema_valid": schema_valid,
            "schema_errors": schema_errors if not schema_valid else [],
            "retry_count": retry_count,
            "fallback_used": fallback_used,
            "final_result_source": final_result_source,
            "exception": exception_text,
            "diagnostics": {
                "request_char_length": request_char_length,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "prompt_budget": prompt_budget,
                "max_new_tokens": effective_max_new_tokens,
                "prompt_truncated_estimate": prompt_truncated_estimate,
                "attempts": attempts,
            },
        },
    )

    if raw_parsed and schema_valid:
        _log_ab_comparison(
            trace_id=trace_id,
            round_id=round_id,
            agent_name=agent_name,
            raw_parsed=raw_parsed,
            final_result=result,
            fallback=fallback,
        )

    audit = {
        "json_parse_ok": json_parse_ok,
        "schema_valid": schema_valid,
        "schema_errors": schema_errors if not schema_valid else [],
        "retry_count": retry_count,
        "fallback_used": fallback_used,
        "final_result_source": final_result_source,
        "exception": exception_text,
        "raw_text": raw_text,
        "raw_parsed": dict(raw_parsed),
    }

    return result, raw_parsed, raw_text, schema_valid, audit


def _disabled_audit() -> dict[str, Any]:
    return {
        "json_parse_ok": False,
        "schema_valid": False,
        "schema_errors": ["llm_agents_disabled"],
        "retry_count": 0,
        "fallback_used": True,
        "final_result_source": "fallback",
        "exception": "",
        "raw_text": "",
        "raw_parsed": {},
    }


def _decision_diagnostics(
    *,
    request: str,
    max_new_tokens: int,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    estimated_prompt_tokens = _estimate_token_count(request)
    return {
        "request_char_length": len(request),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "prompt_budget": _prompt_budget_for(max_new_tokens),
        "max_new_tokens": max_new_tokens,
        "prompt_truncated_estimate": estimated_prompt_tokens > _prompt_budget_for(max_new_tokens),
        "attempts": attempts,
    }


def _finalize_json_decision(
    *,
    agent_name: str,
    context: dict[str, Any],
    fallback: dict[str, Any],
    trace_id: str,
    round_id: int,
    request: str,
    raw_text: str,
    raw_parsed: dict[str, Any],
    schema_errors: list[str],
    retry_count: int,
    exception_text: str,
    attempts: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    result = dict(fallback)
    schema_valid = bool(raw_parsed) and not schema_errors
    if schema_valid:
        result.update(_apply_optional_defaults(agent_name, raw_parsed))
    fallback_used = not schema_valid
    final_result_source = "llm_retried" if schema_valid and retry_count > 0 else "llm" if schema_valid else "fallback"
    audit = {
        "json_parse_ok": bool(raw_parsed),
        "schema_valid": schema_valid,
        "schema_errors": schema_errors if not schema_valid else [],
        "retry_count": retry_count,
        "fallback_used": fallback_used,
        "final_result_source": final_result_source,
        "exception": exception_text,
        "raw_text": raw_text,
        "raw_parsed": dict(raw_parsed),
    }
    metadata = {
        **audit,
        "diagnostics": _decision_diagnostics(
            request=request,
            max_new_tokens=max_new_tokens,
            attempts=attempts,
        ),
    }
    _persist_decision(
        trace_id,
        round_id,
        agent_name,
        context,
        result,
        raw_text,
        metadata=metadata,
    )
    if raw_parsed and schema_valid:
        _log_ab_comparison(
            trace_id=trace_id,
            round_id=round_id,
            agent_name=agent_name,
            raw_parsed=raw_parsed,
            final_result=result,
            fallback=fallback,
        )
    return {"result": result, "audit": audit}


def decide_json_with_audit_batch(
    *,
    agent_name: str,
    prompt: str,
    requests: list[dict[str, Any]],
    max_new_tokens: int | None = None,
    disable_thinking: bool = True,
    stop_after_json: bool = True,
) -> list[dict[str, Any]]:
    """Batch JSON decisions and retry only rows that fail schema validation."""
    if not requests:
        return []

    if not USE_LLM_AGENTS:
        return [
            {
                "result": dict(req.get("fallback") or {}),
                "audit": _disabled_audit(),
            }
            for req in requests
        ]

    effective_max_new_tokens = max_new_tokens or LLM_AGENT_MAX_NEW_TOKENS
    prepared: list[dict[str, Any]] = []
    for req in requests:
        context = dict(req.get("context") or {})
        fallback = dict(req.get("fallback") or {})
        trace_id = str(req.get("trace_id") or "")
        round_id = int(req.get("round_id") or 0)
        request_text = _build_request(prompt, context, trace_id=trace_id, round_id=round_id, agent_name=agent_name)
        prepared.append(
            {
                "context": context,
                "fallback": fallback,
                "trace_id": trace_id,
                "round_id": round_id,
                "request": request_text,
                "raw_text": "",
                "raw_parsed": {},
                "schema_errors": ["not_run"],
                "retry_count": 0,
                "exception": "",
                "attempts": [],
            }
        )

    pending_indices = list(range(len(prepared)))
    try:
        for attempt in range(2):
            if not pending_indices:
                break
            prompts = [
                prepared[idx]["request"]
                if attempt == 0
                else _build_retry_request(prepared[idx]["request"], prepared[idx]["schema_errors"])
                for idx in pending_indices
            ]
            raw_outputs = run_model_batch(
                AGENT_BASE_MODEL_NAME,
                prompts,
                max_new_tokens=effective_max_new_tokens,
                prepend_math_instruction=False,
                disable_thinking=disable_thinking,
                stop_after_json=stop_after_json,
            )
            next_pending: list[int] = []
            for idx, raw_text in zip(pending_indices, raw_outputs, strict=False):
                row = prepared[idx]
                raw_parsed = _extract_json(str(raw_text or ""))
                schema_errors = _validate_decision(
                    agent_name,
                    raw_parsed,
                    row["fallback"],
                    row["context"],
                )
                row["raw_text"] = str(raw_text or "")
                row["raw_parsed"] = raw_parsed
                row["schema_errors"] = schema_errors
                row["retry_count"] = attempt
                row["attempts"].append(
                    {
                        "attempt": attempt,
                        "raw_text_len": len(str(raw_text or "")),
                        "raw_text": str(raw_text or "")[:1000],
                        "json_parse_ok": bool(raw_parsed),
                        "schema_errors": list(schema_errors),
                    }
                )
                if schema_errors:
                    next_pending.append(idx)
            pending_indices = next_pending
    except Exception as exc:
        exception_text = f"{type(exc).__name__}: {exc}"
        for idx in pending_indices:
            prepared[idx]["exception"] = exception_text
            prepared[idx]["raw_text"] = exception_text

    return [
        _finalize_json_decision(
            agent_name=agent_name,
            context=row["context"],
            fallback=row["fallback"],
            trace_id=row["trace_id"],
            round_id=row["round_id"],
            request=row["request"],
            raw_text=row["raw_text"],
            raw_parsed=row["raw_parsed"],
            schema_errors=row["schema_errors"],
            retry_count=row["retry_count"],
            exception_text=row["exception"],
            attempts=row["attempts"],
            max_new_tokens=effective_max_new_tokens,
        )
        for row in prepared
    ]


def decide_json(
    agent_name: str,
    prompt: str,
    context: dict[str, Any],
    fallback: dict[str, Any],
    trace_id: str = "",
    round_id: int = 0,
    max_new_tokens: int | None = None,
    disable_thinking: bool = True,
    stop_after_json: bool = True,
) -> dict[str, Any]:
    """让固定底模 Agent 产出 JSON 决策；失败时回退到确定性策略。

    Prompt 中只描述 schema（字段名/类型/取值范围），不注入 fallback JSON 示例值，
    防止小模型照抄默认值。
    """
    result, _raw_parsed, _raw_text, _schema_valid, _audit = _run_json_decision(
        agent_name=agent_name,
        prompt=prompt,
        context=context,
        fallback=fallback,
        trace_id=trace_id,
        round_id=round_id,
        max_new_tokens=max_new_tokens,
        disable_thinking=disable_thinking,
        stop_after_json=stop_after_json,
    )
    return result


def decide_json_with_audit(
    agent_name: str,
    prompt: str,
    context: dict[str, Any],
    fallback: dict[str, Any],
    trace_id: str = "",
    round_id: int = 0,
    max_new_tokens: int | None = None,
    disable_thinking: bool = True,
    stop_after_json: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the JSON decision plus parse/schema audit metadata."""
    result, _raw_parsed, _raw_text, _schema_valid, audit = _run_json_decision(
        agent_name=agent_name,
        prompt=prompt,
        context=context,
        fallback=fallback,
        trace_id=trace_id,
        round_id=round_id,
        max_new_tokens=max_new_tokens,
        disable_thinking=disable_thinking,
        stop_after_json=stop_after_json,
    )
    return result, audit


def decide_json_leaf(
    agent_name: str,
    prompt: str,
    context: dict[str, Any],
    field_name: str,
    fallback_value: Any,
    trace_id: str = "",
    round_id: int = 0,
    max_new_tokens: int | None = None,
    disable_thinking: bool = True,
) -> tuple[Any, bool]:
    """让 LLM Agent 仅决策单个叶子字段。

    将决策粒度拆到字段级别，使 prompt 简单到 0.6B 小模型也能稳定输出。
    调用方仍需自行校验/裁剪返回值（如 clamp 到合法范围）。
    返回 (字段值, 是否成功解析到 JSON)。
    """
    result, _raw_parsed, _raw_text, schema_valid, _audit = _run_json_decision(
        agent_name=agent_name,
        prompt=prompt,
        context=context,
        fallback={field_name: fallback_value},
        trace_id=trace_id,
        round_id=round_id,
        max_new_tokens=max_new_tokens or LLM_AGENT_LEAF_MAX_NEW_TOKENS,
        disable_thinking=disable_thinking,
        stop_after_json=True,
    )
    return result.get(field_name, fallback_value), schema_valid


def _log_ab_comparison(
    trace_id: str,
    round_id: int,
    agent_name: str,
    raw_parsed: dict[str, Any],
    final_result: dict[str, Any],
    fallback: dict[str, Any],
) -> None:
    """A/B 对比日志：对比 LLM 原始输出 vs 最终采纳结果。

    记录两类差异：
    - llm_overrides_fallback: LLM 修改了 fallback 且被最终采纳的字段
    - llm_clamped_by_code: LLM 输出被代码层校验/裁剪掉的字段
    用于评估 Agent 决策质量和代码层校验的必要性。
    """
    if not trace_id:
        return
    try:
        path = get_session_dir(trace_id) / "llm_ab_comparison.jsonl"
        # Find keys where LLM deviated from fallback AND survived into final
        llm_overrides = {}
        for k, v in raw_parsed.items():
            if k in fallback and str(v) != str(fallback[k]):
                llm_overrides[k] = {"raw": v, "fallback": fallback[k], "final": final_result.get(k, v)}
        # Find keys where LLM was clamped/rejected
        clamped = {}
        for k, v in raw_parsed.items():
            if k in final_result and str(v) != str(final_result.get(k)):
                clamped[k] = {"raw": v, "final": final_result[k]}
        entry = {
            "round": round_id,
            "agent": agent_name,
            "llm_overrides_fallback": llm_overrides,
            "llm_clamped_by_code": clamped,
            "llm_is_decision_maker": bool(llm_overrides),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def prompt_for_agent(state: dict[str, Any], agent_name: str, default_prompt: str) -> str:
    """从提示词设计师产物中读取提示词；没有则使用默认框架。"""
    # Code domain: use code-specific prompts when available
    import os
    if os.getenv("DOMAIN", "").strip().lower() == "code":
        from src.tools.agent_prompts import CODE_DOMAIN_PROMPTS
        if agent_name in CODE_DOMAIN_PROMPTS:
            return CODE_DOMAIN_PROMPTS[agent_name]

    if agent_name in _CRITICAL_PROMPT_AGENTS or agent_name.startswith("parameter_master"):
        return default_prompt
    pack = state.get("agent_prompt_pack", {}) if isinstance(state, dict) else {}
    if not isinstance(pack, dict):
        return default_prompt
    prompts = pack.get("prompts", {})
    if not isinstance(prompts, dict):
        return default_prompt
    prompt = prompts.get(agent_name)
    return str(prompt) if prompt else default_prompt


def clamp_distribution(raw: Any, allowed: set[str], fallback: dict[str, float]) -> dict[str, float]:
    """清洗 LLM 给出的权重，确保只留下允许标签且总和为 1。"""
    if not isinstance(raw, dict):
        raw = {}
    cleaned: dict[str, float] = {}
    for key, value in raw.items():
        if str(key) not in allowed:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight > 0:
            cleaned[str(key)] = weight
    if not cleaned:
        cleaned = dict(fallback)
    total = sum(cleaned.values())
    if total <= 0:
        cleaned = dict(fallback)
        total = sum(cleaned.values())
    return {key: value / total for key, value in cleaned.items() if value > 0}


def load_prompt_overrides(path: str) -> dict[str, str]:
    """DSPy 优化后可把 prompt 覆盖写成 JSON，运行时按需读取。"""
    if not path:
        return {}
    prompt_path = Path(path)
    if not prompt_path.exists():
        return {}
    try:
        loaded = json.loads(prompt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}
