"""data_builder_helpers — 工具层封装，供 data_builder.py 调用。

层次关系：
    data_builder.py（策略层 — LangGraph 节点）
        ├─ 调用本文件的 Agent 决策函数（_decide_columns / _decide_template 等）
        ├─ 调用 src/tools/data_pipeline/ 的 6 个原子工具
        ├─ 负责 replay buffer、holdout eval、test buffer、probe pool 等状态管理
        └─ 返回完整的 state 字段给 LangGraph

    src/tools/data_builder_helpers.py（工具封装层 — 非独立节点）
        ├─ 封装 Agent 决策逻辑（prompt_for_agent → decide_json）
        ├─ 封装工具调用流程（inspect → build → filter → split → decontaminate → pack）
        └─ USE_LLM_AGENTS=0 时走 fallbacks.py

    src/tools/data_pipeline/（原子工具层）
        └─ inspect_dataset, build_sft_samples, filter_samples, split_train_test,
           decontaminate, pack_for_trainer（纯函数，不做决策）

USE_LLM_AGENTS=0 时跳过所有 LLM 调用，走 fallbacks.py 确定性默认值。
"""

import json
from pathlib import Path

from config.settings import USE_LLM_AGENTS, get_session_dir
from src.models.state import EvoState
from src.tools.agent_prompts import (
    DATASET_INSPECTOR_PROMPT,
    FILTER_PARAMS_PROMPT,
    PROMPT_TEMPLATE_PROMPT,
    RESOURCE_ADAPT_PROMPT,
)
from src.tools.llm_decision import decide_json, decide_json_leaf, prompt_for_agent
from src.tools.data_pipeline.inspect_dataset import inspect_dataset
from src.tools.data_pipeline.pack_for_trainer import pack_for_trainer
from src.tools.data_pipeline.log_parser import parse_training_log
from src.tools.data_pipeline.fallbacks import (
    DEFAULT_COL_DECISION,
    DEFAULT_FILTER_DECISION,
    DEFAULT_SPLIT_DECISION,
    DEFAULT_TEMPLATE_DECISION,
)


# =============================================================================
# Agent 决策函数（遵循现有模式：prompt_for_agent → decide_json*）
# =============================================================================

def _decide_columns(inspect_result: dict, state: EvoState) -> dict:
    """Agent 阅读数据集结构，决定哪列做题目、哪列做答案。"""
    if not USE_LLM_AGENTS:
        return dict(DEFAULT_COL_DECISION)

    prompt = prompt_for_agent(dict(state), "dataset_inspector", DATASET_INSPECTOR_PROMPT)
    columns = inspect_result.get("columns", [])
    candidates = inspect_result.get("column_candidates", {})
    col_names = [c["name"] for c in columns]

    question_col, _ = decide_json_leaf(
        agent_name="dataset_inspector.question_col",
        prompt=prompt,
        context={
            "available_columns": col_names,
            "likely_question_cols": candidates.get("likely_question_cols", []),
            "sample_rows": inspect_result.get("sample_rows", [])[:2],
            "user_goal": state.get("user_goal", ""),
        },
        field_name="question_col",
        fallback_value=DEFAULT_COL_DECISION["question_col"],
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
        max_new_tokens=96,
    )
    if str(question_col) not in col_names:
        question_col = DEFAULT_COL_DECISION["question_col"]

    answer_cols_raw, _ = decide_json_leaf(
        agent_name="dataset_inspector.answer_cols",
        prompt=prompt,
        context={
            "available_columns": col_names,
            "likely_answer_cols": candidates.get("likely_answer_cols", []),
        },
        field_name="answer_cols",
        fallback_value=DEFAULT_COL_DECISION["answer_cols"],
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
        max_new_tokens=96,
    )
    if isinstance(answer_cols_raw, str):
        answer_cols_raw = [answer_cols_raw]
    if not isinstance(answer_cols_raw, list):
        answer_cols_raw = DEFAULT_COL_DECISION["answer_cols"]
    answer_cols = [c for c in answer_cols_raw if str(c) in col_names]
    if not answer_cols:
        answer_cols = DEFAULT_COL_DECISION["answer_cols"]

    return {
        "question_col": str(question_col),
        "answer_cols": [str(c) for c in answer_cols],
        "metadata_cols": [c for c in col_names if c not in [question_col] + answer_cols],
    }


def _decide_template(inspect_result: dict, col_decision: dict, state: EvoState) -> dict:
    """Agent 根据数据类型动态生成 instruction 模板。"""
    style_values = {
        str(row.get("target_style") or "")
        for row in inspect_result.get("sample_rows", [])
        if isinstance(row, dict)
    }
    if not USE_LLM_AGENTS:
        decision = dict(DEFAULT_TEMPLATE_DECISION)
        if "cot" in style_values:
            decision["response_template"] = "{train_output}\n\n答案：{gold_answer}"
        return decision

    prompt = prompt_for_agent(dict(state), "prompt_template", PROMPT_TEMPLATE_PROMPT)
    question_col = col_decision.get("question_col", "question_text")
    answer_cols = col_decision.get("answer_cols", ["gold_answer"])
    columns = inspect_result.get("columns", [])

    template, _ = decide_json_leaf(
        agent_name="prompt_template.template",
        prompt=prompt,
        context={
            "question_col": question_col,
            "answer_cols": answer_cols,
            "user_goal": state.get("user_goal", ""),
            "column_names": [c["name"] for c in columns],
            "sample_rows": inspect_result.get("sample_rows", [])[:2],
            "field_name": "prompt_template",
        },
        field_name="prompt_template",
        fallback_value=DEFAULT_TEMPLATE_DECISION["prompt_template"],
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
        max_new_tokens=128,
    )

    response_template = DEFAULT_TEMPLATE_DECISION.get("response_template")
    if "cot" in style_values and {"train_output", "gold_answer"}.issubset({c["name"] for c in columns}):
        response_template = "{train_output}\n\n答案：{gold_answer}"
    if len(answer_cols) > 1:
        response_template, _ = decide_json_leaf(
            agent_name="prompt_template.response",
            prompt=prompt,
            context={"answer_cols": answer_cols, "field_name": "response_template"},
            field_name="response_template",
            fallback_value="\n".join(f"{{{c}}}" for c in answer_cols),
            trace_id=state.get("trace_id", ""),
            round_id=state.get("round_id", 0),
            max_new_tokens=64,
        )

    return {
        "prompt_template": str(template),
        "response_template": str(response_template) if response_template else None,
        "output_format": "alpaca",
    }


def _decide_filter(inspect_result: dict, col_decision: dict, state: EvoState) -> dict:
    """Agent 决定过滤参数。"""
    if not USE_LLM_AGENTS:
        return dict(DEFAULT_FILTER_DECISION)

    prompt = prompt_for_agent(dict(state), "filter_params", FILTER_PARAMS_PROMPT)
    columns = inspect_result.get("columns", [])
    col_names = [c["name"] for c in columns]

    decision = decide_json(
        agent_name="filter_params",
        prompt=prompt,
        context={
            "columns": col_names,
            "question_col": col_decision.get("question_col", "question_text"),
            "sample_rows": inspect_result.get("sample_rows", [])[:2],
        },
        fallback=DEFAULT_FILTER_DECISION,
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
        max_new_tokens=128,
    )

    result = dict(DEFAULT_FILTER_DECISION)
    for key in result:
        if key in decision:
            val = decision[key]
            if key in ("min_question_len", "max_question_len", "min_answer_len", "max_answer_len", "max_samples"):
                try:
                    result[key] = max(0, int(val))
                except (TypeError, ValueError):
                    pass
            elif key == "dedup_by" and val in ("question", "question+answer", "hash"):
                result[key] = val
            elif key == "lang_filter" and val in ("zh", "en", None):
                result[key] = val
    return result


def _decide_split(inspect_result: dict, filter_result: dict, state: EvoState) -> dict:
    """Return an audit-only split policy without calling an LLM agent."""
    result = dict(DEFAULT_SPLIT_DECISION)
    raw_splits = inspect_result.get("splits_available", [])
    splits = sorted({
        str(split).strip().lower()
        for split in raw_splits
        if str(split).strip()
    })
    has_train = "train" in splits
    has_eval = any(split in splits for split in ("test", "validation", "valid", "dev", "eval"))
    if has_train and has_eval:
        result["strategy"] = "use_existing_split"
        result["test_ratio"] = 0.0
        reason = "dataset exposes train and eval splits; recorded for audit"
    else:
        result["strategy"] = "ratio"
        result["test_ratio"] = DEFAULT_SPLIT_DECISION["test_ratio"]
        reason = "fixed experiment protocol owns train/cotest/test/probe ratios"
    result.update({
        "decision_source": "deterministic_policy",
        "agent_enabled": False,
        "audit_only": True,
        "splits_available": splits,
        "sample_count": filter_result.get("kept", 0),
        "reason": reason,
    })
    return result


def _decide_resource(log_path: str, current_batch: int, current_ga: int, current_max: int, state: EvoState) -> dict:
    """Agent 根据训练日志调整资源配置。"""
    default = {
        "action": "keep",
        "per_device_train_batch_size": current_batch,
        "gradient_accumulation_steps": current_ga,
        "max_samples_override": current_max,
        "reason": "",
    }

    if not USE_LLM_AGENTS:
        return default

    log_info = parse_training_log(log_path)
    oom_count = len(log_info.get("oom_lines", []))

    # 确定性预判断：连续 OOM 跳过大模型直接降配
    if oom_count >= 3:
        new_batch = max(1, current_batch // 2)
        return {
            "action": "reduce_samples",
            "per_device_train_batch_size": new_batch,
            "gradient_accumulation_steps": current_ga * 2,
            "max_samples_override": max(100, current_max // 2),
            "reason": f"连续检测到 {oom_count} 次 OOM",
        }

    if log_info.get("oom_detected"):
        new_batch = max(1, current_batch // 2)
        return {
            "action": "reduce_batch",
            "per_device_train_batch_size": new_batch,
            "gradient_accumulation_steps": current_ga * 2,
            "max_samples_override": current_max,
            "reason": "检测到 OOM",
        }

    # 无 OOM → LLM 微调
    prompt = prompt_for_agent(dict(state), "resource_adapt", RESOURCE_ADAPT_PROMPT)
    decision = decide_json(
        agent_name="resource_adapt",
        prompt=prompt,
        context={
            "log_summary": {k: v for k, v in log_info.items() if k != "oom_lines"},
            "current_batch_size": current_batch,
            "current_ga": current_ga,
            "current_max_samples": current_max,
        },
        fallback={"action": "keep", "reason": "no OOM detected"},
        trace_id=state.get("trace_id", ""),
        round_id=state.get("round_id", 0),
        max_new_tokens=96,
    )

    action = str(decision.get("action", "keep"))
    if action not in ("keep", "reduce_batch", "reduce_samples"):
        action = "keep"
    return {
        "action": action,
        "per_device_train_batch_size": current_batch,
        "gradient_accumulation_steps": current_ga,
        "max_samples_override": current_max,
        "reason": str(decision.get("reason", "")),
    }


def _inspect_standardized_questions(questions: list[dict]) -> dict:
    """Build an inspect_dataset-shaped summary for normalized in-memory rows."""
    sample_rows = [q for q in questions if isinstance(q, dict)][:5]
    split_values = sorted({
        str(q.get("source_dataset_split"))
        for q in questions
        if isinstance(q, dict) and q.get("source_dataset_split")
    })
    subset_values = sorted({
        str(q.get("source_dataset_subset"))
        for q in questions
        if isinstance(q, dict) and q.get("source_dataset_subset")
    })
    requested_split_values = sorted({
        str(q.get("source_dataset_requested_split"))
        for q in questions
        if isinstance(q, dict) and q.get("source_dataset_requested_split")
    })
    reviewer_split_values = sorted({
        str(split)
        for q in questions
        if isinstance(q, dict)
        for split in (q.get("source_dataset_split_names") or [])
        if str(split)
    })
    schema_values: list[dict] = []
    reviewer_first_rows: list[dict] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        schema_value = q.get("source_dataset_schema")
        if isinstance(schema_value, dict) and schema_value:
            schema_values.append(dict(schema_value))
        first_row_value = q.get("source_dataset_first_row")
        if isinstance(first_row_value, dict) and first_row_value:
            reviewer_first_rows.append(dict(first_row_value))
    keys = sorted({str(key) for row in sample_rows for key in row.keys()})
    columns = []
    for key in keys:
        values = [row.get(key) for row in sample_rows]
        string_values = [str(v) for v in values if isinstance(v, str)]
        non_null = [v for v in values if v is not None]
        dtype = "string" if not non_null or all(isinstance(v, str) for v in non_null) else type(non_null[0]).__name__
        columns.append({
            "name": key,
            "dtype": dtype,
            "avg_len": round(sum(len(v) for v in string_values) / len(string_values), 1) if string_values else 0,
            "null_ratio": round(1.0 - (len(non_null) / len(values)), 3) if values else 0.0,
        })

    question_keywords = ("question", "problem", "input", "instruction", "prompt", "query")
    answer_keywords = ("answer", "output", "response", "target", "solution", "completion")
    candidates = {
        "likely_question_cols": [],
        "likely_answer_cols": [],
        "other_string_cols": [],
        "numeric_cols": [],
        "list_dict_cols": [],
    }
    for column in columns:
        name = column["name"]
        lowered = name.lower()
        dtype = str(column.get("dtype", "")).lower()
        if "string" in dtype or "str" in dtype:
            if any(item in lowered for item in question_keywords):
                candidates["likely_question_cols"].append(name)
            if any(item in lowered for item in answer_keywords):
                candidates["likely_answer_cols"].append(name)
            if not any(item in lowered for item in question_keywords + answer_keywords):
                candidates["other_string_cols"].append(name)
        elif any(item in dtype for item in ("int", "float", "number")):
            candidates["numeric_cols"].append(name)
        elif any(item in dtype for item in ("list", "dict", "sequence", "struct")):
            candidates["list_dict_cols"].append(name)

    return {
        "source": "classified_questions",
        "subset": subset_values[0] if len(subset_values) == 1 else None,
        "subsets_available": subset_values,
        "split": split_values[0] if len(split_values) == 1 else "in_memory",
        "requested_split": requested_split_values[0] if len(requested_split_values) == 1 else None,
        "num_rows": len([q for q in questions if isinstance(q, dict)]),
        "columns": columns,
        "splits_available": reviewer_split_values or split_values,
        "first_row": dict(reviewer_first_rows[0]) if reviewer_first_rows else (dict(sample_rows[0]) if sample_rows else {}),
        "sample_rows": sample_rows,
        "column_candidates": candidates,
        "schema": dict(schema_values[0]) if schema_values else {},
        "error": None,
    }


def _coerce_standard_columns(col_decision: dict, inspect_result: dict) -> dict:
    """Keep agent column choices compatible with normalized classifier rows."""
    available = {column["name"] for column in inspect_result.get("columns", [])}
    result = dict(col_decision)
    if result.get("question_col") not in available and "question_text" in available:
        result["question_col"] = "question_text"
    answer_cols = [
        col for col in result.get("answer_cols", [])
        if col in available
    ]
    if not answer_cols and "gold_answer" in available:
        answer_cols = ["gold_answer"]
    result["answer_cols"] = answer_cols or list(DEFAULT_COL_DECISION["answer_cols"])
    metadata_cols = [
        col for col in result.get("metadata_cols", [])
        if col in available and col not in {result.get("question_col"), *result["answer_cols"]}
    ]
    if not metadata_cols:
        metadata_cols = [
            col for col in available
            if col not in {result.get("question_col"), *result["answer_cols"]}
        ]
    result["metadata_cols"] = sorted(metadata_cols)
    return result


def decide_data_pipeline_plan(
    classified_questions: list[dict],
    state: EvoState,
    search_source: str = "",
) -> dict:
    """Return data-pipeline decisions for the data_builder node.

    The graph node owns split state, replay, holdout, and LangGraph message
    output. This helper only inspects data, asks leaf agents for tool
    parameters, and returns deterministic fallbacks when agents are disabled.
    """
    inspect_result = _inspect_standardized_questions(classified_questions)
    if search_source:
        try:
            external_inspect = inspect_dataset(source=search_source, streaming=True)
            if not external_inspect.get("error"):
                inspect_result["external_source"] = external_inspect
        except Exception as exc:
            inspect_result["external_source_error"] = f"{type(exc).__name__}: {exc}"

    col_decision = _coerce_standard_columns(_decide_columns(inspect_result, state), inspect_result)
    template_decision = _decide_template(inspect_result, col_decision, state)
    filter_decision = _decide_filter(inspect_result, col_decision, state)

    current_hyperparams = state.get("current_training_hyperparams", {})
    current_batch = int(current_hyperparams.get("per_device_train_batch_size", 2) or 2)
    current_ga = int(current_hyperparams.get("gradient_accumulation_steps", 8) or 8)
    current_max = int(filter_decision.get("max_samples", 500) or 500)
    trace_id = state.get("trace_id", "")
    round_id = int(state.get("round_id", 0) or 0)
    session_dir = get_session_dir(trace_id) if trace_id else None
    prev_log = str(session_dir.parent / f"train_log_{trace_id}_round{round_id - 1}.txt") if session_dir and round_id > 1 else ""
    if prev_log and Path(prev_log).exists():
        resource_decision = _decide_resource(prev_log, current_batch, current_ga, current_max, state)
        if resource_decision.get("action") == "reduce_samples":
            filter_decision["max_samples"] = resource_decision.get("max_samples_override", current_max)
    else:
        resource_decision = {
            "action": "keep",
            "per_device_train_batch_size": current_batch,
            "gradient_accumulation_steps": current_ga,
            "max_samples_override": current_max,
            "reason": "",
        }

    split_decision = _decide_split(inspect_result, {"kept": len(classified_questions)}, state)
    return {
        "inspect_result": inspect_result,
        "col_decision": col_decision,
        "template_decision": template_decision,
        "filter_decision": filter_decision,
        "split_decision": split_decision,
        "resource_decision": resource_decision,
    }


def resource_overrides_from_plan(plan: dict) -> dict:
    """Extract trainer hyperparameter overrides from a data pipeline plan."""
    resource_decision = plan.get("resource_decision", {}) if isinstance(plan, dict) else {}
    if resource_decision.get("action") not in {"reduce_batch", "reduce_samples"}:
        return {}
    overrides = {}
    if resource_decision.get("per_device_train_batch_size") is not None:
        overrides["per_device_train_batch_size"] = resource_decision["per_device_train_batch_size"]
    if resource_decision.get("gradient_accumulation_steps") is not None:
        overrides["gradient_accumulation_steps"] = resource_decision["gradient_accumulation_steps"]
    return overrides


def pack_existing_splits_for_trainer(
    dataset_dir: Path,
    round_id: int,
    lf_val_path: str | Path | None = None,
    output_format: str = "alpaca",
) -> dict:
    """Register existing train/test/cotest/probe files for LLaMA-Factory."""
    return pack_for_trainer(
        train_path=str(dataset_dir / "train.json"),
        output_dir=str(dataset_dir),
        test_path=str(dataset_dir / "test.json"),
        cotest_path=str(dataset_dir / "cotest.json"),
        probe_path=str(dataset_dir / "probe.json"),
        lf_val_path=str(lf_val_path) if lf_val_path else None,
        dataset_name=f"round_{round_id}_train",
        output_format=output_format,
    )


def write_data_pipeline_log(
    dataset_dir: Path,
    plan: dict,
    stats: dict,
) -> str:
    """Persist data-pipeline decisions next to the round dataset artifacts."""
    log_payload = {
        "plan": {
            "col_decision": plan.get("col_decision", {}),
            "template_decision": plan.get("template_decision", {}),
            "filter_decision": plan.get("filter_decision", {}),
            "split_decision": plan.get("split_decision", {}),
            "resource_decision": plan.get("resource_decision", {}),
        },
        "stats": stats,
    }
    path = dataset_dir / "pipeline_log.json"
    path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
