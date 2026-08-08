"""Data Builder 确定性回退值。

USE_LLM_AGENTS=0 时，编排层跳过所有 Agent 调用，直接用这些默认值。
行为与当前 data_builder.py 硬编码逻辑完全一致，保证向后兼容。
"""

# =============================================================================
# 列选择回退
# =============================================================================
DEFAULT_COL_DECISION: dict = {
    "question_col": "question_text",
    "answer_cols": ["gold_answer"],
    "metadata_cols": [
        "module",
        "category",
        "dynamic_difficulty",
        "source_role",
        "source_dataset_id",
        "source_dataset_row_id",
        "source_dataset_split",
        "source_dataset_subset",
        "source_dataset_requested_split",
    ],
}

# =============================================================================
# 模板回退
# =============================================================================
DEFAULT_TEMPLATE_DECISION: dict = {
    "prompt_template": "请解答下面的题目。\n{question_text}",
    "response_template": None,
    "output_format": "alpaca",
}

# =============================================================================
# 过滤回退
# =============================================================================
DEFAULT_FILTER_DECISION: dict = {
    "min_question_len": 0,
    "max_question_len": 4096,
    "min_answer_len": 0,
    "max_answer_len": 8192,
    "dedup_by": "question",
    "drop_empty_cols": ["question_text", "gold_answer"],
    "lang_filter": None,
    "max_samples": 500,
}

# =============================================================================
# 切分回退
# =============================================================================
DEFAULT_SPLIT_DECISION: dict = {
    "strategy": "ratio",
    "test_ratio": 0.15,
    "seed": 42,
}
