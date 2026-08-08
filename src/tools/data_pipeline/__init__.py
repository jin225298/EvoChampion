"""数据构建流水线 — 6 个原子工具 + Agent 决策层。

分层契约：
- 工具层：纯函数，只执行不决策。全部参数由上层传入。
- Agent 层：读工具输出 JSON，输出结构化决策 JSON。
- 编排层：coordinator 调用 Agent → 工具 → 打包。

USE_LLM_AGENTS=0 时所有 Agent 跳过，走 fallbacks.py 的确定性默认值。
"""

from src.tools.data_pipeline.fallbacks import (
    DEFAULT_COL_DECISION,
    DEFAULT_FILTER_DECISION,
    DEFAULT_SPLIT_DECISION,
    DEFAULT_TEMPLATE_DECISION,
)
from src.tools.data_pipeline.inspect_dataset import inspect_dataset
from src.tools.data_pipeline.build_sft_samples import build_sft_samples
from src.tools.data_pipeline.filter_samples import filter_samples
from src.tools.data_pipeline.split_train_test import split_train_test
from src.tools.data_pipeline.decontaminate import decontaminate
from src.tools.data_pipeline.pack_for_trainer import pack_for_trainer
from src.tools.data_pipeline.log_parser import parse_training_log

__all__ = [
    "inspect_dataset",
    "build_sft_samples",
    "filter_samples",
    "split_train_test",
    "decontaminate",
    "pack_for_trainer",
    "parse_training_log",
    "DEFAULT_COL_DECISION",
    "DEFAULT_TEMPLATE_DECISION",
    "DEFAULT_FILTER_DECISION",
    "DEFAULT_SPLIT_DECISION",
]
