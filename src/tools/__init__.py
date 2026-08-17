from src.tools.hf_search import search_hf_datasets
from src.tools.llm_factory import launch_training


def judge_answer(*args, **kwargs):
    from src.tools.model_runner import judge_answer as _judge_answer

    return _judge_answer(*args, **kwargs)


def judge_prediction_for_item(*args, **kwargs):
    from src.tools.model_runner import judge_prediction_for_item as _judge_prediction_for_item

    return _judge_prediction_for_item(*args, **kwargs)


def run_model_once(*args, **kwargs):
    from src.tools.model_runner import run_model_once as _run_model_once

    return _run_model_once(*args, **kwargs)


__all__ = [
    "search_hf_datasets",
    "judge_answer",
    "judge_prediction_for_item",
    "run_model_once",
    "launch_training",
]
