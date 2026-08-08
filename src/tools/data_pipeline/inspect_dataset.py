"""inspect_dataset — 侦察兵工具。

扫描数据集结构，输出列信息、样本行、统计摘要。
Agent 读取此 JSON 后决定哪列做题目、哪列做答案。
纯函数，只检测不决策。
"""

import json
import importlib
from typing import Any


def inspect_dataset(
    source: str,
    subset: str | None = None,
    split: str = "train",
    max_sample_rows: int = 5,
    streaming: bool = True,
) -> dict[str, Any]:
    """扫描数据集，返回结构和样本信息。

    Args:
        source: HF dataset ID（如 "gsm8k"）或本地路径。
        subset: 数据集子集名（如 "main"），可选。
        split: 数据分割（"train", "test" 等）。
        max_sample_rows: 返回的样本行数。
        streaming: 是否流式加载。

    Returns:
        {
            "source": "gsm8k",
            "subset": "main",
            "split": "train",
            "requested_split": "train",
            "num_rows": 7473,
            "columns": [{"name": "question", "dtype": "string", "avg_len": 120, "null_ratio": 0.0}, ...],
            "splits_available": ["train", "test"],
            "first_row": {"question": "...", "answer": "..."},
            "sample_rows": [{"question": "...", "answer": "..."}, ...],
            "column_candidates": {
                "likely_question_cols": ["question"],
                "likely_answer_cols": ["answer"],
                "other_string_cols": [],
                "numeric_cols": [],
                "list_dict_cols": [],
            },
            "error": null,
        }
    """
    result: dict[str, Any] = {
        "source": source,
        "subset": subset,
        "split": split,
        "requested_split": split,
        "num_rows": 0,
        "columns": [],
        "splits_available": [],
        "first_row": {},
        "sample_rows": [],
        "column_candidates": {
            "likely_question_cols": [],
            "likely_answer_cols": [],
            "other_string_cols": [],
            "numeric_cols": [],
            "list_dict_cols": [],
        },
        "error": None,
    }

    try:
        datasets_module = importlib.import_module("datasets")
        get_dataset_split_names = datasets_module.get_dataset_split_names
        load_dataset = datasets_module.load_dataset
        normalized_subset = subset if subset and subset != "default" else None

        # ── 获取可用 splits ──
        try:
            result["splits_available"] = list(get_dataset_split_names(source, config_name=normalized_subset))
        except TypeError:
            try:
                result["splits_available"] = list(get_dataset_split_names(source))
            except Exception:
                pass
        except Exception:
            pass

        selected_split = split
        if result["splits_available"] and selected_split not in result["splits_available"]:
            selected_split = "train" if "train" in result["splits_available"] else result["splits_available"][0]
        result["split"] = selected_split

        # ── 加载数据 ──
        try:
            ds = load_dataset(
                source,
                name=normalized_subset,
                split=selected_split,
                streaming=streaming,
            )
        except Exception:
            ds = load_dataset(
                source,
                name=None if normalized_subset else "main",
                split=selected_split,
                streaming=streaming,
            )

        # ── 列信息 ──
        features = getattr(ds, "features", None)
        if features is not None and hasattr(features, "items"):
            for name, feat in features.items():
                dtype = str(getattr(feat, "dtype", "")).lower()
                result["columns"].append({
                    "name": str(name),
                    "dtype": dtype,
                    "avg_len": 0,
                    "null_ratio": 0.0,
                })

        # ── 样本行 + 动态统计 ──
        col_lengths: dict[str, list[int]] = {}
        null_counts: dict[str, int] = {}
        row_count = 0

        for row in ds:
            if row_count < max_sample_rows:
                sample = {}
                for k, v in dict(row).items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        sample[str(k)] = v
                    elif isinstance(v, (list, dict)):
                        sample[str(k)] = v
                    else:
                        sample[str(k)] = str(v)[:200]
                if row_count == 0:
                    result["first_row"] = dict(sample)
                result["sample_rows"].append(sample)

            for k, v in dict(row).items():
                key = str(k)
                if v is None:
                    null_counts[key] = null_counts.get(key, 0) + 1
                elif isinstance(v, str):
                    col_lengths.setdefault(key, []).append(len(v))
                elif isinstance(v, (list, dict)):
                    col_lengths.setdefault(key, []).append(len(str(v)))

            row_count += 1
            if row_count >= 2000:
                break

        result["num_rows"] = min(row_count, 2000)

        # ── 补充列统计 ──
        for col in result["columns"]:
            name = col["name"]
            lengths = col_lengths.get(name, [])
            if lengths:
                col["avg_len"] = round(sum(lengths) / len(lengths), 1)
            if row_count > 0:
                col["null_ratio"] = round(null_counts.get(name, 0) / row_count, 3)

        # Some HF feature objects stringify nested list/struct columns with an empty dtype.
        # Use sampled runtime values as a deterministic fallback so schema logic can see
        # conversation columns instead of forcing the LLM to guess from truncated strings.
        sampled_types: dict[str, str] = {}
        for sample in result["sample_rows"]:
            for name, value in sample.items():
                if isinstance(value, list):
                    sampled_types.setdefault(name, "list")
                elif isinstance(value, dict):
                    sampled_types.setdefault(name, "struct")

        for col in result["columns"]:
            if not col.get("dtype") and col["name"] in sampled_types:
                col["dtype"] = sampled_types[col["name"]]

        # 若 features 为空，从首行推断列信息
        if not result["columns"] and result["sample_rows"]:
            first = result["sample_rows"][0]
            for name, val in first.items():
                if isinstance(val, str):
                    dtype = "string"
                elif isinstance(val, (int, float)):
                    dtype = "number"
                elif isinstance(val, bool):
                    dtype = "bool"
                elif isinstance(val, list):
                    dtype = "list"
                elif isinstance(val, dict):
                    dtype = "struct"
                else:
                    dtype = "other"
                result["columns"].append({
                    "name": name,
                    "dtype": dtype,
                    "avg_len": 0,
                    "null_ratio": 0.0,
                })

        # ── 分类候选列 ──
        question_keywords = ("question", "problem", "input", "instruction", "prompt", "query")
        answer_keywords = ("answer", "output", "response", "target", "solution", "completion")

        for col in result["columns"]:
            name_lower = col["name"].lower()
            dtype = col.get("dtype", "")
            is_string = "string" in dtype or "text" in dtype or "large_string" in dtype

            if is_string:
                if any(kw in name_lower for kw in question_keywords):
                    result["column_candidates"]["likely_question_cols"].append(col["name"])
                if any(kw in name_lower for kw in answer_keywords):
                    result["column_candidates"]["likely_answer_cols"].append(col["name"])
                if not any(kw in name_lower for kw in question_keywords + answer_keywords):
                    result["column_candidates"]["other_string_cols"].append(col["name"])
            elif any(t in dtype for t in ("int", "float", "number")):
                result["column_candidates"]["numeric_cols"].append(col["name"])
            elif any(t in dtype for t in ("list", "sequence", "struct")):
                result["column_candidates"]["list_dict_cols"].append(col["name"])

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# CLI 入口
if __name__ == "__main__":
    import sys

    source_arg = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
    subset_arg = sys.argv[2] if len(sys.argv) > 2 else "main"
    split_arg = sys.argv[3] if len(sys.argv) > 3 else "train"
    output = inspect_dataset(source=source_arg, subset=subset_arg, split=split_arg)
    print(json.dumps(output, ensure_ascii=False, indent=2))
