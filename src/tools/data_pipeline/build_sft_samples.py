"""build_sft_samples — 样本构建工具。

按 Agent 指定的列选择、模板、输出格式，将原始数据拼成训练样本。
纯格式化引擎——不做任何"智能"判断，参数全部由上层传入。

关键设计：
- prompt_template 是 Agent 生成的字符串，含 {col_name} 占位符
- 脚本只做 template.format(**row)
- response_template 可选，用于多列答案拼接
"""

import json
from pathlib import Path
from typing import Any


def build_sft_samples(
    input_path: str,
    question_cols: list[str],
    answer_cols: list[str],
    prompt_template: str,
    output_path: str,
    *,
    response_template: str | None = None,
    output_format: str = "alpaca",
    extra_context_cols: list[str] | None = None,
    metadata_cols: list[str] | None = None,
) -> dict[str, Any]:
    """按 Agent 决策将原始数据拼成 SFT 训练样本。

    Args:
        input_path: 原始数据文件路径（JSON 数组或 JSONL）。
        question_cols: 组成题目的列名列表。
        answer_cols: 组成答案的列名列表（按顺序拼接，用换行分隔）。
        prompt_template: 提示词模板，如 "请解答下面的{domain}题：\n{problem}"。
        output_path: 输出文件路径。
        response_template: 答案模板，如 "{solution}\n最终答案：{answer}"，可选。
        output_format: "alpaca" | "sharegpt" | "raw"。
        extra_context_cols: 额外上下文列（如 "system"）。
        metadata_cols: 透传但不参与训练的列。

    Returns:
        {"output_path": "...", "sample_count": 4500, "skipped": 0, "format": "alpaca"}
    """
    # ── 加载原始数据 ──
    input_p = Path(input_path)
    if not input_p.exists():
        return {"output_path": output_path, "sample_count": 0, "skipped": 0, "format": output_format, "error": f"input not found: {input_path}"}

    raw_text = input_p.read_text(encoding="utf-8", errors="replace").strip()

    if raw_text.startswith("["):
        rows: list[dict] = json.loads(raw_text)
    else:
        rows = []
        for line in raw_text.split("\n"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not isinstance(rows, list):
        return {"output_path": output_path, "sample_count": 0, "skipped": 0, "format": output_format, "error": "input is not a list"}

    # ── 构建样本 ──
    samples: list[dict] = []
    skipped = 0

    for row in rows:
        if not isinstance(row, dict):
            skipped += 1
            continue

        # 提取题目文本
        question_parts = []
        for col in question_cols:
            val = row.get(col)
            if val is not None:
                question_parts.append(str(val))
        question_text = "\n".join(question_parts).strip()

        # 提取答案文本
        answer_parts = []
        for col in answer_cols:
            val = row.get(col)
            if val is not None:
                answer_parts.append(str(val))
        answer_text = "\n".join(answer_parts).strip()

        if not question_text or not answer_text:
            skipped += 1
            continue

        # 应用模板
        template_vars: dict[str, str] = {}
        for col in question_cols + answer_cols:
            val = row.get(col)
            template_vars[col] = str(val) if val is not None else ""

        instruction = prompt_template
        try:
            instruction = prompt_template.format(**template_vars)
        except KeyError:
            # 模板中引用了不存在的列 → 用原始模板
            pass

        response = answer_text
        if response_template:
            try:
                response = response_template.format(**template_vars)
            except KeyError:
                pass

        # 构建元数据
        metadata: dict[str, Any] = {}
        if metadata_cols:
            for col in metadata_cols:
                if col in row:
                    metadata[col] = row[col]

        if output_format == "alpaca":
            sample: dict[str, Any] = {
                "instruction": instruction,
                "input": question_text if question_text != instruction else "",
                "output": response,
            }
        elif output_format == "sharegpt":
            sample = {
                "conversations": [
                    {"from": "human", "value": instruction + "\n" + question_text},
                    {"from": "gpt", "value": response},
                ]
            }
        else:
            sample = {
                "text": instruction + "\n" + question_text + "\n" + response,
            }

        if extra_context_cols and output_format == "alpaca":
            for col in extra_context_cols:
                if col in row:
                    sample["system"] = str(row[col])

        if metadata:
            sample["metadata"] = metadata

        samples.append(sample)

    # ── 写出 ──
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    output_p.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_path": str(output_p),
        "sample_count": len(samples),
        "skipped": skipped,
        "format": output_format,
    }


# CLI 入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按 Agent 决策构建 SFT 样本")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--question_cols", nargs="+", required=True)
    parser.add_argument("--answer_cols", nargs="+", required=True)
    parser.add_argument("--prompt_template", required=True)
    parser.add_argument("--response_template", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_format", default="alpaca")
    parser.add_argument("--extra_context_cols", nargs="*", default=None)
    parser.add_argument("--metadata_cols", nargs="*", default=None)
    args = parser.parse_args()

    result = build_sft_samples(
        input_path=args.input_path,
        question_cols=args.question_cols,
        answer_cols=args.answer_cols,
        prompt_template=args.prompt_template,
        response_template=args.response_template,
        output_path=args.output_path,
        output_format=args.output_format,
        extra_context_cols=args.extra_context_cols,
        metadata_cols=args.metadata_cols,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
