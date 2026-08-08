"""filter_samples — 过滤/清洗工具。

确定性规则过滤：长度、去重、语言、空列检查。
Agent 决定阈值，脚本执行。无"智能"判断。
"""

import json
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any


def filter_samples(
    input_path: str,
    output_path: str,
    *,
    min_question_len: int = 0,
    max_question_len: int = 4096,
    min_answer_len: int = 0,
    max_answer_len: int = 8192,
    dedup_by: str = "question",
    drop_empty_cols: list[str] | None = None,
    lang_filter: str | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """确定性过滤 — Agent 决定阈值，脚本无脑执行。

    Args:
        input_path: 输入文件路径（JSON 数组或 JSONL）。
        dedup_by: "question" 按 question 去重 | "question+answer" | "hash"。
        drop_empty_cols: 哪些列为空就丢弃整行。
        lang_filter: "zh" 只保留中文 | "en" 只保留英文 | None 不过滤。
        max_samples: 最终保留的最大样本数。

    Returns:
        {"output_path": "...", "kept": 4500, "dropped": 320, "drop_reasons": {...}}
    """
    drop_reasons: dict[str, int] = OrderedDict()

    # ── 加载 ──
    input_p = Path(input_path)
    if not input_p.exists():
        return {"output_path": output_path, "kept": 0, "dropped": 0, "drop_reasons": {"input_not_found": 1}}

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

    kept: list[dict] = []
    seen_hashes: set[str] = set()
    total = len(rows)

    # ── 语言检测辅助 ──
    def _detect_lang(text: str) -> str:
        """基于 Unicode 范围的粗略语言检测。"""
        if not text:
            return "unknown"
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff")
        if cjk > len(text) * 0.15:
            return "zh"
        return "en"

    for row in rows:
        if not isinstance(row, dict):
            drop_reasons["not_dict"] = drop_reasons.get("not_dict", 0) + 1
            continue

        # ── 空列检查 ──
        if drop_empty_cols:
            has_empty = False
            for col in drop_empty_cols:
                val = row.get(col)
                if val is None or (isinstance(val, str) and not val.strip()):
                    has_empty = True
                    break
            if has_empty:
                drop_reasons["empty_col"] = drop_reasons.get("empty_col", 0) + 1
                continue

        # ── 提取字段 ──
        q_text = _extract_field(row, "input", "instruction", "question_text", "question")
        a_text = _extract_field(row, "output", "answer", "gold_answer", "response")

        # ── 长度过滤 ──
        if min_question_len > 0 and len(q_text) < min_question_len:
            drop_reasons["question_too_short"] = drop_reasons.get("question_too_short", 0) + 1
            continue
        if len(q_text) > max_question_len:
            drop_reasons["question_too_long"] = drop_reasons.get("question_too_long", 0) + 1
            continue
        if min_answer_len > 0 and len(a_text) < min_answer_len:
            drop_reasons["answer_too_short"] = drop_reasons.get("answer_too_short", 0) + 1
            continue
        if len(a_text) > max_answer_len:
            drop_reasons["answer_too_long"] = drop_reasons.get("answer_too_long", 0) + 1
            continue

        # ── 语言过滤 ──
        if lang_filter:
            lang = _detect_lang(q_text)
            if lang != lang_filter:
                drop_reasons["lang_mismatch"] = drop_reasons.get("lang_mismatch", 0) + 1
                continue

        # ── 去重 ──
        if dedup_by == "hash":
            key = hashlib.md5(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        elif dedup_by == "question+answer":
            key = hashlib.md5((q_text + a_text).encode()).hexdigest()
        else:
            key = hashlib.md5(q_text.encode()).hexdigest()

        if key in seen_hashes:
            drop_reasons["duplicate"] = drop_reasons.get("duplicate", 0) + 1
            continue
        seen_hashes.add(key)

        kept.append(row)

    # ── 截断 ──
    if max_samples and len(kept) > max_samples:
        drop_reasons["exceed_max_samples"] = len(kept) - max_samples
        kept = kept[:max_samples]

    # ── 写出 ──
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    output_p.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_path": str(output_p),
        "kept": len(kept),
        "dropped": total - len(kept),
        "total": total,
        "drop_reasons": dict(drop_reasons),
    }


def _extract_field(row: dict, *field_names: str) -> str:
    """从行中按优先级提取第一个存在且非空的字段值。

    遍历 field_names，返回首个非 None 值的字符串形式；全部不存在则返回 ""。
    """
    for name in field_names:
        val = row.get(name)
        if val is not None:
            return str(val)
    return ""


# CLI 入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="确定性过滤样本")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--min_question_len", type=int, default=0)
    parser.add_argument("--max_question_len", type=int, default=4096)
    parser.add_argument("--min_answer_len", type=int, default=0)
    parser.add_argument("--max_answer_len", type=int, default=8192)
    parser.add_argument("--dedup_by", default="question")
    parser.add_argument("--drop_empty_cols", nargs="*", default=None)
    parser.add_argument("--lang_filter", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    result = filter_samples(
        input_path=args.input_path,
        output_path=args.output_path,
        min_question_len=args.min_question_len,
        max_question_len=args.max_question_len,
        min_answer_len=args.min_answer_len,
        max_answer_len=args.max_answer_len,
        dedup_by=args.dedup_by,
        drop_empty_cols=args.drop_empty_cols,
        lang_filter=args.lang_filter,
        max_samples=args.max_samples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
