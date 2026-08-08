"""decontaminate — 防污染工具。

确定性检测候选训练数据与评估 benchmark 的重叠度。
必须确定性、可审计、可复现——不能让 Agent 自己判断。

支持三种检测方法：
- exact_match: 精确字符串匹配
- ngram_overlap: n-gram 重叠率（推荐）
- minhash: MinHash 近似去重
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def decontaminate(
    candidate_path: str,
    benchmark_paths: list[str],
    output_clean_path: str,
    output_report_path: str,
    *,
    method: str = "ngram_overlap",
    ngram_n: int = 13,
    threshold: float = 0.8,
    match_on: str = "question",
) -> dict[str, Any]:
    """检测并剔除与 benchmark 重叠的训练样本。

    Args:
        candidate_path: 候选训练数据文件。
        benchmark_paths: benchmark 文件列表。
        output_clean_path: 清洗后输出路径。
        output_report_path: 污染报告输出路径。
        method: "exact_match" | "ngram_overlap" | "minhash"。
        ngram_n: n-gram 的 n 值（仅 ngram_overlap 方法）。
        threshold: 重叠率阈值，超过即视为污染。
        match_on: 对哪一列做匹配（"question" | "question+answer"）。

    Returns:
        {
            "clean_path": "...",
            "report_path": "...",
            "candidate_total": 4500,
            "clean_count": 4300,
            "contaminated_count": 200,
            "method": "ngram_overlap",
        }
    """
    # ── 加载候选数据 ──
    candidate_path_o = Path(candidate_path)
    if not candidate_path_o.exists():
        return {"error": f"candidate not found: {candidate_path}"}
    candidates = _load_json_array(candidate_path_o)

    # ── 加载 benchmark 数据 ──
    benchmark_texts: list[str] = []
    for bp in benchmark_paths:
        bp_path = Path(bp)
        if not bp_path.exists():
            continue
        items = _load_json_array(bp_path)
        for item in items:
            text = _extract_match_text(item, match_on)
            if text:
                benchmark_texts.append(text)

    if not benchmark_texts:
        _write_json(candidates, Path(output_clean_path))
        _write_json({"summary": {"note": "no benchmark data to compare against"}}, Path(output_report_path))
        return {
            "clean_path": output_clean_path,
            "report_path": output_report_path,
            "candidate_total": len(candidates),
            "clean_count": len(candidates),
            "contaminated_count": 0,
            "method": method,
        }

    contaminated: list[dict] = []
    clean: list[dict] = []

    # ── 执行检测 ──
    if method == "exact_match":
        _benchmark_set = {_normalize(t) for t in benchmark_texts}
        for item in candidates:
            text = _normalize(_extract_match_text(item, match_on))
            if text in _benchmark_set:
                contaminated.append(item)
            else:
                clean.append(item)

    elif method == "ngram_overlap":
        benchmark_ngram_sets = [_get_ngrams(_normalize(t), ngram_n) for t in benchmark_texts]
        for item in candidates:
            text = _normalize(_extract_match_text(item, match_on))
            candidate_ngrams = _get_ngrams(text, ngram_n)
            if not candidate_ngrams:
                clean.append(item)
                continue
            max_overlap = 0.0
            for bench_ngrams in benchmark_ngram_sets:
                if not bench_ngrams:
                    continue
                intersection = candidate_ngrams & bench_ngrams
                overlap = len(intersection) / len(candidate_ngrams)
                max_overlap = max(max_overlap, overlap)
            if max_overlap >= threshold:
                contaminated.append({
                    **item,
                    "_decontam_overlap_ratio": round(max_overlap, 4),
                })
            else:
                clean.append(item)

    elif method == "minhash":
        # 简化版 MinHash：用 n-gram 集合的哈希值做近似比较
        _benchmark_sigs = {_minhash_signature(t, ngram_n) for t in benchmark_texts}
        for item in candidates:
            text = _normalize(_extract_match_text(item, match_on))
            sig = _minhash_signature(text, ngram_n)
            # 简单判断：签名是否在 benchmark 中出现
            if sig in _benchmark_sigs:
                contaminated.append(item)
            else:
                clean.append(item)

    else:
        return {"error": f"unknown method: {method}"}

    # ── 写出 ──
    _write_json(clean, Path(output_clean_path))

    report = {
        "contaminated": [
            {
                "candidate_index": i,
                "candidate_sample": _truncate_sample(contaminated[i], match_on),
                "overlap_ratio": contaminated[i].get("_decontam_overlap_ratio"),
                "action": "removed",
            }
            for i in range(min(len(contaminated), 50))
        ],
        "summary": {
            "candidate_total": len(candidates),
            "clean": len(clean),
            "contaminated": len(contaminated),
            "method": method,
            "ngram_n": ngram_n,
            "threshold": threshold,
            "benchmark_files": benchmark_paths,
        },
    }
    _write_json(report, Path(output_report_path))

    return {
        "clean_path": output_clean_path,
        "report_path": output_report_path,
        "candidate_total": len(candidates),
        "clean_count": len(clean),
        "contaminated_count": len(contaminated),
        "method": method,
    }


# ── 辅助函数 ──

def _load_json_array(path: Path) -> list[dict]:
    """从文件加载 JSON 数组（兼容 JSONL 格式）。

    若文件以 '[' 开头，按标准 JSON 数组解析；否则逐行解析为 JSONL。
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if text.startswith("["):
        return json.loads(text)
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_json(data: Any, path: Path) -> None:
    """将数据写入 JSON 文件，自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_match_text(item: dict, match_on: str) -> str:
    """从数据行中提取用于匹配的文本字段。

    match_on="question+answer" 时拼接题目和答案；否则只提取题目文本。
    兼容多种字段名（input/question_text/question, output/gold_answer/answer）。
    """
    if match_on == "question+answer":
        question = item.get("input") or item.get("question_text") or item.get("question") or ""
        answer = item.get("output") or item.get("gold_answer") or item.get("answer") or ""
        return f"{question}\n{answer}"
    # 默认只取 question
    return str(
        item.get("input")
        or item.get("question_text")
        or item.get("question")
        or item.get("instruction", "")
    )


def _normalize(text: str) -> str:
    """归一化文本：小写、合并空白、去标点，用于模糊匹配。"""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _get_ngrams(text: str, n: int) -> set[str]:
    """生成字符级 n-gram 集合，用于计算文本重叠率。"""
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _minhash_signature(text: str, ngram_n: int = 13) -> int:
    """生成 MinHash 签名：取所有 n-gram 哈希值中的最小值。

    用于近似去重：签名相同的文本视为近似重复。
    """
    ngrams = _get_ngrams(_normalize(text), ngram_n)
    if not ngrams:
        return 0
    return min(hash(ng) for ng in ngrams)


def _truncate_sample(item: dict, match_on: str) -> str:
    """截取匹配文本的前 200 字符，用于污染报告中的样本展示。"""
    text = _extract_match_text(item, match_on)
    return text[:200] + "..." if len(text) > 200 else text


# CLI 入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="防污染检测")
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--benchmark_paths", nargs="+", required=True)
    parser.add_argument("--output_clean_path", required=True)
    parser.add_argument("--output_report_path", required=True)
    parser.add_argument("--method", default="ngram_overlap", choices=["exact_match", "ngram_overlap", "minhash"])
    parser.add_argument("--ngram_n", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--match_on", default="question")
    args = parser.parse_args()

    result = decontaminate(
        candidate_path=args.candidate_path,
        benchmark_paths=args.benchmark_paths,
        output_clean_path=args.output_clean_path,
        output_report_path=args.output_report_path,
        method=args.method,
        ngram_n=args.ngram_n,
        threshold=args.threshold,
        match_on=args.match_on,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
