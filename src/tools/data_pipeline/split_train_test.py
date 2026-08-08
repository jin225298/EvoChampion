"""split_train_test — 切分工具。

支持三种策略：
- use_existing_split: 数据自带 split 列 → 按列值分组，不重新随机切
- ratio: 按比例随机切分
- holdout_by_key: 按某列的每个类别留一点做 holdout
"""

import json
import random
from pathlib import Path
from typing import Any


def split_train_test(
    input_path: str,
    output_dir: str,
    *,
    strategy: str = "ratio",
    test_ratio: float = 0.15,
    existing_split_col: str | None = None,
    train_label: str = "train",
    test_label: str = "test",
    holdout_key_col: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """切分数据集为 train/test。

    Args:
        input_path: 输入 JSON 数组文件路径。
        output_dir: 输出目录。
        strategy: "use_existing_split" | "ratio" | "holdout_by_key"。
        test_ratio: ratio 策略的测试集比例。
        existing_split_col: 数据中表示 split 的列名（如 "split"）。
        train_label: split 列中表示训练的值（如 "train"）。
        test_label: split 列中表示测试的值（如 "test"）。
        holdout_key_col: holdout_by_key 策略按哪列分层（如 "category"）。
        seed: 随机种子。

    Returns:
        {
            "train_path": "...",
            "test_path": "...",
            "train_count": 4000,
            "test_count": 500,
            "strategy": "ratio",
        }
    """
    # ── 加载 ──
    input_p = Path(input_path)
    if not input_p.exists():
        return {"error": f"input not found: {input_path}"}

    rows: list[dict] = json.loads(input_p.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(rows, list):
        return {"error": "input is not a JSON array"}

    rng = random.Random(seed)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    test_rows: list[dict] = []

    # ── 策略 1: 使用数据自带的 split ──
    if strategy == "use_existing_split" and existing_split_col:
        for row in rows:
            split_val = str(row.get(existing_split_col, "")).lower()
            if split_val == train_label.lower():
                train_rows.append(row)
            elif split_val == test_label.lower():
                test_rows.append(row)
        if not test_rows:
            # 如果自带 split 列但没 test 值，按 ratio 兜底
            rng.shuffle(rows)
            split_idx = max(1, int(len(rows) * (1 - test_ratio)))
            train_rows = rows[:split_idx]
            test_rows = rows[split_idx:]

    # ── 策略 2: 按比例随机切 ──
    elif strategy == "ratio":
        shuffled = list(rows)
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - test_ratio)))
        train_rows = shuffled[:split_idx]
        test_rows = shuffled[split_idx:]

    # ── 策略 3: 按 key 分层 holdout ──
    elif strategy == "holdout_by_key" and holdout_key_col:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            key = str(row.get(holdout_key_col, "unknown"))
            groups.setdefault(key, []).append(row)

        for group in groups.values():
            rng.shuffle(group)
            # 每类至少留 1 个做 holdout，按比例至少留 10%
            n_holdout = max(1, int(len(group) * max(test_ratio, 0.1)))
            train_rows.extend(group[n_holdout:])
            test_rows.extend(group[:n_holdout])

        rng.shuffle(train_rows)
        rng.shuffle(test_rows)

    else:
        return {"error": f"unknown strategy: {strategy}"}

    # ── 写出 ──
    train_path = output_p / "train.json"
    test_path = output_p / "test.json"

    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "strategy": strategy,
    }


# CLI 入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="切分数据集")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strategy", default="ratio", choices=["use_existing_split", "ratio", "holdout_by_key"])
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--existing_split_col", default=None)
    parser.add_argument("--train_label", default="train")
    parser.add_argument("--test_label", default="test")
    parser.add_argument("--holdout_key_col", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = split_train_test(
        input_path=args.input_path,
        output_dir=args.output_dir,
        strategy=args.strategy,
        test_ratio=args.test_ratio,
        existing_split_col=args.existing_split_col,
        train_label=args.train_label,
        test_label=args.test_label,
        holdout_key_col=args.holdout_key_col,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
