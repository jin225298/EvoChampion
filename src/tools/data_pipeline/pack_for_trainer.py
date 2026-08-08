"""pack_for_trainer — 打包工具。

将构建好的训练/测试数据打包为 LLaMA-Factory 可直接消费的格式：
- 写 dataset_info.json（注册数据集）
- 写 YAML 配置覆盖片段（可选）
- 返回 DatasetBundle 兼容字典
"""

import json
import shutil
from pathlib import Path
from typing import Any


def _ensure_alpaca_optional_columns(path: Path, column_mapping: dict | None) -> None:
    if not column_mapping:
        return
    system_col = column_mapping.get("system")
    history_col = column_mapping.get("history")
    if not system_col and not history_col:
        return
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(rows, list):
        return
    changed = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if system_col and system_col not in row:
            row[system_col] = ""
            changed = True
        if history_col and history_col not in row:
            row[history_col] = []
            changed = True
    if changed:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def pack_for_trainer(
    train_path: str,
    output_dir: str,
    *,
    test_path: str | None = None,
    cotest_path: str | None = None,
    probe_path: str | None = None,
    lf_val_path: str | None = None,
    dataset_name: str = "round_N_train",
    output_format: str = "alpaca",
    column_mapping: dict | None = None,
) -> dict[str, Any]:
    """将训练数据打包为 LLaMA-Factory 格式。

    产出：
    - output_dir/train.json（从 train_path 复制）
    - output_dir/test.json（如果提供）
    - output_dir/dataset_info.json

    Args:
        train_path: 训练数据 JSON 文件路径。
        output_dir: 输出目录。
        test_path: 测试数据路径，可选。
        cotest_path: 共评估数据路径，可选。
        probe_path: 探针数据路径，可选。
        lf_val_path: LLaMA-Factory 训练时 eval 数据路径，可选。
        dataset_name: LLaMA-Factory 数据集注册名。
        output_format: "alpaca" | "sharegpt"。
        column_mapping: 列映射（{"prompt": "instruction", "query": "input", "response": "output"}）。

    Returns:
        兼容现有 DatasetBundlePayload 的字典结构。
    """
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    # ── 默认列映射 ──
    if column_mapping is None:
        if output_format == "alpaca":
            column_mapping = {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            }
        elif output_format == "sharegpt":
            column_mapping = {
                "messages": "conversations",
            }

    # ── 复制/移动训练文件 ──
    train_src = Path(train_path)
    train_dst = output_p / "train.json"
    if train_src != train_dst:
        shutil.copy2(train_src, train_dst)
    if output_format == "alpaca":
        _ensure_alpaca_optional_columns(train_dst, column_mapping)

    # ── 复制测试文件 ──
    if test_path:
        test_src = Path(test_path)
        test_dst = output_p / "test.json"
        if test_src != test_dst:
            shutil.copy2(test_src, test_dst)

    # ── 复制 cotest 文件 ──
    if cotest_path:
        cotest_src = Path(cotest_path)
        cotest_dst = output_p / "cotest.json"
        if cotest_src != cotest_dst:
            shutil.copy2(cotest_src, cotest_dst)

    # ── 复制 probe 文件 ──
    if probe_path:
        probe_src = Path(probe_path)
        probe_dst = output_p / "probe.json"
        if probe_src != probe_dst:
            shutil.copy2(probe_src, probe_dst)

    # ── 复制 LLaMA-Factory eval 文件 ──
    lf_val_dataset_name = ""
    if lf_val_path:
        lf_val_src = Path(lf_val_path)
        lf_val_dst = output_p / "lf_val.json"
        if lf_val_src != lf_val_dst:
            shutil.copy2(lf_val_src, lf_val_dst)
        if output_format == "alpaca":
            _ensure_alpaca_optional_columns(lf_val_dst, column_mapping)
        lf_val_dataset_name = f"{dataset_name}_lf_val"

    # ── 写 dataset_info.json ──
    dataset_info: dict[str, dict] = {
        dataset_name: {
            "file_name": "train.json",
            "formatting": output_format,
            "columns": column_mapping or {},
        }
    }
    if lf_val_dataset_name:
        dataset_info[lf_val_dataset_name] = {
            "file_name": "lf_val.json",
            "formatting": output_format,
            "columns": column_mapping or {},
        }
    info_path = output_p / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 统计训练样本数 ──
    train_count = 0
    try:
        train_data = json.loads(train_dst.read_text(encoding="utf-8"))
        if isinstance(train_data, list):
            train_count = len(train_data)
    except Exception:
        pass

    # ── 组装返回（兼容 DatasetBundlePayload） ──
    return {
        "train_path": str(train_dst),
        "test_path": str(output_p / "test.json") if test_path else "",
        "cotest_path": str(output_p / "cotest.json") if cotest_path else "",
        "probe_path": str(output_p / "probe.json") if probe_path else "",
        "lf_val_path": str(output_p / "lf_val.json") if lf_val_path else "",
        "dataset_info_path": str(info_path),
        "dataset_dir": str(output_p),
        "train_dataset_name": dataset_name,
        "lf_val_dataset_name": lf_val_dataset_name,
        "train_count": train_count,
        "format": output_format,
    }


# CLI 入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="打包为 LLaMA-Factory 格式")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--cotest_path", default=None)
    parser.add_argument("--probe_path", default=None)
    parser.add_argument("--lf_val_path", default=None)
    parser.add_argument("--dataset_name", default="round_N_train")
    parser.add_argument("--output_format", default="alpaca")
    args = parser.parse_args()

    result = pack_for_trainer(
        train_path=args.train_path,
        output_dir=args.output_dir,
        test_path=args.test_path,
        cotest_path=args.cotest_path,
        probe_path=args.probe_path,
        lf_val_path=args.lf_val_path,
        dataset_name=args.dataset_name,
        output_format=args.output_format,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
