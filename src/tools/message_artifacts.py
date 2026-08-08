"""
消息产物（Message Artifact）落地/读取工具

核心职责：
1. 将大体积数据（题目列表、搜索结果等）写入 session 目录下的 JSON 文件，避免在 LangGraph
   节点间直接传递大消息体
2. 返回轻量级的 DataArtifactRef 引用，节点间只传递引用地址而非数据本身
3. 消费者节点通过引用路径读回原始数据

数据流：
  节点 A → write_json_artifact(data) → DataArtifactRef → RoutedMessage(仅含 ref)
  节点 B → load_json_artifact(ref) → 完整数据

使用场景：
  所有需要在节点间传递大数据负载的地方（classifier、screening_entry、filter 等），
  避免 State 对象体积膨胀导致序列化/传递性能下降。
"""

import json
import uuid
from pathlib import Path
from typing import Any

from config.settings import get_session_dir
from src.models.messages import DataArtifactRef


# Threshold for compact vs. pretty-printed JSON (in bytes of serialized output)
_COMPACT_JSON_SIZE_THRESHOLD: int = 1024 * 100  # 100KB


def _jsonable(item: Any) -> Any:
    """递归将 Pydantic 对象或复杂嵌套结构转换为纯 JSON 可序列化格式。

    处理规则：
    - Pydantic 对象 → model_dump(mode="json")
    - dict → 递归转换每个 value
    - list → 递归转换每个元素
    - 其他类型 → 原样返回

    Args:
        item: 任意 Python 对象

    Returns:
        JSON 可序列化的纯 Python 结构（dict/list/str/int/float/bool/None）
    """
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return {key: _jsonable(value) for key, value in item.items()}
    if isinstance(item, list):
        return [_jsonable(value) for value in item]
    return item


def write_json_artifact(
    trace_id: str,
    round_id: int,
    producer: str,
    name: str,
    data: Any,
    kind: str = "json",
) -> DataArtifactRef:
    """将大体积数据落地为 session 目录下的 JSON 文件，返回可传递的引用地址。

    文件路径模式：{session_dir}/message_artifacts/round_{round_id}/{artifact_id}.json

    For large payloads (>100KB), uses compact JSON (no indentation) to reduce
    disk usage and I/O time. Small payloads keep pretty-printed indentation
    for readability. Artifact metadata (content_type, schema_version) is
    included when supported by DataArtifactRef.

    Args:
        trace_id: 当前会话标识，用于定位 session 目录
        round_id: 当前训练轮次
        producer: 生产者节点名（如 "classifier"、"screening_entry"）
        name: 产物逻辑名（如 "classified_questions"）
        data: 任意待序列化数据
        kind: 产物类型标记，默认 "json"

    Returns:
        DataArtifactRef，包含 artifact_id、local_path、count、kind，
        调用方应将此引用放入 RoutedMessage.payload 中传递
    """
    artifact_id = f"r{round_id}_{producer}_{name}_{uuid.uuid4().hex[:10]}"
    artifact_dir = get_session_dir(trace_id) / "message_artifacts" / f"round_{round_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{artifact_id}.json"
    payload = _jsonable(data)

    # Choose JSON format based on payload size
    compact_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(compact_payload) > _COMPACT_JSON_SIZE_THRESHOLD:
        # Large payload: use compact format
        raw = compact_payload
    else:
        # Small payload: use pretty-printed format for readability
        raw = json.dumps(payload, ensure_ascii=False, indent=2)

    path.write_text(raw, encoding="utf-8")
    count = len(payload) if isinstance(payload, list) else 1
    return DataArtifactRef(
        artifact_id=artifact_id,
        local_path=str(path),
        count=count,
        kind=kind,
    )


def load_json_artifact(ref: DataArtifactRef | dict | str | None) -> Any:
    """按引用读取已落地的产物数据；引用为空时返回 None。

    兼容三种引用形式：
    - DataArtifactRef 对象 → 取 local_path
    - dict（消息反序列化后无类型信息） → 取 "local_path" 键
    - str → 直接作为路径

    Args:
        ref: 产物引用，可以是 DataArtifactRef、dict 或路径字符串

    Returns:
        反序列化后的数据，或 None（ref 为空 / 文件不存在）
    """
    if ref is None:
        return None
    if isinstance(ref, DataArtifactRef):
        path = ref.local_path
    elif isinstance(ref, dict):
        path = str(ref.get("local_path") or "")
    else:
        path = str(ref)
    if not path:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists() or artifact_path.is_dir():
        return None
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def load_payload_list(inline_items: list[Any], ref: DataArtifactRef | dict | str | None) -> list[dict]:
    """从内联数据或产物引用中统一获取题目/结果的 dict 列表。

    这是 classifier、screening_entry、filter 等节点的统一数据消费入口。
    优先使用内联数据（inline_items），为空时从 ref 文件读取。

    Args:
        inline_items: 可能内联在消息体中的题目/结果列表
        ref: 产物引用（内联数据为空时的 fallback 数据源）

    Returns:
        标准化的 list[dict]，无效条目已被过滤
    """
    if inline_items:
        return [_jsonable(item) for item in inline_items if item is not None]
    loaded = load_json_artifact(ref)
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)]
    return []
