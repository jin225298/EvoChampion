"""
搜索 DAG（Search DAG）—— MCTS 搜索树的有向无环图实现

核心职责：
1. 将每轮训练-评估的结果建模为 DAG 节点，策略决策建模为有向边，
   形成可探索、可回溯的搜索空间
2. 每个节点记录本轮评估指标（accuracy / forgetting_score / stability_score）
   和 MCTS 估值（value_estimate / visit_count）
3. 支持增量均值回传（backpropagate_value）：叶子节点的 reward 沿祖先链向上传播，
   更新所有祖先的 value_estimate，实现 MCTS 的价值估计
4. 持久化到 session 目录，支持跨轮次恢复

MCTS over DAG 架构：
  root_node (bootstrap 创建)
      │
      ├── node_r1 (第 1 轮策略 A)      ← edge: action="promote", reward=0.72
      │       │
      │       ├── node_r2a (第 2 轮继承)  ← edge: action="promote", reward=0.68
      │       └── node_r2b (第 2 轮回退)  ← edge: action="prune",    reward=0.65
      │
      └── node_r1b (第 1 轮策略 B)     ← edge: action="explore", reward=0.70

  每轮结束时 backpropagate_value 将 reward 回传到根节点。

使用场景：
  bootstrap → create_root_node / load_search_dag → 初始化或恢复 DAG
  strategy_policy → add_search_node / add_search_edge / backpropagate_value → 扩展 DAG
  strategy_inspector → save_search_dag → 持久化
"""

import json
from pathlib import Path
from uuid import uuid4

from src.models.messages import SearchDAGEdgePayload, SearchDAGNodePayload


def create_root_node(trace_id: str) -> SearchDAGNodePayload:
    """创建 DAG 根节点。

    根节点无父节点、无评估指标，仅作为搜索树的起点。
    trace_id 参数保留用于接口一致性，当前实现不依赖它。

    Args:
        trace_id: 会话标识

    Returns:
        SearchDAGNodePayload，parent_node_ids 为空列表，visit_count=0
    """
    return SearchDAGNodePayload(
        node_id=f"node_{uuid4().hex[:8]}",
        parent_node_ids=[],
        visit_count=0,
    )


def add_search_node(
    parent_node_id: str,
    accuracy: float | None = None,
    forgetting_score: float | None = None,
    stability_score: float | None = None,
    value_estimate: float | None = None,
    visit_count: int = 0,
    metrics: dict | None = None,
) -> SearchDAGNodePayload:
    """创建搜索节点，作为某个已有节点的子节点。

    一个新节点代表一次训练-评估迭代的结果。评估指标在三重 gate
    通过后由 evaluator 写入，MCTS 估值由 backpropagate_value 更新。

    Args:
        parent_node_id: 父节点 ID
        accuracy: 本轮探针集准确率
        forgetting_score: 旧能力遗忘分（越低越好）
        stability_score: 稳定性分
        value_estimate: MCTS 价值估计（初始为 None，回传后更新）
        visit_count: 访问次数
        metrics: 额外指标字典

    Returns:
        SearchDAGNodePayload，parent_node_ids 为 [parent_node_id]
    """
    return SearchDAGNodePayload(
        node_id=f"node_{uuid4().hex[:8]}",
        parent_node_ids=[parent_node_id],
        accuracy=accuracy,
        forgetting_score=forgetting_score,
        stability_score=stability_score,
        visit_count=visit_count,
        value_estimate=value_estimate,
        metrics=metrics or {},
    )


def add_search_edge(
    from_node_id: str,
    to_node_id: str,
    action_type: str,
    action_summary: str,
    round_id: int | None = None,
    decision: str | None = None,
    reward: float | None = None,
    action_metadata: dict | None = None,
) -> SearchDAGEdgePayload:
    """创建 DAG 边，记录父子节点间的策略动作。

    action_type 通常为 add_dataset / reuse_buffer_data / keep_branch / mcts_action，
    由 strategy_inspector 根据 gate 和 MCTS 结果决策。

    Args:
        from_node_id: 父节点 ID
        to_node_id: 子节点 ID
        action_type: 动作类型（add_dataset / reuse_buffer_data / keep_branch / mcts_action）
        action_summary: 动作摘要描述
        round_id: 关联的训练轮次
        decision: 决策结论
        reward: 该动作获得的 reward（用于 MCTS 回传）
        action_metadata: 额外动作元数据

    Returns:
        SearchDAGEdgePayload
    """
    return SearchDAGEdgePayload(
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        action_type=action_type,
        action_summary=action_summary,
        round_id=round_id,
        decision=decision,
        reward=reward,
        action_metadata=action_metadata or {},
    )


def save_search_dag(
    nodes: list[dict],
    edges: list[dict],
    current_node_id: str,
    session_dir: Path,
) -> None:
    """持久化整个 DAG（节点 + 边 + 当前节点）到 session 目录。

    Args:
        nodes: DAG 节点列表
        edges: DAG 边列表
        current_node_id: 当前活跃节点 ID（最新轮次的节点）
        session_dir: 当前 session 目录
    """
    dag_path = session_dir / "search_dag.json"
    dag_data = {
        "current_node_id": current_node_id,
        "nodes": nodes,
        "edges": edges,
    }
    with open(dag_path, "w", encoding="utf-8") as f:
        json.dump(dag_data, f, ensure_ascii=False, indent=2)


def load_search_dag(session_dir: Path) -> tuple[list[dict], list[dict], str]:
    """从 session 目录加载 DAG。

    Args:
        session_dir: 当前 session 目录

    Returns:
        (nodes, edges, current_node_id) 三元组，不存在时返回空
    """
    dag_path = session_dir / "search_dag.json"
    if not dag_path.exists():
        return [], [], ""
    with open(dag_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("nodes", []), data.get("edges", []), data.get("current_node_id", "")


def update_node_metrics(
    nodes: list[dict],
    node_id: str,
    accuracy: float | None = None,
    forgetting_score: float | None = None,
    stability_score: float | None = None,
) -> list[dict]:
    """更新节点评估指标并递增 visit_count。

    在三重 gate 通过后由策略层调用，将 evaluator 的评估结果写入对应节点。

    Args:
        nodes: 当前 DAG 节点列表
        node_id: 待更新的节点 ID
        accuracy: 探针集准确率
        forgetting_score: 遗忘分
        stability_score: 稳定性分

    Returns:
        更新后的节点列表（原始列表被原地修改）
    """
    updated = []
    for node in nodes:
        if node.get("node_id") == node_id:
            if accuracy is not None:
                node["accuracy"] = accuracy
            if forgetting_score is not None:
                node["forgetting_score"] = forgetting_score
            if stability_score is not None:
                node["stability_score"] = stability_score
            node["visit_count"] = node.get("visit_count", 0) + 1
        updated.append(node)
    return updated


def backpropagate_value(
    nodes: list[dict],
    edges: list[dict],
    leaf_node_id: str,
    reward: float,
) -> list[dict]:
    """MCTS 增量均值回传：将叶子节点的 reward 沿祖先链向上传播。

    使用增量均值公式更新每个祖先的 value_estimate：
      new_value = old_value + (reward - old_value) / visit_count

    这等价于维护所有历史 reward 的算术平均值，无需存储全部历史值。

    Args:
        nodes: 当前 DAG 节点列表
        edges: 当前 DAG 边列表
        leaf_node_id: 叶子节点 ID（回传起点）
        reward: 本轮获得的 reward 值

    Returns:
        更新后的节点列表（祖先节点的 value_estimate 和 visit_count 已更新）
    """
    # 构建子→父映射
    parent_by_child = {
        edge.get("to_node_id"): edge.get("from_node_id")
        for edge in edges
        if edge.get("to_node_id") and edge.get("from_node_id")
    }
    node_by_id = {node.get("node_id"): dict(node) for node in nodes}

    current_id = leaf_node_id
    visited: set[str] = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        node = node_by_id.get(current_id)
        if node is None:
            break

        visits = int(node.get("visit_count", 0) or 0)
        old_value = float(node.get("value_estimate", 0.0) or 0.0)
        new_visits = visits + 1
        node["visit_count"] = new_visits
        # 增量均值：new = old + (reward - old) / n
        node["value_estimate"] = old_value + (reward - old_value) / new_visits
        node_by_id[current_id] = node

        # 沿父节点链继续向上
        current_id = parent_by_child.get(current_id, "")

    return [node_by_id.get(node.get("node_id"), node) for node in nodes]
