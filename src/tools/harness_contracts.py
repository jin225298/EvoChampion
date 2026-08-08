from src.models.messages import (
    AgentName,
    MessageType,
    RoutedMessage,
)

# =============================================================================
# 消息路由契约：定义合法的 (消息类型, 发送者, 接收者) 三元组
# 整个系统的控制流由这 18 条规则约束，router 节点逐条校验
# =============================================================================
CONTRACT_RULES = [
    (MessageType.GOAL_REQUEST, AgentName.SYSTEM, AgentName.TEACHER),
    (MessageType.SEARCH_REQUEST, AgentName.TEACHER, AgentName.PARAMETER_MASTER),
    (MessageType.SEARCH_REQUEST, AgentName.PARAMETER_MASTER, AgentName.SEARCHER),
    (MessageType.SEARCH_REQUEST, AgentName.SCREENING_ENTRY, AgentName.SEARCHER),
    (MessageType.SEARCH_REQUEST, AgentName.SEARCHER, AgentName.HF_SEARCH_TOOL),
    (MessageType.SEARCH_RESULT, AgentName.HF_SEARCH_TOOL, AgentName.DATASET_REVIEWER),
    (MessageType.SEARCH_RESULT, AgentName.DATASET_REVIEWER, AgentName.DATASET_REVIEWER),
    (MessageType.SEARCH_RESULT, AgentName.DATASET_REVIEWER, AgentName.SCREENING_ENTRY),
    (MessageType.DATASET_SCHEMA_REQUEST, AgentName.SCREENING_ENTRY, AgentName.DATASET_SCHEMA_AGENT),
    (MessageType.DATASET_SCHEMA_RESULT, AgentName.DATASET_SCHEMA_AGENT, AgentName.SCREENING_ENTRY),
    (MessageType.MATERIALIZED_DATASET, AgentName.SCREENING_ENTRY, AgentName.FILTER),
    (MessageType.ROLLOUT_REQUEST, AgentName.FILTER, AgentName.ROLLOUT_DISPATCHER),
    (MessageType.ROLLOUT_TASK, AgentName.ROLLOUT_DISPATCHER, AgentName.ROLLOUT_WORKER),
    (MessageType.ROLLOUT_TASK_RESULT, AgentName.ROLLOUT_WORKER, AgentName.ROLLOUT_AGGREGATOR),
    (MessageType.ROLLOUT_RESULT, AgentName.ROLLOUT_AGGREGATOR, AgentName.FILTER),
    (MessageType.ROLLOUT_RESULT, AgentName.ROLLOUT_AGGREGATOR, AgentName.TEACHER),
    (MessageType.FILTER_RESULT, AgentName.FILTER, AgentName.CLASSIFIER),
    (MessageType.CLASSIFICATION_RESULT, AgentName.CLASSIFIER, AgentName.DATA_BUILDER),
    (MessageType.DATASET_BUNDLE, AgentName.DATA_BUILDER, AgentName.TRAINER),
    (MessageType.DATASET_BUNDLE, AgentName.PARAMETER_MASTER, AgentName.TRAINER),
    (MessageType.DIAGNOSTIC_REQUEST, AgentName.PARAMETER_MASTER, AgentName.TRAINER),
    (MessageType.TRAIN_REQUEST, AgentName.TRAINER, AgentName.EVALUATOR),
    (MessageType.DIAGNOSTIC_RESULT, AgentName.TRAINER, AgentName.STRATEGY_INSPECTOR),
    (MessageType.EVAL_RESULT, AgentName.EVALUATOR, AgentName.STRATEGY_INSPECTOR),
    (MessageType.INSPECTION_RESULT, AgentName.STRATEGY_INSPECTOR, AgentName.TEACHER),
    (MessageType.INSPECTION_RESULT, AgentName.STRATEGY_INSPECTOR, AgentName.SYSTEM),
]

# 将契约列表转为 O(1) 查询字典: (消息类型, 发送者) → {允许的接收者集合}
SENDER_RECEIVER_MAP: dict[tuple[MessageType, AgentName], set[AgentName]] = {}
for msg_type, sender, receiver in CONTRACT_RULES:
    key = (msg_type, sender)
    if key not in SENDER_RECEIVER_MAP:
        SENDER_RECEIVER_MAP[key] = set()
    SENDER_RECEIVER_MAP[key].add(receiver)


def validate_message_contract(message: RoutedMessage) -> list[str]:
    """校验单条消息的 (类型, 发送者, 接收者) 是否符合契约。

    router 节点在每条消息进入时调用此函数。
    若 ENFORCE_HARNESS_CONTRACTS=true，违反契约直接抛异常阻断。
    """
    violations = []

    mt = message.header.message_type
    sender = message.header.sender
    receiver = message.header.receiver

    expected_receivers = SENDER_RECEIVER_MAP.get((mt, sender))
    if expected_receivers is not None and receiver not in expected_receivers:
        violations.append(
            f"Contract violation: {sender.value} sending {mt.value} to {receiver.value}, "
            f"expected one of {[r.value for r in expected_receivers]}"
        )

    return violations


def validate_invariant(invariant_name: str, state: dict) -> list[str]:
    """校验系统状态不变量（预留，当前未被任何节点调用）。

    三条规则确保执行顺序不会错乱：
    - search_before_filter:     有物化数据必须有搜索记录
    - dataset_before_train:     有候选模型必须有数据集路径
    - rollout_before_filter_post: 有后置过滤结果必须有 rollout 记录
    """
    violations = []

    if invariant_name == "search_before_filter":
        search_results = state.get("search_results", [])
        materialized = state.get("materialized_dataset_questions", [])
        if materialized and not search_results:
            violations.append("Invariant: filter received data before search results exist")

    elif invariant_name == "dataset_before_train":
        dataset_dir = state.get("dataset_dir", "")
        candidate_model = state.get("candidate_model_path", "")
        if candidate_model and not dataset_dir:
            violations.append("Invariant: training started without dataset bundle")

    elif invariant_name == "rollout_before_filter_post":
        rollout_runs = state.get("rollout_runs", [])
        filtered = state.get("filtered_questions", [])
        if filtered and not rollout_runs:
            violations.append("Invariant: post-rollout filter has results without rollout runs")

    return violations
