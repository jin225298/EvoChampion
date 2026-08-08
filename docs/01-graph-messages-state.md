# 01. Graph、消息协议与状态

本模块解释系统的机制层：LangGraph 图如何连接节点，消息如何被 router 校验和持久化，大 payload 如何用 artifact ref 传递，以及 `EvoState` 中哪些字段承载跨轮状态。

## 入口文件

- `src/harness.py`
- `src/nodes/router.py`
- `src/models/messages.py`
- `src/models/state.py`
- `src/tools/harness_contracts.py`
- `src/tools/message_artifacts.py`

## Graph 结构

`src/harness.py` 用 `StateGraph(EvoState)` 注册所有节点，并定义条件路由：

- `START -> bootstrap -> prompt_designer -> router`
- 普通消息由 `router` 根据 `pending_message.header.receiver` 路由。
- 发给 `filter` 的消息会再按 `message_type` 分到：
  - `MATERIALIZED_DATASET -> filter_pre_rollout`
  - `ROLLOUT_RESULT -> filter_post_rollout`
  - `ROLLOUT_REQUEST -> rollout_dispatcher`
- `rollout_dispatcher` 使用 `fan_out_to_workers()` 返回 `Send("rollout_worker", shard_state)`，并行处理题目 shard。
- `rollout_worker -> rollout_aggregator -> router`，聚合后再回到常规消息流。

## Router 职责

`router_node` 做四件事：

1. 用 `RoutedMessage.model_validate()` 校验消息外壳。
2. 根据 `MESSAGE_TYPE_TO_PAYLOAD` 校验 payload 类型。
3. 用 `validate_message_contract()` 检查 `(message_type, sender, receiver)` 是否符合契约。
4. 将消息写入 `artifacts/session_<trace_id>/router_messages/` 和 `router_messages.jsonl`。

router 不判断业务语义。比如它知道 `FILTER_RESULT` 可以从 `FILTER` 发到 `CLASSIFIER`，但不会判断题目是否足够、难度是否合理。

## 消息协议

核心对象在 `src/models/messages.py`：

- `MessageHeader`
  - `trace_id`
  - `round_id`
  - `sender`
  - `receiver`
  - `message_type`
  - `route_status`
  - `schema_version`
- `RoutedMessage`
  - `header`
  - `payload`
- `DataArtifactRef`
  - `artifact_id`
  - `local_path`
  - `count`
  - `kind`
  - `content_type`
  - `schema_version`

主要 payload：

- `GoalRequestPayload`
- `SearchRequestPayload`
- `SearchResultPayload`
- `DatasetSchemaRequestPayload`
- `DatasetSchemaResultPayload`
- `MaterializedDatasetPayload`
- `RolloutRequestPayload`
- `RolloutResultPayload`
- `FilterResultPayload`
- `ClassificationResultPayload`
- `DatasetBundlePayload`
- `TrainResultPayload`
- `EvalResultPayload`
- `InspectionResultPayload`

## 契约校验

`src/tools/harness_contracts.py` 定义合法消息三元组。例如：

```text
GOAL_REQUEST       SYSTEM             -> TEACHER
SEARCH_REQUEST     TEACHER            -> PARAMETER_MASTER
SEARCH_REQUEST     PARAMETER_MASTER   -> SEARCHER
SEARCH_RESULT      HF_SEARCH_TOOL     -> DATASET_REVIEWER
MATERIALIZED_DATASET SCREENING_ENTRY  -> FILTER
ROLLOUT_REQUEST    FILTER             -> ROLLOUT_DISPATCHER
ROLLOUT_RESULT     ROLLOUT_AGGREGATOR -> FILTER
FILTER_RESULT      FILTER             -> CLASSIFIER
DATASET_BUNDLE     DATA_BUILDER       -> TRAINER
TRAIN_RESULT       TRAINER            -> EVALUATOR
EVAL_RESULT        EVALUATOR          -> STRATEGY_INSPECTOR
INSPECTION_RESULT  STRATEGY_INSPECTOR -> TEACHER 或 SYSTEM
```

契约表中还保留了 `ROLLOUT_AGGREGATOR -> TEACHER` 的 `ROLLOUT_RESULT` 合法边，用于兼容历史/实验路径；当前 `rollout_aggregator_node` 的实际主路径始终把 `ROLLOUT_RESULT` 发给 `FILTER`。

`ENFORCE_HARNESS_CONTRACTS=true` 时，违反契约会进入 `format_error` 分支。

## Artifact Ref 机制

大量题目和 rollout 结果不会内联在消息里。`src/tools/message_artifacts.py` 提供：

- `write_json_artifact(trace_id, round_id, producer, name, data)`
- `load_json_artifact(ref)`
- `load_payload_list(inline_items, ref)`

文件路径模式：

```text
artifacts/session_<trace_id>/message_artifacts/round_<round_id>/<artifact_id>.json
```

常见大 payload：

- `screening_entry` 写 `materialized_questions`
- `filter` 写 `rollout_questions`
- `rollout_aggregator` 写 `scored_questions`
- `classifier` 写 `classified_questions`

## EvoState 关键字段

`src/models/state.py` 的 `EvoState` 是全图共享状态。常见字段：

基础执行：

- `user_goal`
- `trace_id`
- `round_id`
- `pending_message`
- `message_log`
- `route_decision`

数据流：

- `candidate_dataset_refs`
- `materialized_dataset_questions`
- `candidate_questions`
- `filtered_questions`
- `classified_questions`
- `train_questions`
- `lf_val_questions`
- `cotest_questions`
- `test_questions`
- `probe_questions`

rollout：

- `rollout_runs`
- `rollout_difficulty_distribution`
- `rollout_hard_ratio`
- `rollout_pass_count_histogram`
- `difficulty_threshold_policy`
- `difficulty_teacher_feedback`

训练评估：

- `champion_model_path`
- `candidate_model_path`
- `metrics_before`
- `metrics_after`
- `should_stop`
- `should_promote_candidate`
- `termination_reason`

策略状态：

- `current_search_node_id`
- `search_dag_nodes`
- `search_dag_edges`
- `current_training_hyperparams`
- `current_action_metadata`
- `sampling_plan`
- `replay_buffer_entries`
- `replay_sample_ratio_override`

数据生命周期：

- `dataset_pool`
- `pool_cursor`
- `consumed_dataset_ids`
- `dataset_states_path`
- `cross_dataset_pool`
- `quota_met`
- `quota_shortfall`
- `quota_accumulated_questions`
- `heldout_registry_path`
- `reserved_dataset_question_ids`

## Reducer

`rollout_runs` 使用自定义 reducer：

```text
old + new
```

这让多个 `rollout_worker` 并行返回的 shard 结果可以汇聚到同一个 state 字段。若新值是 `None`，reducer 返回空列表，用于清理旧 rollout 结果。

## 失败路径

- router 校验失败：生成 `FormatErrorPayload`，路由到 `format_error`，图结束。
- contract violation：若开启 enforcement，会变成 format error。
- artifact ref 文件不存在：消费端通常得到空列表或 fallback 行为。
- rollout fan-out 题目为空：返回空发送列表，后续节点需要根据消息/状态处理。
