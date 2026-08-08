# 05. 策略控制：Teacher、Parameter Master、DAG/MCTS 与 Replay

本模块解释系统的“决策层”：谁决定搜什么、采多少、怎么调训练参数、什么时候 replay、什么时候换数据。这里也是最容易和 `filter` 执行层混淆的地方。

## 入口文件

- `src/nodes/teacher.py`
- `src/nodes/parameter_master.py`
- `src/tools/strategy_policy.py`
- `src/tools/search_dag.py`
- `src/tools/replay_buffer.py`
- `src/tools/llm_decision.py`
- `config/mcts_action_space.json`

## 决策链路

```text
GOAL_REQUEST / INSPECTION_RESULT
  -> teacher
  -> SearchRequestPayload with sampling_plan
  -> parameter_master
  -> training_hyperparams + replay ratio + action_metadata
  -> SearchRequestPayload
  -> searcher
```

`teacher` 更关注“搜什么数据、目标难度和 dataset policy hint”。
`parameter_master` 更关注“怎么训练、怎么采样、是否 replay、是否复用数据、是否 merge shards”。

## teacher

`teacher_node` 的输入可能是：

- 首轮 `GOAL_REQUEST`
- 上一轮 `INSPECTION_RESULT`
- probe diagnostic focus

输出：

```text
TEACHER -> PARAMETER_MASTER
MessageType.SEARCH_REQUEST
```

payload 中的 `sampling_plan` 常见字段：

- `version`
- `source`
- `primary_axis=dynamic_difficulty`
- `secondary_axis=module`
- `difficulty_weights`
- `module_weights`
- `target_difficulty`
- `target_bucket`
- `dataset_policy_hint`
- `data_pressure`
- `llm_agent`

## teacher 的 leaf decisions

`teacher` 有多个 leaf LLM 子决策：

- `teaching_teacher.search_query`
- `teaching_teacher.target_difficulty`
- `teaching_teacher.difficulty_weights`
- `teaching_teacher.dataset_policy_hint`

它们都会有 deterministic fallback。输出会被清洗、枚举值 clamp、分布归一化。

## teacher fallback 逻辑

`decide_teacher_search()` 的 fallback 分支：

- round 0：直接用用户 goal 作为语义搜索请求。
- hard ratio 过高：降低 hard 权重，转向 medium/easy，并给 `merge_or_replace` hint。
- train pool 太小：倾向 merge shards 扩充数据。
- 正常模式：根据 probe per-difficulty acc 聚焦弱项。

## parameter_master

`parameter_master_node` 输入 `SearchRequestPayload`，输出两类可能：

- 如果决定复用上轮 exact batch：直接发 `DatasetBundlePayload` 给 `TRAINER`。
- 否则更新 action metadata 后，把 search request 发给 `SEARCHER`。

它负责：

- 选择 MCTS action。
- 选择训练超参。
- 选择 replay ratio。
- 合并 teacher sampling plan 和 curriculum plan。
- 决定 dataset selection mode。
- 决定 rollback 后是否 retry same data。
- 生成 `current_action_metadata`。

## action space

默认动作空间在 `config/mcts_action_space.json`。每个 action 可包含：

- `learning_rate`
- `num_train_epochs`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `replay_sample_ratio`
- `finetuning_type`
- 其他 LLaMA-Factory 静态字段

模板不是最终搜索单位，而是冷启动锚点、安全边界和静态字段来源。

## 连续参数 surrogate-UCB

当 `MCTS_CONTINUOUS_ACTION_ENABLED=true` 时，parameter master 会基于 DAG 历史 edge：

- 读取真实训练参数。
- 读取 raw reward。
- 在连续空间生成候选点。
- 对 `learning_rate` 用 log 尺度。
- 对 `replay_sample_ratio`、`num_train_epochs` 用线性尺度。
- 综合预测收益、探索不确定性、历史相似证据和训练成本。

最终选择的是具体参数点，而不是离散模板名。

关键配置：

- `MCTS_CONTINUOUS_ACTION_ENABLED`
- `MCTS_CONTINUOUS_CANDIDATES`
- `MCTS_CONTINUOUS_BANDWIDTH`
- `MCTS_CONTINUOUS_EXPLORATION_BETA`
- `MCTS_CONTINUOUS_EPOCH_COST_WEIGHT`
- `MCTS_TUNER_COLD_START_EDGES`

## curriculum sampling plan

`build_curriculum_sampling_plan()` 根据 probe easy/medium/hard accuracy 生成难度配比。

直觉：

- medium 很低：主攻 medium，保留 easy 巩固。
- medium 中等：继续 medium，开始加入 hard。
- hard 很低：提高 hard 占比。
- 都还行：更均衡。

parameter master 会用 `_merge_teacher_sampling_plan()` 合并 teacher plan：

- teacher 的 data pressure 可覆盖 difficulty weights。
- teacher module weights 会并入。
- dataset policy hint 会保留。

## replay ratio

replay ratio 由多层信号决定：

- action space 默认值。
- teacher plan 的 replay hint。
- mutation scale。
- replay teacher leaf decision。
- forgetting signal。

forgetting signal 包含：

- old error rate
- frozen degrade
- rollback streak
- last inspection decision

如果有遗忘风险，fallback 会提高 replay ratio。

## retry same data

rollback/prune 后，parameter master 会决定是否复用同一批数据再训：

- 如果上轮失败但 action metadata 允许 retry，可直接使用 `last_dataset_bundle`。
- 连续失败达到阈值时，会强制 fresh dataset，避免困在同一批坏数据上。
- 复用前会校验 dataset bundle 路径必须在当前 session 内，避免引用不安全路径。

## DAG/MCTS

DAG 节点和边在 `strategy_inspector` 每轮结束时更新，parameter master 下一轮读取。

节点：

- `node_id`
- `parent_node_ids`
- `metrics`
- `visit_count`
- `value_estimate`

边：

- `from_node_id`
- `to_node_id`
- `action_type`
- `action_summary`
- `round_id`
- `decision`
- `reward`
- `action_metadata`

策略层会基于历史：

- rollback streak
- probe/cotest/new skill 指标
- data pressure
- edge reward
- action metadata

选择下一轮父节点和动作。

## LLM leaf decision 设计

LLM agent 不直接自由生成复杂对象，而是拆成多个小 leaf：

- 每个 leaf 只输出一个 JSON 字段。
- 输出会经过 enum whitelist、float clamp、distribution normalization。
- 失败时回退 deterministic fallback。
- prompt 通过 prompt designer 注入 goal-specific context。

这减少了“一个大 JSON 全错导致整轮失败”的风险。

## 与 filter 的边界

策略层决定：

- `difficulty_weights`
- `module_weights`
- `replay_sample_ratio`
- `dataset_selection_mode`
- `training_hyperparams`
- `target_bucket`

filter 执行：

- 从物化题池采样。
- 复用 cached rollout。
- 去掉 heldout/used/defeated。
- 按配额补窗。
- 混合 replay。
- 生成给 classifier 的候选题。

因此，`filter` 是执行者，不是所有采样策略的唯一所有者。
