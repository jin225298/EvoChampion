# 02. Bootstrap、Prompt Designer 与首轮启动

本模块解释系统如何初始化一次进化会话：创建或恢复 `trace_id`、DAG、replay、probe、checkpoint，然后用 prompt designer 将用户目标注入各策略 agent。

## 入口文件

- `main.py`
- `config/settings.py`
- `src/nodes/bootstrap.py`
- `src/nodes/prompt_designer.py`
- `src/tools/search_dag.py`
- `src/tools/dataset_bank.py`
- `src/tools/question_registry.py`
- `src/tools/mathbench_probe.py`

## CLI 到 graph

`main.py` 做三件事：

1. 解析命令行参数。
2. 将 CLI 覆盖写入环境变量，例如 `BENCHMARK_DATASET_ID`、`MAX_ROUNDS`、`BASE_MODEL_NAME`。
3. 调用 `compile_graph().invoke({"user_goal": goal})`。

常用覆盖：

```text
--benchmark
--benchmark-subset
--benchmark-split
--benchmark-eval-split
--benchmark-question-key
--benchmark-answer-key
--benchmark-format
--instruction-prefix
--max-rounds
--model
```

## Bootstrap 职责

`bootstrap_node` 是图的第一个业务节点。它负责：

- 创建或恢复 `trace_id`。
- 加载 `evolution_checkpoint.json`。
- 加载或创建 search DAG root。
- 加载 replay buffer。
- 确定 `round_id`。
- 确定当前 `champion_model_path`。
- 构建 global probe 和 frozen probe。
- 注册 holdout/probe/external probe 到 `heldout_registry.json`。
- warmup/evaluate base model probe baseline。
- 发送第一条 `GOAL_REQUEST` 给 `TEACHER`。

## trace_id 与恢复

配置项：

- `EVOLVE_TRACE_ID`
- `CHAMPION_MODEL_PATH`
- `BASE_MODEL_NAME`
- `CANDIDATE_MODEL_DIR`

如果设置了 `EVOLVE_TRACE_ID`，系统会尝试复用对应 session：

```text
artifacts/session_<trace_id>/
  evolution_checkpoint.json
  search_dag.json
  replay_buffer.json
  mastered_memory.json
```

checkpoint 中常见字段：

- `next_round_id`
- `champion_model_path`
- `data_window_offset`
- `data_window_size`
- `replay_sample_ratio_override`
- `target_bucket`
- `rollout_difficulty_distribution`
- `round_data_stats`

如果 checkpoint 不存在，系统从 round 0 开始，并创建 DAG root。

## Probe 初始化

global probe 来源由配置控制：

- `GLOBAL_PROBE_SOURCE=benchmark_holdout`
- `GLOBAL_PROBE_SOURCE=benchmark_eval`
- seed probe bank fallback

关键配置：

- `BENCHMARK_DATASET_ID`
- `BENCHMARK_SUBSET`
- `BENCHMARK_EVAL_SPLIT`
- `GLOBAL_PROBE_SIZE`
- `FROZEN_PROBE_SIZE`
- `GLOBAL_PROBE_STRATIFIED_BY_MODULE`
- `PROBE_BANK_PATH`

当 `FROZEN_PROBE_EVAL_METHOD=mathbench_opencompass` 时，bootstrap 会写一个 MathBench marker，并跳过本地 frozen probe 构建；实际评估交给 OpenCompass。

## heldout registry

`heldout_registry.json` 防止训练污染评估集。bootstrap 会注册：

- global/frozen probe
- external probe
- benchmark holdout

后续 `filter`、`data_builder` 会通过 `drop_registered_questions()` 避免把这些题放入训练。

注意：当前外部 probe 的主要作用是 registry 隔离，避免这些题进入训练或普通测试 split。评估晋升 gate 当前主要使用 frozen probe 或 MathBench/OpenCompass；`external_probe_acc` 字段存在，但 evaluator 主路径没有实际用外部 probe 分数做晋升门控。

## Prompt Designer

`prompt_designer_node` 在 bootstrap 后立即运行。它会创建 `agent_prompt_pack.json`，并把用户目标变成结构化上下文注入策略 agent prompt。

它有 8 个 leaf decision：

1. `domain_goal`
2. `target_capabilities`
3. `search_keywords`
4. `boundary_signals`
5. `rare_signals`
6. `classifier_labels`
7. `classifier_label_notes`
8. `instruction_prefix`

输出文件：

```text
artifacts/session_<trace_id>/agent_prompt_pack.json
```

后续节点通过 `prompt_for_agent(state, agent_name, fallback_prompt)` 读取 prompt pack。

## Prompt 缓存

同一个 session 中，`prompt_designer` 会优先加载已存在的 `agent_prompt_pack.json`，避免每轮重复调用 LLM。

如果 leaf LLM 失败，会使用 `_fallback_prompt_design()`：

- 原始 goal 作为 domain goal。
- goal 派生搜索关键词。
- 默认 classifier labels 来自 `get_classifier_labels()`。
- instruction prefix 默认为中文解题指令。

## 首条消息

bootstrap 最终发出：

```text
SYSTEM -> TEACHER
MessageType.GOAL_REQUEST
Payload: GoalRequestPayload(goal=user_goal)
```

这条消息进入 router 后，系统正式开始搜索和训练循环。
