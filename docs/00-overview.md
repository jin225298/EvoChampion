# 00. 系统总览

EvoChampion 是一个围绕“目标能力提升”的模型自进化训练闭环。它不是单一训练脚本，而是一张 LangGraph 状态图：每轮从目标和历史指标出发，找数据、审数据、标难度、构建训练包、训练候选模型、评估候选模型，再决定是否晋升为新的 champion。

## 核心术语

- `champion`：当前被系统认可的最好模型。rollout 难度标注、训练起点和评估对照都围绕它展开。
- `candidate`：本轮训练产生的新模型。只有通过评估 gate 和 `strategy_inspector` 决策后才会晋升为 champion。
- `round_id`：进化轮次。每轮都会写独立数据、训练、评估和 DAG 记录。
- `trace_id`：一次实验会话 ID。所有 artifacts 都写到 `artifacts/session_<trace_id>/`。
- `dynamic_difficulty`：相对于当前 champion 的题目动态难度，由两阶段 rollout 判定，详见 [04-rollout-filter-difficulty.md](04-rollout-filter-difficulty.md)。
- `sampling_plan`：策略层给执行层的采样计划，通常以 `dynamic_difficulty` 为主轴、`module` 为次轴。
- `DAG/MCTS`：历史搜索和训练动作图。`parameter_master` 用它选择下一轮动作和超参，`strategy_inspector` 用评估结果回写节点/边。
- `replay buffer`：晋升成功的数据会进入 replay，用于后续防遗忘和稳定训练。
- `heldout registry`：防数据泄漏的注册表，阻止 holdout/probe/test 题进入训练。

## 一轮循环

```text
bootstrap
  -> prompt_designer
  -> router
  -> teacher
  -> parameter_master
  -> searcher
  -> hf_search_tool
  -> dataset_reviewer
  -> screening_entry
  -> dataset_schema_agent
  -> filter_pre_rollout
  -> rollout_dispatcher
  -> rollout_worker(s)
  -> rollout_aggregator
  -> filter_post_rollout
  -> classifier
  -> data_builder
  -> trainer
  -> evaluator
  -> strategy_inspector
  -> teacher 或 system
```

图由 `src/harness.py` 组装。普通节点都经 `router` 校验消息；rollout worker 使用 `langgraph.types.Send` 并行 fan-out，结果通过 `EvoState.rollout_runs` reducer 汇总到 `rollout_aggregator`。

## 三个分层

机制层：

- LangGraph 图、router、消息协议、artifact ref、state reducer、dataset state。
- 目标是保证消息结构和产物传递可靠，不直接做业务策略。

策略层：

- `teacher`、`parameter_master`、`data_builder`、`strategy_inspector` 及其 LLM leaf decision。
- 目标是决定搜什么、怎么配难度、怎么调训练参数、是否 replay、是否晋升。

执行层：

- `hf_search_tool`、`dataset_reviewer`、`screening_entry`、`filter`、`rollout_worker`、`trainer`、`evaluator`。
- 目标是把策略计划落到数据、模型推理、训练、评估和持久化产物。

## 关键边界

- `filter` 不是 rollout 难度策略的唯一控制者。它消费上游 `sampling_plan` 和 rollout 标签，负责去重、缓存复用、配额执行、补窗和 replay 混合。
- `rollout_aggregator` 内的 difficulty teacher 是 advisory。它记录 hard-dominated 等信号，但当前流程不会因为一批题太难整批拒绝。
- 训练题 `dynamic_difficulty` 由 rollout worker 的 cascade stage 判定，不由 pass-rate 阈值或 filter 覆盖。
- 大 payload 用 `DataArtifactRef` 传路径，不直接塞入 router 消息。

## 代码地图

```text
main.py
config/settings.py
config/templates/llama_factory_sft.yaml
config/mcts_action_space.json

src/harness.py
src/models/messages.py
src/models/state.py

src/nodes/
  bootstrap.py
  prompt_designer.py
  router.py
  teacher.py
  parameter_master.py
  searcher.py
  hf_search_tool.py
  dataset_reviewer.py
  screening_entry.py
  dataset_schema_agent.py
  filter.py
  rollout_dispatcher.py
  rollout_worker.py
  rollout_aggregator.py
  classifier.py
  data_builder.py
  trainer.py
  evaluator.py
  strategy_inspector.py

src/tools/
  strategy_policy.py
  model_runner.py
  llm_factory.py
  dataset_adapter.py
  dataset_cleaner_codegen.py
  dataset_state.py
  question_registry.py
  message_artifacts.py
  mathbench_probe.py
```

## 最小运行路径

```bash
python main.py "提高数学能力" \
  --benchmark gsm8k \
  --benchmark-subset main \
  --benchmark-question-key question \
  --benchmark-answer-key answer \
  --benchmark-format gsm8k \
  --benchmark-eval-split test \
  --max-rounds 3
```

这个命令会经过完整图，只是规模由配置控制。真实 GPU 实验通常使用 `run.sh` 和 `run_job.sh`，详见 [08-runtime-configuration.md](08-runtime-configuration.md)。
