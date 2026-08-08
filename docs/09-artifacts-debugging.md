# 09. Artifacts 与调试指南

本模块解释一次运行会写哪些文件，以及遇到问题时从哪里开始查。

## session 根目录

所有会话产物位于：

```text
artifacts/session_<trace_id>/
```

常见文件和目录：

```text
router_messages/
router_messages.jsonl
message_artifacts/
global_probe_set.json
frozen_probe_set.json
agent_prompt_pack.json
teacher_decisions.jsonl
search_dag.json
replay_buffer.json
evolution_checkpoint.json
mastered_memory.json
heldout_registry.json
dataset_states/
holdout_eval.json
test_buffer.json
probe_pool.json
round_<n>_datasets/
round_<n>_diagnostic/
inference_traces/
```

候选模型一般在：

```text
CANDIDATE_MODEL_DIR/
  candidate_<trace_id>_round<n>/
  train_log_<trace_id>_round<n>.txt
  config_<trace_id>_round<n>.yaml
```

## router_messages

用途：检查节点间消息是否按预期流转。

`router_messages.jsonl` 每行包含：

- timestamp
- round
- sender
- receiver
- message_type
- route_status
- message_file
- payload_summary

如果流程突然结束，先看最后一条消息的 receiver/message_type。

## message_artifacts

用途：查看大 payload。

路径示例：

```text
message_artifacts/round_0/r0_filter_rollout_questions_*.json
message_artifacts/round_0/r0_rollout_aggregator_scored_questions_*.json
message_artifacts/round_0/r0_classifier_classified_questions_*.json
```

常见检查：

- materialized questions 是否有 `question_text` 和 `gold_answer`。
- rollout questions 是否被 heldout/dataset_state 过滤空。
- scored questions 的 `dynamic_difficulty` 分布。
- classified questions 的 `module/category` 是否正常。

## round datasets

每轮数据目录：

```text
round_<n>_datasets/
  train.json
  lf_val.json
  cotest.json
  test.json
  probe.json
  dataset_info.json
  split_manifest.json
  pipeline_log.json
  round_heldout_candidates.json
  test_accuracy.json
```

调试重点：

- `train.json` 为空：trainer 会跳过训练。
- `dataset_info.json` 错：LLaMA-Factory 找不到数据集。
- `split_manifest.json`：看 difficulty/module 分布、old/new test 数量、data pressure。
- `probe.json`：本轮准备进入持久 probe pool 的 intake，不是当前晋升 gate 直接读取的 frozen probe。
- `test_accuracy.json`：strategy_inspector 会用上一轮高准确 test 进入 old ability buffer。

## search_dag

`search_dag.json` 记录策略搜索历史。

看点：

- 每轮新增 edge 的 `decision`。
- `reward` 是否符合预期。
- `action_metadata.training_hyperparams`。
- `action_metadata.replay_sample_ratio`。
- `action_metadata.sampling_plan`。
- rollback/prune 是否集中在某种 action。

## replay_buffer

`replay_buffer.json` 只记录 promote 成功后允许存储的数据。

看点：

- `dynamic_difficulty` / `bucket` 分布。
- `module` 分布。
- `used_in_rounds` 是否增长。
- 是否有 holdout/probe 泄漏。

## dataset_states

`dataset_states/` 记录 dataset/item 生命周期和 rollout cache。

看点：

- dataset 是否 `exhausted`。
- item 是否 `unused/reserved/used/defeated`。
- `pass_rate/difficulty/rollout_count` 是否已经缓存。
- `review_blacklisted_until` 是否导致数据集被跳过。
- `rollout_model_key/config_hash/judge_version` 是否和当前一致。

## heldout_registry

用于排查数据泄漏或训练池突然变少。

常见状态：

- active holdout
- probe holdout
- train seen
- retired
- external probe

如果 filter/data_builder 丢了大量题，检查 registry 是否把候选都挡掉了。

外部 probe 当前在这里主要表现为泄漏隔离状态；不要把它和当前 evaluator 的晋升 gate 混为一谈，晋升 gate 主要读取 frozen probe 或 MathBench/OpenCompass 分数。

## inference traces

评估和 rollout 会写推理追踪。字段通常包括：

- trace_id
- round_id
- stage
- model_role
- model_path
- question_id
- prompt
- gold_answer
- prediction
- correct
- split_role
- module
- dynamic_difficulty
- metadata

用途：

- 查某题为什么判错。
- 查 champion/candidate 是否用了同一 prompt。
- 查 LLM judge 的 reason/schema errors。
- 查 disable_thinking 是否符合阶段。

## 常见问题定位

训练集为空：

1. 看 `round_<n>_datasets/train.json`。
2. 看 `split_manifest.json` 的 counts。
3. 看 filter scored artifact 是否为空。
4. 看 screening_entry materialized questions 是否为空。
5. 看 heldout registry 和 dataset_state 是否过滤过多。

rollout 全是 hard：

1. 看 scored_questions 的 `rollout_stage` 和 `prediction`。
2. 看 `judge_answer` 是否能解析 gold/pred。
3. 看 `INSTRUCTION_PREFIX` 是否合适。
4. 看 answer-only 是否被正确 disable thinking。
5. 看数据 schema 是否把答案/过程列取错。

HF 搜不到数据：

1. 看 `last_search_feedback`。
2. 看 `SEARCH_FALLBACK_MODE`。
3. 看 `SEARCH_FALLBACK_DATASETS`。
4. 看网络代理和 HF_ENDPOINT。
5. 看 dataset_review 是否全部 reject。

训练失败：

1. 看 `train_log_<trace_id>_round<n>.txt`。
2. 看 LLaMA-Factory config YAML。
3. 看 `dataset_info.json`。
4. 看 `TRAINING_TIMEOUT_SECONDS`。
5. 看 `BASE_MODEL_NAME` 是否可解析到本地 snapshot。

candidate 不晋升：

1. 看 evaluator 输出的 gate。
2. 看 `metrics_after`。
3. 看 `strategy_inspector` 日志。
4. 看 `search_dag.json` 本轮 edge。
5. 看 old/new ability 和 frozen probe 是否互相冲突。

显存没有释放：

1. 看 Ray actor 是否还在。
2. 看 `RAY_NAMESPACE` 是否和当前 job 唯一。
3. 看 strategy_inspector 是否清理 non-champion actor。
4. 降低 vLLM memory/batch 参数。

## 轻量检查命令

```bash
git diff --check -- README.md docs
PYTHONPATH=. uv run pytest tests/test_messages.py -q
PYTHONPATH=. uv run pytest tests/test_project_dependencies.py -q
```

README-only 修改通常只需要 Markdown 检查和链接检查；涉及行为代码时再跑相关测试。
