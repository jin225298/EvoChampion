# 06. Classifier 与 Data Builder

本模块解释 rollout/filter 之后，题目如何被分类、如何拆成 train/lf_val/cotest/test/probe，如何写成 LLaMA-Factory 可消费的数据包。

## 入口文件

- `src/nodes/classifier.py`
- `src/nodes/data_builder.py`
- `src/tools/data_builder_helpers.py`
- `src/tools/data_pipeline/`
- `src/tools/question_registry.py`
- `src/tools/cot_format.py`

## 主链路

```text
FILTER_RESULT
  -> classifier
  -> CLASSIFICATION_RESULT
  -> data_builder
  -> DATASET_BUNDLE
  -> trainer
```

## classifier

`classifier_node` 接收 filter 输出的题目，给每题补充：

- `category`
- `module`
- `confidence`

分类优先级：

1. GLiNER 可用时，用 GLiNER 对 `question_text` 做 zero-shot entity/category prediction。
2. GLiNER 不可用时，沿用已有 `module`。
3. 若没有 module，则用 `infer_module()` 规则 fallback。

关键配置：

- `USE_GLINER`
- `GLINER_MODEL_NAME`
- `CLASSIFIER_TARGET_LABELS`
- `CLASSIFIER_MIN_CONFIDENCE`

如果 GLiNER 加载失败，会全局禁用，避免每轮重复失败。

## data_builder 输入

`data_builder_node` 接收 `ClassificationResultPayload`，通过 artifact ref 读回题目。

题目应包含：

- `question_id`
- `question_text`
- `gold_answer`
- `rollout_gold_answer`
- `train_output`
- `target_style`
- `evaluation_method`
- `source_dataset_id`
- `source_dataset_row_id`
- `module` / `category`
- `dynamic_difficulty`
- `pass_count`
- `rollout_count`
- `pass_rate`

## holdout registry 防泄漏

data_builder 在 split 前会使用 `heldout_registry.json`：

- 阻止 active holdout/probe 题进入 train。
- 阻止 train-seen/retired 题进入新 eval split。
- 将本轮新产生的 holdout/probe 候选交给 strategy_inspector 决定是否 commit。

相关函数：

- `drop_registered_questions()`
- `mark_questions_active_holdout()`
- `mark_questions_probe_holdout()`
- `mark_questions_train_seen()`
- `retire_holdout_questions()`

## split 逻辑

新题先按 sampling plan 选择，再按 difficulty/module 分层拆分：

- train
- lf_val
- cotest
- test
- probe_pool_intake

关键比例：

- `TRAIN_SPLIT_RATIO`
- `COTEST_SPLIT_RATIO`
- `TEST_SPLIT_RATIO`
- `PROBE_POOL_INTAKE_RATIO`
- `LF_VAL_SPLIT_RATIO`

如果 `TRAIN_LF_EVAL_ENABLED=true`，会从 train 中 carve 一部分作为 LLaMA-Factory eval set。

## replay 与 old/new ability test

data_builder 会把 replay 题只混入 train，不进入 eval split。

test 中有两类：

- `old_ability`：来自历史高准确 test buffer，用于遗忘检测。
- `new_ability`：来自本轮 fresh data，用于新能力评估。

`split_role` 会写入 Alpaca record metadata，evaluator 后续用它区分 old/new。

## stable holdout eval

首次运行时，data_builder 会创建稳定 holdout：

```text
artifacts/session_<trace_id>/holdout_eval.json
```

这些题会从训练候选中移除，避免训练污染长期对照集。

## probe pool

`probe_pool.json` 是持久 probe 候选池。data_builder 本轮会写入 `probe.json` 作为 probe_pool_intake，strategy_inspector 在非 rollback 决策后把 intake 写入持久池。

注意不要把 `round_<n>_datasets/probe.json` 和 bootstrap 构建的 `frozen_probe_set.json` 混淆。当前 evaluator 的晋升 gate 读取 frozen probe 或 MathBench/OpenCompass；`probe.json` 主要是本轮候选样本的 intake 产物。

## 数据格式输出

每轮写入：

```text
artifacts/session_<trace_id>/round_<round_id>_datasets/
  train.json
  lf_val.json
  cotest.json
  test.json
  probe.json
  dataset_info.json
  split_manifest.json
  round_heldout_candidates.json
  pipeline_log.json
```

`dataset_info.json` 是 LLaMA-Factory 数据集注册文件，包含：

```json
{
  "round_N_train": {
    "file_name": "train.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system",
      "history": "history"
    }
  }
}
```

## Alpaca record

`to_alpaca_record()` 输出：

- `instruction`
- `input`
- `output`
- `system`
- `history`

并保留 metadata：

- `split_tag`
- `split_role`
- `question_id`
- `gold_answer`
- `rollout_gold_answer`
- `train_output`
- `evaluation_method`
- `target_style`
- `module`
- `dynamic_difficulty`
- source dataset metadata

## CoT / answer output 处理

`target_style` 控制训练输出：

- `answer`：输出最终答案。
- `cot`：输出推理过程和最终 boxed answer。

`src/tools/cot_format.py` 会探测 tokenizer chat template 的 thinking delimiters，例如 Qwen3 的 `<think>` / `</think>`，并构造适合模型的训练输出。

## data_builder leaf decisions

data_builder 还会对计划做最后一层 leaf decision：

- `data_builder.difficulty_weights`
- `data_builder.module_weights`
- `data_builder.replay_sample_ratio`

它们以当前 classified distribution、difficulty teacher feedback、available modules 和 sampling_plan 为上下文。

## pass-through quota

如果 filter 已经完成 quota-balanced pool，data_builder 会进入 pass-through 模式：

- `pass_through_quota_balanced`
- `pass_through_quota_partial`

这时 data_builder 不会再次按 sampling plan 强行重采样，以免破坏 filter 的补窗配额结果。

## round_data_stats

data_builder 会写回：

- `input_count`
- `train_count`
- `lf_val_count`
- `cotest_count`
- `test_count`
- `train_eligible_count`
- `test_eligible_count`
- `hard_count`
- `hard_ratio`
- `low_train_signal`
- `hard_dominated_signal`
- `needs_more_data`
- `fresh_reserved_by_dataset`
- `data_builder_sampling_mode`

这些数据压力信号会被下一轮 teacher/parameter_master 使用。
