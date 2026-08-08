# 07. Training、Evaluation 与 Strategy Inspection

本模块解释数据包如何进入 LLaMA-Factory 训练，候选模型如何评估，以及最终如何决定 promote、rollback、prune、keep_branch 或 stop。

## 入口文件

- `src/nodes/trainer.py`
- `src/tools/llm_factory.py`
- `src/nodes/evaluator.py`
- `src/nodes/strategy_inspector.py`
- `src/tools/strategy_policy.py`
- `src/tools/candidate_cleanup.py`
- `src/tools/mathbench_probe.py`

## 主链路

```text
DATASET_BUNDLE
  -> trainer
  -> TRAIN_RESULT
  -> evaluator
  -> EVAL_RESULT
  -> strategy_inspector
  -> INSPECTION_RESULT
  -> teacher 或 system
```

## trainer

`trainer_node` 接收 `DatasetBundlePayload`。

它会：

- 检查 `train.json` 是否为空。
- 从 state 读取 `current_training_hyperparams`。
- 从 state 读取 `current_action_metadata`。
- 决定 `finetuning_type`、LoRA 参数、packing、resume。
- 调用 `launch_training()`。
- 输出 `TrainResultPayload`。

如果训练集为空，会跳过 LLaMA-Factory，写 `round_<n>_training_skipped/train_log.txt`，并返回 `success=false`。

## LLaMA-Factory 启动

`src/tools/llm_factory.py` 中 `launch_training()` 做以下事情：

1. 解析模型路径到本地 HF snapshot，绕开 hf-xet 问题。
2. 读取 `config/templates/llama_factory_sft.yaml`。
3. 注入 model、dataset_dir、dataset_name、output_dir、hyperparameters。
4. 写临时 config YAML。
5. 先用 offline cache 模式训练。
6. offline 失败后尝试 online 模式。
7. 验证输出目录是否有可加载模型文件。
8. 如果是 LoRA 且训练成功，尝试 merge 到 base model。

实际子进程命令是 `llamafactory-cli train <config.yaml>`；LoRA 导出时是 `llamafactory-cli export <export_config.yaml>`。因此运行环境需要能直接找到 `llamafactory-cli`，当前训练代码不会读取 `LLAMA_FACTORY_ROOT` 后再切目录执行。

输出路径：

```text
CANDIDATE_MODEL_DIR/
  candidate_<trace_id>_round<n>/
  config_<trace_id>_round<n>.yaml
  train_log_<trace_id>_round<n>.txt
```

## 训练结果

`TrainResultPayload` 包含：

- `candidate_model_path`
- `train_log_path`
- `success`
- `error_message`
- `trainer_log_jsonl_path`
- `training_loss_jsonl_path`
- `all_results_path`
- `trainer_state_path`
- `train_results_path`

evaluator 会把这些路径整理为 `training_summary`。

## diagnostic mode

如果 state 中 `diagnostic_mode=probe_diagnostic`，或收到 `DIAGNOSTIC_REQUEST`，trainer 不训练，而是：

- 用 champion 评估 frozen probe。
- 找到最弱 difficulty bucket。
- 写 `round_<n>_diagnostic/diagnostic.json`。
- 写 `worst_probe_questions.json`。
- 发 `DIAGNOSTIC_RESULT` 给 `strategy_inspector`。

下一轮 teacher 会用这些 focus questions 调整搜索和采样。

## evaluator

`evaluator_node` 接收 `TrainResultPayload`。

正常路径：

1. 检查 candidate 是否可加载。
2. 加载 `test.json`、`cotest.json`、frozen probe。
3. champion 和 candidate 在完全相同 prompts 上批量推理。
4. 对 test/cotest/frozen 分别判题。
5. 根据 `split_role` 拆 old/new ability。
6. 计算 forgetting_delta。
7. 计算 frozen probe accuracy。
8. 可选跑 frozen 3-shot。
9. 可选跑 MathBench/OpenCompass。
10. 调 `decide_evaluation_gates()` 生成 gate。
11. 发 `EvalResultPayload` 给 strategy_inspector。

训练失败路径：

- 跳过 candidate 推理。
- 用 champion 在 frozen probe 或 MathBench 上生成基线结果。
- 强制 `pass_new_skill_gate=false` 和 `should_promote_candidate=false`。

## 评估指标

`EvalResultPayload` 中常用指标：

- `new_skill_acc_before`
- `new_skill_acc_after`
- `cotest_acc_before`
- `cotest_acc_after`
- `old_ability_acc_before`
- `old_ability_acc_after`
- `new_ability_acc_before`
- `new_ability_acc_after`
- `forgetting_delta`
- `probe_acc_frozen`
- `probe_acc_champion`
- `probe_easy_acc`
- `probe_medium_acc`
- `probe_hard_acc`
- `pass_old_skill_gate`
- `pass_new_skill_gate`
- `pass_probe_gate`
- `pass_frozen_gate`
- `should_stop`
- `should_promote_candidate`

## Gate 语义

主要 gate：

- old skill gate：旧能力不能明显下降。
- new skill gate：新能力要有提升。
- probe gate：冻结 probe 达标。
- frozen gate：相对于 champion/base 的退化不能超过容忍。

当前晋升 gate 不使用外部 probe 分数。外部 probe 会在 bootstrap 阶段注册进 registry 防泄漏；`EvalResultPayload` 保留 `external_probe_acc` 字段，但 evaluator 主路径仍以 frozen probe 或 MathBench/OpenCompass 作为 probe gate 信号。

关键配置：

- `EVAL_OLD_SKILL_GATE`
- `EVAL_NEW_SKILL_GATE`
- `EVAL_PROBE_ACC_GATE`
- `MIN_NEW_SKILL_GAIN`
- `FROZEN_DEGRADE_TOLERANCE`
- `REWARD_FORGETTING_PENALTY_WEIGHT`
- `PROMOTION_TOTAL_RELATIVE_GAIN`

## MathBench/OpenCompass

当 `FROZEN_PROBE_EVAL_METHOD=mathbench_opencompass`：

- bootstrap 写 marker。
- evaluator 不加载本地 frozen prompts。
- 对 champion 和 candidate 调 `run_mathbench_probe()`。
- OpenCompass 结果被解析为 `probe_acc_frozen`。

关键配置：

- `MATHBENCH_OPENCOMPASS_ROOT`
- `MATHBENCH_OPENCOMPASS_PYTHON`
- `MATHBENCH_DATASET`
- `MATHBENCH_DATASET_FILTER`
- `MATHBENCH_SUMMARIZER`
- `MATHBENCH_WORK_DIR`
- `MATHBENCH_MAX_SEQ_LEN`
- `MATHBENCH_MAX_OUT_LEN`
- `MATHBENCH_HF_BATCH_SIZE`
- `MATHBENCH_NUM_GPUS`
- `MATHBENCH_EXTRA_ARGS`

## strategy_inspector

`strategy_inspector_node` 是一轮闭环的最终裁决者。

它会：

- 用 deterministic policy 得到初始决策。
- 用 inspection agent leaf decision 做受控 override。
- 更新 `should_stop`、`should_promote_candidate`、`rollback_streak`。
- finalize reserved dataset questions。
- 非 rollback 时 commit heldout/probe/train registry 变更。
- promote 时更新 champion model path。
- promote 时写 mastered memory。
- 更新 search DAG node/edge。
- promote 且允许 replay 时更新 replay buffer。
- 持久化 search DAG、replay buffer、evolution checkpoint。
- 清理非 champion candidate/vLLM actor。
- 若未结束，`round_id += 1` 并返回 teacher。

## 决策类型

- `promote`：candidate 成为新 champion，数据进入 replay，checkpoint 更新。
- `provisional_promote`：保留分支但不完全晋升。
- `keep_branch`：保留当前分支用于后续探索。
- `prune`：剪枝坏分支。
- `rollback`：回滚，通常不 commit 本轮 heldout/test registry 变更。
- `diagnostic`：诊断模式完成，跳过训练后回到 teacher。

## replay buffer 更新

只有 promote 且 `should_store_to_replay_buffer=true` 时，训练题进入 replay buffer。

replay entry 包含：

- `question_id`
- `question_text`
- `gold_answer`
- `rollout_gold_answer`
- `train_output`
- `target_style`
- `evaluation_method`
- `dynamic_difficulty`
- `bucket`
- `module`
- `source_round`
- `success_score`
- `used_in_rounds`

## 持久化产物

```text
artifacts/session_<trace_id>/
  search_dag.json
  replay_buffer.json
  evolution_checkpoint.json
  mastered_memory.json
  round_<n>_datasets/test_accuracy.json
  inference_traces/
```

候选模型：

```text
CANDIDATE_MODEL_DIR/candidate_<trace_id>_round<n>/
```

训练日志：

```text
CANDIDATE_MODEL_DIR/train_log_<trace_id>_round<n>.txt
```
