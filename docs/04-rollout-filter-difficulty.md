# 04. Rollout、难度判定与 Filter 执行层

本模块解释题目如何被当前 champion 模型评估、如何得到 `dynamic_difficulty`，以及 `filter` 如何消费策略计划完成配额执行、补窗、缓存复用和 replay 混合。

## 入口文件

- `src/nodes/filter.py`
- `src/nodes/rollout_dispatcher.py`
- `src/nodes/rollout_worker.py`
- `src/nodes/rollout_aggregator.py`
- `src/tools/difficulty_tagger.py`
- `src/tools/model_runner.py`
- `src/tools/llm_judge.py`
- `src/tools/replay_buffer.py`

## 总链路

```text
MaterializedDatasetPayload
  -> filter_pre_rollout
  -> RolloutRequestPayload
  -> rollout_dispatcher
  -> rollout_worker(s)
  -> rollout_aggregator
  -> RolloutResultPayload
  -> filter_post_rollout
  -> FilterResultPayload
  -> classifier
```

## 难度判定

训练题 `dynamic_difficulty` 是相对于当前 champion 的动态难度。它不是数据集自带标签，也不是人工预标注。

`src/tools/difficulty_tagger.py` 使用两阶段 cascade：

1. 先跑 answer-only rollout：`disable_thinking=True`，只要求输出最终答案。
2. answer-only 答对：`easy`。
3. answer-only 答错，再跑 thinking rollout：`disable_thinking=False`。
4. thinking 答对：`medium`。
5. thinking 仍错：`hard`。
6. 无法判断：`unknown`。

语义：

```text
easy   = 当前 champion 不开思考就能答对
medium = 当前 champion 不开思考答错，但开思考能答对
hard   = 当前 champion 开思考也答不对
```

判题方式：

- 有标准答案：走 `judge_answer()`，内部优先使用 `math-verify`，再尝试 SymPy / numeric fallback。
- 需要语义判定或证明：走 `llm_judge`。

## ROLLOUT_TIMES 的当前含义

历史逻辑里有基于多次 pass count 的难度阈值。当前训练题难度标注函数中，`rollout_count` 参数保留为兼容旧调用，但实际训练难度评估固定做一次两阶段 cascade。

`rollout_aggregator` 仍会记录：

- `pass_count`
- `rollout_count`
- `pass_rate`
- `rollout_pass_count_histogram`
- `difficulty_threshold_policy`

这些是可审计质量信号和评估/兼容字段，不覆盖训练题的 stage-based `dynamic_difficulty`。

## rollout_dispatcher

`rollout_dispatcher` 将 `candidate_questions` 切成多个 shard，用 `Send("rollout_worker", shard_state)` 并行发送。

并发上限：

- `ROLLOUT_MAX_CONCURRENT`

每个 worker state 包含：

- `candidate_questions`
- `champion_model_path`
- `trace_id`
- `round_id`
- `rollout_worker_idx`
- `rollout_start_idx`
- `rollout_config_hash`
- `rollout_judge_version`

## rollout_worker

`rollout_worker_node` 调用 `tag_questions_by_pass_rate()`，返回 `rollout_runs`。

每条 rollout row 包括：

- `question_id`
- `question_text`
- `gold_answer`
- `prediction`
- `correct`
- `rollout_stage`
- `dynamic_difficulty`
- `evaluation_method`
- `needs_judge`
- source dataset metadata
- `rollout_model_key`
- `rollout_config_hash`
- `rollout_judge_version`

## rollout_aggregator

`rollout_aggregator_node` 合并所有 worker 的结果：

- 按 `question_id` 合并。
- 统计 `correct_flags`。
- 计算 `pass_count`、`rollout_count`、`pass_rate`。
- 选取 worker 投票中的有效 `dynamic_difficulty`。
- 写 `scored_questions` artifact。
- 计算难度分布和 hard ratio。
- 调用 difficulty teacher 生成 advisory metadata。
- 始终发送 `ROLLOUT_RESULT` 给 `FILTER`。

重要边界：

- difficulty teacher 的 `accept_batch` 会被强制视为继续主流程。
- hard-dominated 只作为 data pressure/advisory signal。
- 当前不会整批 reject，也不会打回 teacher 重试。

## filter_pre_rollout

pre 阶段输入 `MaterializedDatasetPayload`，输出 `RolloutRequestPayload` 或伪 `RolloutResultPayload`。

步骤：

1. 从 `questions_ref` 读 materialized questions。
2. 丢弃 forbidden fallback schema 题。
3. 根据 `dataset_state` 丢弃已 used/defeated 题。
4. 根据 heldout registry 丢弃不能进训练的题。
5. 根据 `rollout_model_key + rollout_config_hash + rollout_judge_version` 查缓存。
6. 有缓存且 window 非新探索时复用 cached scored questions。
7. 对 uncached candidates 应用 `sample_questions_for_round()`。
8. 混合 replay buffer。
9. 若全缓存，直接构造伪 rollout result。
10. 否则写 `rollout_questions` artifact 并发给 dispatcher。

## filter_post_rollout

post 阶段输入 `RolloutResultPayload`，输出 `FilterResultPayload` 或触发补窗。

步骤：

1. 合并 cached scored 和实时 rollout scored。
2. 收集 mastered 题目，但仍保留在训练池中强化旧知识。
3. 更新题目 window metadata。
4. 更新 `dataset_state` pass_rate/difficulty cache。
5. 将 fresh 题追加到 `cross_dataset_pool`。
6. 按 `sampling_plan.difficulty_weights` 计算目标配额。
7. 优先从 active dataset pool 填充。
8. 配额不足时从其他 dataset pool 补充。
9. 仍不足且还有预算时，设置 replenishment 状态回到 `screening_entry` 加载下一 window。
10. budget 耗尽时用部分配额继续。
11. 最终混合 replay 并发给 `classifier`。

## quota 与 pool target

`FILTER_TARGET_QUESTIONS_PER_ROUND` 是最终训练题数目标。因为 data_builder 还会拆出 train/cotest/test/probe/lf_val，filter post 会通过 `_post_rollout_pool_target()` 把训练目标放大为 split 前 pool target。

这解释了为什么 filter 可能选出多于最终 train 数量的题：它是在为后续 split 预留空间。

## replay 混合

replay 来源于历史 promote 成功的数据。filter pre 和 post 都会考虑 replay，但补窗循环中会禁用 replay ratio，避免 replay 影响 fresh data quota 补齐。

replay 采样按 bucket/difficulty 感知，字段包括：

- `dynamic_difficulty`
- `bucket`
- `module`
- `used_in_rounds`

## mastered 题目

mastered 判定基于 correct count：

- 若 rollout_count 等于 `ROLLOUT_TIMES`，使用 `MASTERED_CORRECT_THRESHOLD`。
- 否则使用 `ceil(rollout_count * MASTERED_CORRECT_RATIO)`。

当前 post filter 中，mastered 题不会被过滤掉；它们保留在训练池中强化旧能力，但不会直接进入 test pool。

## 关键配置

- `FILTER_TARGET_QUESTIONS_PER_ROUND`
- `FILTER_SAMPLING_METHOD`
- `FILTER_DEDUPLICATE`
- `FILTER_DROP_MASTERED`
- `TARGET_BUCKET_RATIO`
- `ROLLOUT_MAX_NEW_TOKENS`
- `ROLLOUT_TEMPERATURE`
- `ROLLOUT_TOP_P`
- `ROLLOUT_MAX_CONCURRENT`
- `MASTERED_CORRECT_RATIO`
- `MASTERED_CORRECT_THRESHOLD`
- `REPLAY_BUFFER_SAMPLE_RATIO`
- `MAX_PROFILE_WINDOWS_PER_ROUND`
- `MAX_PROFILE_ITEMS_PER_ROUND`

## 关键产物

```text
message_artifacts/round_<n>/r<n>_filter_rollout_questions_*.json
message_artifacts/round_<n>/r<n>_rollout_aggregator_scored_questions_*.json
dataset_states/<dataset>_state.json
router_messages.jsonl
```

## 常见误解

- 不是 `filter` 判断题目难度。
- 不是 pass-rate 阈值覆盖当前训练题难度。
- 不是 hard batch 会被整批拒绝。
- `difficulty_teacher_feedback` 是后续策略的输入信号，不是流程中断条件。
