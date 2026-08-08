# 03. 数据准入、搜索、审查与物化

本模块解释从“我要找什么数据”到“得到可 rollout 的标准化题目列表”的过程。这里的核心问题是：HF 数据集 schema 千差万别，系统必须先搜索、审查、识别字段、窗口化加载，再把原始 row 转成统一题目格式。

## 入口文件

- `src/nodes/searcher.py`
- `src/nodes/hf_search_tool.py`
- `src/nodes/dataset_reviewer.py`
- `src/nodes/screening_entry.py`
- `src/nodes/dataset_schema_agent.py`
- `src/tools/hf_search.py`
- `src/tools/dataset_adapter.py`
- `src/tools/dataset_cleaner_codegen.py`
- `src/tools/dataset_state.py`

## 数据准入主链路

```text
teacher / parameter_master
  -> SearchRequestPayload
  -> searcher
  -> hf_search_tool
  -> dataset_reviewer
  -> screening_entry
  -> dataset_schema_agent
  -> screening_entry
  -> MaterializedDatasetPayload
  -> filter_pre_rollout
```

## SearchRequest

`SearchRequestPayload` 包含：

- `search_sources`
- `search_query`
- `goal`
- `retrieval_mode`
- `dataset_role`
- `sampling_owner`
- `target_labels`
- `sampling_plan`

注意：`sampling_owner="filter"` 是历史字段名。当前语义是“题目候选落地执行由 filter 处理”，不代表所有采样策略由 filter 决定。

## searcher

`searcher_node` 接收来自 `teacher` 或 `parameter_master` 的 search request。

它会：

- 调用 `search_expert` LLM leaf，对 query 和 sources 做轻量改写。
- 用 `normalize_content_search_query()` 清洗 query。
- 带上 `last_search_feedback`，避免重复无效搜索。
- 发给 `HF_SEARCH_TOOL`。

失败或 LLM 不可用时，fallback 为透传 teacher/parameter_master 的请求。

## hf_search_tool

`hf_search_tool_node` 负责实际 HF 搜索。

关键行为：

- `search_generation` 每次递增，用 progressive limit 扩大搜索范围。
- 搜索结果会过滤 `consumed_dataset_ids`，避免重复消费同一 dataset。
- 若 action metadata 要求 `dataset_selection_mode=merge_shards`，且结果集中都是同一 dataset，会通过 `build_dataset_shard_refs()` 生成 shard refs。
- 使用 `DatasetStateManager` 记录 cache hit/miss 反馈。
- 新 refs 只放入 `dataset_review_pending_refs`，不会直接进入 `dataset_pool`。

关键配置：

- `SEARCH_DATASET_REPO_LIMIT`
- `SEARCH_TIMEOUT_SECONDS`
- `SEARCH_FALLBACK_MODE`
- `SEARCH_FALLBACK_DATASETS`

## dataset_reviewer

`dataset_reviewer_node` 对候选数据集做准入判断。它的目标不是加载全量训练集，而是快速判断 dataset 是否值得进入后续 schema/物化阶段。

典型行为：

- 对 pending refs 异步 review。
- 抽样若干 rows 进行可用性判断。
- 识别 stuck/failed dataset，并用 backoff 避免反复卡住同一源。
- 只把 accepted refs 交给 `screening_entry`。

关键配置：

- `DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS`
- `DATASET_REVIEW_TIMEOUT_KILL_GRACE_SECONDS`
- `DATASET_REVIEW_FAILURE_BACKOFF_SECONDS`
- `DATASET_REVIEW_FAILURES_BEFORE_BACKOFF`

dataset review 的失败会写入 dataset state 的 review 字段，例如：

- `review_fail_count`
- `review_blacklisted_until`
- `review_failure_reason`
- `review_last_stage`

## screening_entry

`screening_entry_node` 是数据物化主节点。它负责：

- 从 accepted refs 或 dataset pool 中选择下一个 ref。
- 对 dataset 做 inspect，得到 split/columns/first row。
- 请求 `dataset_schema_agent` 做字段识别。
- 用 schema 加载指定 window。
- 将 raw rows 标准化成内部题目 dict。
- 维护 `dataset_pool`、`pool_cursor`、`consumed_dataset_ids`。
- 在补窗模式下选择下一窗口或下一 dataset。
- 若当前数据源不可用，触发 fresh search。

输出：

```text
SCREENING_ENTRY -> FILTER
MessageType.MATERIALIZED_DATASET
Payload: MaterializedDatasetPayload
```

大题目列表通常通过 `questions_ref` 传递。

## dataset_schema_agent

`dataset_schema_agent_node` 很小，但很关键：

- 输入 `DatasetSchemaRequestPayload(dataset_ref, inspect_result)`。
- 调用 `detect_columns_via_llm()`。
- 输出 `DatasetSchemaResultPayload(dataset_ref, schema, inspect_result)`。

schema 通常包括：

- question/input/problem 字段
- answer/output/target 字段
- reasoning/process/solution 字段
- final answer marker
- target style
- judge mode

## dataset_adapter

`src/tools/dataset_adapter.py` 是数据读取和字段适配层。它负责：

- HF dataset 加载。
- split/subset fallback。
- schema inspection。
- 根据 LLM column decision 构造 schema。
- 将 row 转换为系统内部题目。
- 支持 answer-only、CoT、证明、llm_judge 等不同目标格式。

## cleaner codegen

`src/tools/dataset_cleaner_codegen.py` 提供可选数据清洗代码生成。

配置：

- `DATA_CLEANER_PROVIDER=off|deepseek|local`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_CLEANER_MODEL`
- `DATA_CLEANER_MAX_REPAIR_ATTEMPTS`

DeepSeek 模式会生成确定性 cleaner，并经过安全检查。不要把它理解成默认启用；默认是 `off`。

## dataset_state

`DatasetStateManager` 持久化 dataset 和 item 状态。它是窗口化、缓存复用、补窗和防重复使用的基础。

dataset 状态：

- `unused`
- `in_use`
- `exhausted`

item 状态：

- `unused`
- `reserved`
- `used`
- `defeated`

item 还会缓存 rollout 信息：

- `pass_rate`
- `difficulty`
- `rollout_count`
- `rollout_model_key`
- `rollout_config_hash`
- `rollout_judge_version`
- `rollout_stage`

这使得后续窗口复用时可以跳过重复 rollout。

## 补窗模式

当 `filter_post_rollout` 发现 quota 不足，会设置：

- `data_replenishment_needed=True`
- `quota_shortfall`
- `quota_accumulated_questions`

随后 `screening_entry` 进入 replenishment mode：

- 优先尝试当前 dataset 的下一个 window。
- 若当前源耗尽，尝试 `dataset_pool` 或 `previous_dataset_refs`。
- 若没有可用 ref，可能触发 fresh search。
- 会维护 `windows_loaded_this_round` 和 `profile_items_loaded_this_round`，避免无限补窗。

关键配置：

- `DATASET_WINDOW_SIZE`
- `DATASET_PROFILE_WINDOW_SIZE`
- `MAX_PROFILE_WINDOWS_PER_ROUND`
- `MAX_PROFILE_ITEMS_PER_ROUND`
- `SCREENING_MIN_NEXT_WINDOW_REMAINDER`

## 失败与降级

- HF API 不可用：`SEARCH_FALLBACK_MODE=predefined` 时用 fallback datasets。
- review 卡住：按 timeout/backoff 跳过该 dataset。
- schema 识别失败：可能进入 fallback schema；fallback schema 题目会在 filter 前被丢弃。
- window 无题：标记 exhausted，尝试下一个 ref 或 fresh search。
- 全部 dataset 被拒：screening_entry 会请求新的 search，避免空训练继续。
