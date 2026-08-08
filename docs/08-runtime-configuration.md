# 08. 运行时、配置与服务器实验

本模块解释如何配置本地调试和服务器正式实验，以及 vLLM/Ray、HF cache、LLaMA-Factory、DeepSeek cleaner、MathBench 等运行时开关。

## 入口文件

- `config/settings.py`
- `.env.example`
- `run.sh`
- `run_job.sh`
- `requirements.txt`
- `pyproject.toml`

## 配置加载顺序

`config/settings.py` 会：

1. 定位项目根目录。
2. 如果 `.env` 存在，用 `python-dotenv` 加载。
3. 从环境变量读取所有配置。
4. 对部分路径创建目录，例如 `artifacts/`、`data/`。

命令行参数会在 `main.py` 中写入环境变量，因此优先级高于 `.env`。

## 最小本地配置

```bash
cp .env.example .env
```

至少确认：

```text
BASE_MODEL_NAME
CHAMPION_MODEL_PATH
AGENT_BASE_MODEL_NAME
CANDIDATE_MODEL_DIR
TRAINING_CONFIG_TEMPLATE
llamafactory-cli 已在 PATH 中
```

轻量调试建议：

```text
MAX_ROUNDS=1
FILTER_TARGET_QUESTIONS_PER_ROUND=20
ROLLOUT_MAX_CONCURRENT=1
GLOBAL_PROBE_SIZE=20
FROZEN_PROBE_SIZE=20
USE_VLLM=0
```

当前训练题难度标注固定执行一次 answer-only/thinking cascade；`ROLLOUT_TIMES` 主要保留给历史多次 rollout/mastered 阈值兼容，不是调高训练题 rollout 重复次数的主开关。

## 服务器脚本

`run.sh` 是正式实验脚本，包含：

- conda 环境激活。
- HF mirror/cache 环境。
- fallback datasets。
- 大规模 round/filter/rollout 设置。
- MathBench/OpenCompass 设置。
- Qwen base/champion/agent model 设置。
- vLLM/Ray actor 设置。
- py_compile 预检查。
- 最终调用 `python -u main.py ...`。

`run_job.sh` 是 Slurm wrapper：

- 设置 partition/gpu/cpu/mem/time。
- 加载 `.env`。
- 设置集群代理。
- 激活 conda。
- `bash -n run.sh`。
- `python -m py_compile ...`。
- `--dry-run` 模式只检查不运行。

推荐提交前：

```bash
bash run_job.sh --dry-run
```

## HF cache 与 fallback

常用环境变量：

```text
HF_ENDPOINT
HF_HOME
HF_DATASETS_CACHE
HF_HUB_CACHE
HUGGINGFACE_HUB_CACHE
HF_MODULES_CACHE
TRANSFORMERS_CACHE
XET_DISABLE
HF_HUB_ENABLE_HF_TRANSFER
HF_HUB_OFFLINE
HF_DATASETS_OFFLINE
TRANSFORMERS_OFFLINE
```

搜索 fallback：

```text
SEARCH_FALLBACK_MODE=empty|predefined
SEARCH_FALLBACK_DATASETS=gsm8k:main:train,...
```

HFD 下载：

```text
USE_HFD_DATASET_DOWNLOAD
HFD_SCRIPT_PATH
HFD_DATASET_CACHE_DIR
HFD_DOWNLOAD_TOOL
HFD_DOWNLOAD_THREADS
HFD_DOWNLOAD_JOBS
```

## 离线 offset-cache 模式

离线 offset-cache 模式让正式训练中所有模型和数据读取都走已有本地 cache，避免在 Slurm 作业中临时访问 Hugging Face、datasets-server 或下载数据集。适合已有完整本地 cache、希望训练期间完全断网的场景。

核心开关：

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DATASET_CACHE_MODE=offset
DATASET_OFFSET_CACHE_MODE=1
HFD_DATASET_CACHE_ONLY=1
DATASET_REVIEW_USE_DATASETS_SERVER_ROWS=0
DATASET_REVIEW_ALLOW_HFD_DURING_REVIEW=0
DATASET_SHARD_SELECTION_POLICY=single
DATA_WINDOW_RETRY_LIMIT=1
```

数据集 cache 路径：

```text
HFD_DATASET_CACHE_DIR=/path/to/hfd-datasets
```

语义拆开看：

- `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 约束 Hugging Face Hub、datasets 和 transformers 不发起在线请求。模型 snapshot、数据集 metadata 和 Arrow/cache 文件必须预先存在。
- `DATASET_CACHE_MODE=offset` 或 `DATASET_OFFSET_CACHE_MODE=1` 会让策略层按 offset window 推进数据窗口；当前窗口不够时，系统请求下一个窗口，而不是把多个 shard 混在一次选择里。
- `DATASET_SHARD_SELECTION_POLICY=single` 保持单窗口选择。即使出现 data pressure 或 rollback，`parameter_master` 也不会自动切到 `merge_shards`。
- `DATA_WINDOW_RETRY_LIMIT=1` 限制 loader recovery 的次数；一次窗口加载失败或配额不足后，只允许进入下一次 recovery，再耗尽就把错误交给图流程处理。
- `HFD_DATASET_CACHE_ONLY=1` 让 `src/tools/dataset_adapter.py` 只使用已有 HFD 目录。若 `HFD_DATASET_CACHE_DIR/<safe_dataset_id>/` 存在，则打印 `Using existing hfd dataset cache only` 并从该目录加载；若不存在，则回退到 `datasets.load_dataset(dataset_id)` 的普通 cache 路径。在 offline 变量开启时，这个回退不会下载，cache miss 会表现为加载失败或该 ref 被拒绝。
- `DATASET_REVIEW_USE_DATASETS_SERVER_ROWS=0` 禁止 dataset review 阶段请求 `https://datasets-server.huggingface.co/rows` 样本行接口。
- `DATASET_REVIEW_ALLOW_HFD_DURING_REVIEW=0` 在当前 cache-only/offset 语义下表示 review 阶段不触发下载；它不等于禁止读取 HFD cache。`dataset_reviewer` 会在 `HFD_DATASET_CACHE_ONLY=1`、`DATASET_OFFSET_CACHE_MODE=1` 或 `DATASET_CACHE_MODE=offset/offline/cache_only` 时允许读取已有 HFD cache 中的样本行。

`HFD_DATASET_CACHE_DIR` 应指向整理好的 HFD 数据集 cache 目录。如果通过 `.env` 或 wrapper 脚本加载配置，注意确保运行脚本中的显式导出不会被 `.env` 里更旧的路径覆盖。

正常日志信号：

```text
[dataset_adapter] Using existing hfd dataset cache only: ...
[dataset_reviewer] ... verdict=accept ...
[screening_entry] Loaded window 0:1000 ...
[filter] Quota shortfall remains; requesting next dataset window ...
```

offset 模式下出现 quota shortfall 并不一定是错误。它通常表示当前 `DATASET_WINDOW_SIZE` 内满足 review、schema、rollout 和 filter 条件的题目不足，系统正在推进到 `1000:2000`、`2000:3000` 等后续窗口补齐配额。

## vLLM/Ray 推理

`src/tools/model_runner.py` 支持 transformers、in-process vLLM 和 Ray named actor vLLM。

主要开关：

```text
USE_VLLM
USE_VLLM_FOR_LOCAL_CHECKPOINTS
USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS
VLLM_MAX_CACHED_ENGINES
VLLM_GPU_MEMORY_UTILIZATION
VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION
VLLM_ENFORCE_EAGER
VLLM_ENABLE_PREFIX_CACHING
VLLM_ENABLE_CHUNKED_PREFILL
VLLM_MAX_MODEL_LEN
VLLM_MAX_NUM_SEQS
VLLM_MAX_NUM_BATCHED_TOKENS
```

Ray actor：

```text
RAY_ADDRESS
RAY_NAMESPACE
VLLM_RAY_INFLIGHT_PER_CALL
VLLM_RAY_GENERATE_TIMEOUT_SECONDS
VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS
VLLM_RAY_ACTOR_NUM_GPUS
VLLM_RAY_ACTOR_MAX_RESTARTS
VLLM_RAY_ACTOR_MAX_TASK_RETRIES
VLLM_RAY_ACTOR_MAX_CONCURRENCY
```

设计要点：

- rollout worker 会并行，但 vLLM engine load/use 有锁，避免多个线程同时抢同一张 GPU。
- Ray named actor 让多 worker 共享一个 vLLM engine。
- 模型切换时会清理 HF cache、in-process vLLM cache 或 stale Ray actor。
- candidate 晋升/回滚后，strategy_inspector 会清理非 champion vLLM actors。

## thinking 开关

`run_model_batch(..., disable_thinking=True)` 会在 chat template 支持时设置 `enable_thinking=False`。

用途：

- rollout answer-only 阶段禁用 thinking。
- searcher 等短 JSON 决策常禁用 thinking。
- LLM judge 通常保持 thinking enabled。

## LLaMA-Factory

关键配置：

```text
TRAINING_CONFIG_TEMPLATE
TRAIN_FINETUNING_TYPE
TRAINING_TIMEOUT_SECONDS
TRAIN_LF_EVAL_ENABLED
TRAIN_EVAL_STRATEGY
TRAIN_EVAL_STEPS
TRAIN_EVAL_BATCH_SIZE
TRAIN_LOAD_BEST_MODEL_AT_END
TRAIN_SAVE_STEPS
TRAIN_SAVE_TOTAL_LIMIT
TRAIN_SAVE_ONLY_MODEL
TRAIN_RESUME_ENABLED
TRAIN_PACKING
TRAIN_NEAT_PACKING
TRAIN_TOKENIZED_PATH
```

LoRA：

```text
LORA_RANK
LORA_ALPHA
LORA_DROPOUT
LORA_TARGET_MODULES
```

候选模型保留：

```text
CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED
```

`src/tools/llm_factory.py` 实际调用 `llamafactory-cli train/export`。因此最关键的是当前 Python/conda 环境能直接执行 `llamafactory-cli`；`LLAMA_FACTORY_ROOT` 在 `settings.py` 和 `.env.example` 中仍保留，但当前训练启动逻辑不依赖它切换工作目录。

## DeepSeek cleaner codegen

默认关闭：

```text
DATA_CLEANER_PROVIDER=off
```

启用 DeepSeek：

```text
DATA_CLEANER_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CLEANER_MODEL=deepseek-v4-flash
DATA_CLEANER_MAX_REPAIR_ATTEMPTS=2
DATA_CLEANER_REQUEST_TIMEOUT_SECONDS=120
```

用途：为复杂 dataset row 生成确定性 cleaner，补齐 question/answer/reasoning/target style 等字段。

## MathBench/OpenCompass

启用：

```text
FROZEN_PROBE_EVAL_METHOD=mathbench_opencompass
FIXED_BENCHMARK_SOURCE=mathbench_opencompass
```

关键配置：

```text
MATHBENCH_OPENCOMPASS_ROOT
MATHBENCH_OPENCOMPASS_PYTHON
MATHBENCH_DATASET
MATHBENCH_DATASET_FILTER
MATHBENCH_SUMMARIZER
MATHBENCH_WORK_DIR
MATHBENCH_MAX_SEQ_LEN
MATHBENCH_MAX_OUT_LEN
MATHBENCH_HF_BATCH_SIZE
MATHBENCH_NUM_GPUS
MATHBENCH_MODEL_KWARGS
MATHBENCH_TOKENIZER_KWARGS
MATHBENCH_EXTRA_ARGS
```

运行前请确认 OpenCompass 环境可单独跑通。

## 常用规模参数

```text
MAX_ROUNDS
FILTER_TARGET_QUESTIONS_PER_ROUND
SCREENING_ENTRY_MAX_QUESTIONS
DATASET_WINDOW_SIZE
DATASET_PROFILE_WINDOW_SIZE
MAX_PROFILE_WINDOWS_PER_ROUND
MAX_PROFILE_ITEMS_PER_ROUND
ROLLOUT_TIMES
ROLLOUT_MAX_CONCURRENT
INFERENCE_BATCH_SIZE
EVAL_TEST_MAX_ITEMS
EVAL_COTEST_MAX_ITEMS
EVAL_MASTERED_MAX_ITEMS
GLOBAL_PROBE_SIZE
FROZEN_PROBE_SIZE
```

## 常见资源问题

显存不足：

- 降低 `VLLM_GPU_MEMORY_UTILIZATION`。
- 降低 `VLLM_MAX_NUM_SEQS`。
- 降低 `VLLM_MAX_NUM_BATCHED_TOKENS`。
- 降低 `INFERENCE_BATCH_SIZE`。
- 减少 `ROLLOUT_MAX_CONCURRENT`。
- 确认没有 stale Ray actor。

网络不稳：

- 使用 HF mirror。
- 设置 `SEARCH_FALLBACK_MODE=predefined`。
- 准备 fallback datasets。
- 预下载 base model 和常用数据集。

训练超时：

- 降低 `FILTER_TARGET_QUESTIONS_PER_ROUND`。
- 降低 `num_train_epochs`。
- 提高 `TRAINING_TIMEOUT_SECONDS`。
- 检查 LLaMA-Factory eval 是否太频繁。
