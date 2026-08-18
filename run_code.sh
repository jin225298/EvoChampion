#!/bin/bash
# Code-domain self-evolution smoke test on Qwen3-0.6B.
# DOMAIN=code: correctness is decided by *executing tests*, never by an LLM
# reading the code and judging behavioral equivalence.
#
# Loop: data load -> review -> standardize -> rollout difficulty tagging ->
#       train (LoRA) -> code-test eval -> promote/rollback -> next round.
# This milestone runs MAX_ROUNDS=1 to prove the loop is wired end-to-end.
set -eo pipefail

echo "Job started at $(date)"
echo "Host: $(hostname)"

CONDA_SH="${CONDA_SH:-/home/kang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-agentevolver}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
export DOMAIN=code
export ANSWER_VERIFIER_TYPE=auto

# ---------------------------------------------------------------------------
# Workspace on /data2 — own folder named after the execution folder
# (13641-harrier). Never write candidate models / caches into the shared
# /home quota.
# ---------------------------------------------------------------------------
WORKSPACE_DIR="${WORKSPACE_DIR:-/data2/group_何向南/kang/13641-harrier}"
export HF_HOME="${HF_HOME:-$WORKSPACE_DIR/huggingface}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export CANDIDATE_MODEL_DIR="${CANDIDATE_MODEL_DIR:-$WORKSPACE_DIR/code_candidates}"
export CODE_EXEC_CACHE_DIR="${CODE_EXEC_CACHE_DIR:-$WORKSPACE_DIR/code_exec_cache}"
mkdir -p "$CANDIDATE_MODEL_DIR" "$HF_HOME" "$CODE_EXEC_CACHE_DIR"

# ---------------------------------------------------------------------------
# Network / cache
# ---------------------------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export XET_DISABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
# NOTE: the Search section below flips these to offline=1 for the smoke run
# (model is pre-cached, data is local) so the compute node never hangs on the
# flaky hub network.
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
export REQUESTS_CONNECT_TIMEOUT=30
export REQUESTS_READ_TIMEOUT=180

# Slurm compute nodes do not run the login-node localhost proxy (127.0.0.1:1081).
# hf-mirror.com is reachable directly from the cluster, so clear any inherited
# localhost proxy to avoid routing downloads through a non-existent proxy.
if [[ "${http_proxy:-}" == *127.0.0.1* || "${http_proxy:-}" == *localhost* ]]; then
  export http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY=""
  echo "[run_code] cleared localhost proxy; using direct connection to $HF_ENDPOINT"
fi

# ---------------------------------------------------------------------------
# Search: code datasets that ship executable tests. The cluster's compute
# nodes have flaky DNS/SSL to hf-mirror.com, so the smoke run uses a LOCAL
# synthetic code dataset (data/code_train) shipped in the repo. The reviewer
# loads it non-streaming (no network needed).
# ---------------------------------------------------------------------------
export USE_HFD_DATASET_DOWNLOAD="${USE_HFD_DATASET_DOWNLOAD:-0}"
export SEARCH_TIMEOUT_SECONDS=60
export SEARCH_DATASET_REPO_LIMIT=8
export SEARCH_FALLBACK_MODE=predefined
CODE_TRAIN_DIR="${CODE_TRAIN_DIR:-/data2/group_何向南/kang/13641-harrier/EvoChampion/data/code_train}"
export SEARCH_FALLBACK_DATASETS="${CODE_TRAIN_DIR}::train"
# Offline mode: model + datasets come from the cache / repo; no hub access from
# the compute node (avoids hanging on flaky DNS/SSL). Forced to 1 here so it
# overrides the HF_HUB_OFFLINE=0 default above.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Reviewer: fail fast on any hub attempt; local data loads instantly.
export DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS="${DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS:-60}"
export DATASET_REVIEW_FIRST_ACCEPT_TIMEOUT_SECONDS="${DATASET_REVIEW_FIRST_ACCEPT_TIMEOUT_SECONDS:-60}"
export DATASET_REVIEW_REPLENISHMENT_WAIT_SECONDS="${DATASET_REVIEW_REPLENISHMENT_WAIT_SECONDS:-30}"
export DATASET_REVIEW_ROWS_TIMEOUT_SECONDS="${DATASET_REVIEW_ROWS_TIMEOUT_SECONDS:-30}"
export DATASET_REVIEW_STREAMING_TIMEOUT_SECONDS="${DATASET_REVIEW_STREAMING_TIMEOUT_SECONDS:-30}"

# ---------------------------------------------------------------------------
# Smoke-test scale (MAX_ROUNDS=1)
# ---------------------------------------------------------------------------
export MAX_ROUNDS=1
export FILTER_TARGET_QUESTIONS_PER_ROUND=8
export SCREENING_ENTRY_MAX_QUESTIONS=8
export DATASET_PROFILE_WINDOW_SIZE=8
export MAX_PROFILE_WINDOWS_PER_ROUND=1
export MAX_PROFILE_ITEMS_PER_ROUND=8
export HOLDOUT_EVAL_SIZE=4
export ROLLOUT_TIMES="${ROLLOUT_TIMES:-2}"
export ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-1024}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"

# ---------------------------------------------------------------------------
# Judging: execution only. No LLM-as-judge in the code domain.
# ---------------------------------------------------------------------------
export USE_LLM_AS_JUDGE=0
export LLM_JUDGE_MAX_ITEMS=0
export LLM_JUDGE_BATCH_SIZE="${LLM_JUDGE_BATCH_SIZE:-64}"
export FROZEN_PROBE_EVAL_METHOD=local
export CODE_EXEC_TIMEOUT_SECONDS="${CODE_EXEC_TIMEOUT_SECONDS:-10}"
export CODE_EXEC_MAX_MEMORY_MB="${CODE_EXEC_MAX_MEMORY_MB:-512}"
export CODE_EXEC_MAX_WORKERS="${CODE_EXEC_MAX_WORKERS:-8}"
export CODE_EXEC_CACHE_ENABLED="${CODE_EXEC_CACHE_ENABLED:-1}"

# ---------------------------------------------------------------------------
# Benchmark: local code smoke dataset (train.json / test.json)
# ---------------------------------------------------------------------------
export BENCHMARK_DATASET_ID="${BENCHMARK_DATASET_ID:-/data2/group_何向南/kang/13641-harrier/EvoChampion/data/code_smoke}"
export BENCHMARK_SPLIT=train
export BENCHMARK_EVAL_SPLIT=test
export BENCHMARK_QUESTION_KEY=question
export BENCHMARK_ANSWER_KEY=answer
export BENCHMARK_FORMAT=generic
# Local directory has no HF "config" name; empty subset avoids a failing
# name="main" first attempt in the bidirectional retry.
export BENCHMARK_SUBSET="${BENCHMARK_SUBSET:-}"

# ---------------------------------------------------------------------------
# Training: LoRA for small data (anti-forgetting, memory-friendly)
# ---------------------------------------------------------------------------
export TRAIN_FINETUNING_TYPE=lora
export BASE_MODEL_NAME=Qwen/Qwen3-0.6B
export CHAMPION_MODEL_PATH=Qwen/Qwen3-0.6B
export AGENT_BASE_MODEL_NAME=Qwen/Qwen3-0.6B
export CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED=2
# Space-efficient: save only model weights (no optimizer state) and keep just
# one checkpoint, so the small LoRA candidate + merged model fit the quota.
export TRAIN_SAVE_ONLY_MODEL=true
export TRAIN_SAVE_TOTAL_LIMIT=1
export LORA_RANK=8
export LORA_ALPHA=16

# ---------------------------------------------------------------------------
# Instruction prefix (code-oriented). prompt_designer may override this at
# runtime via the instruction_designer leaf; this is the safe default.
# ---------------------------------------------------------------------------
if [[ -z "${INSTRUCTION_PREFIX:-}" ]]; then
  export INSTRUCTION_PREFIX='请编写 Python 函数解决下面的问题，只输出一个 ```python 代码块，不要解释：\n\n'
fi

# ---------------------------------------------------------------------------
# vLLM / Ray inference (reuse the math stack; small model, modest settings)
# ---------------------------------------------------------------------------
export USE_VLLM=1
export USE_VLLM_FOR_LOCAL_CHECKPOINTS=1
export USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS=0
export VLLM_LOCAL_CHECKPOINT_SUBPROCESS_TIMEOUT_SECONDS=3600
export VLLM_MAX_CACHED_ENGINES=1
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION=0
export VLLM_ENFORCE_EAGER=0
export VLLM_ENABLE_PREFIX_CACHING=1
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
if (( VLLM_MAX_MODEL_LEN < 4096 )); then
  export VLLM_MAX_MODEL_LEN=4096
fi
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
export VLLM_RAY_INFLIGHT_PER_CALL="${VLLM_RAY_INFLIGHT_PER_CALL:-64}"
export VLLM_RAY_MAX_INFLIGHT_OUTPUT_TOKENS="${VLLM_RAY_MAX_INFLIGHT_OUTPUT_TOKENS:-102400}"
export VLLM_RAY_GENERATE_TIMEOUT_SECONDS=1800
export VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS="${VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS:-8}"
export DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR=1
export VLLM_RAY_ACTOR_NUM_GPUS=1
export VLLM_RAY_ACTOR_MAX_RESTARTS=0
export VLLM_RAY_ACTOR_MAX_TASK_RETRIES=0
export VLLM_RAY_ACTOR_MAX_CONCURRENCY="${VLLM_RAY_ACTOR_MAX_CONCURRENCY:-16}"
export VLLM_RAY_ACTOR_SHUTDOWN_TIMEOUT_SECONDS=45
export RAY_ADDRESS=local
export RAY_NAMESPACE="${RAY_NAMESPACE:-evocode-${SLURM_JOB_ID:-local}}"
export VLLM_NO_USAGE_STATS=1
export RAY_process_group_cleanup_enabled=true
export RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper=true

echo "=== Environment ==="
echo "DOMAIN: $DOMAIN"
echo "ANSWER_VERIFIER_TYPE: $ANSWER_VERIFIER_TYPE"
echo "BENCHMARK_DATASET_ID: $BENCHMARK_DATASET_ID"
echo "BENCHMARK_FORMAT: $BENCHMARK_FORMAT"
echo "CANDIDATE_MODEL_DIR: $CANDIDATE_MODEL_DIR"
echo "HF_HOME: $HF_HOME"
echo "CODE_EXEC_CACHE_DIR: $CODE_EXEC_CACHE_DIR"
echo "TRAIN_FINETUNING_TYPE: $TRAIN_FINETUNING_TYPE"
echo "FROZEN_PROBE_EVAL_METHOD: $FROZEN_PROBE_EVAL_METHOD"
echo "USE_LLM_AS_JUDGE: $USE_LLM_AS_JUDGE"
echo "MAX_ROUNDS: $MAX_ROUNDS"
echo "SEARCH_FALLBACK_DATASETS: $SEARCH_FALLBACK_DATASETS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Avoid killing unrelated Slurm jobs on shared nodes.
if [[ "${CLEAN_OWN_GPU_PROCS:-0}" == "1" ]]; then
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  pkill -9 -f "main.py" 2>/dev/null || true
  sleep 2
fi
echo "GPU memory after cleanup: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"

python -u -m py_compile \
  config/settings.py \
  src/harness.py \
  src/models/state.py \
  src/nodes/strategy_inspector.py \
  src/tools/candidate_cleanup.py \
  src/tools/strategy_policy.py \
  src/tools/code_execution.py

echo "=== Starting main.py (code domain) ==="
python -u main.py "提高 Python 代码能力" \
  --benchmark "$BENCHMARK_DATASET_ID" \
  --benchmark-split "$BENCHMARK_SPLIT" \
  --benchmark-eval-split "$BENCHMARK_EVAL_SPLIT" \
  --benchmark-question-key "$BENCHMARK_QUESTION_KEY" \
  --benchmark-answer-key "$BENCHMARK_ANSWER_KEY" \
  --benchmark-format "$BENCHMARK_FORMAT" \
  --max-rounds "$MAX_ROUNDS"

exit_code=$?
echo "=== main.py exited with code: $exit_code at $(date) ==="
exit "$exit_code"
