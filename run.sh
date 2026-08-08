#!/bin/bash
# MathBench-probed evolution on Qwen3-0.6B
# Only specifies benchmark identity — all training/evolution params decided by the system
set -eo pipefail

echo "Job started at $(date)"
echo "Host: $(hostname)"

CONDA_SH="${CONDA_SH:-/home/<USER>/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-evochampion}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

# Network / cache
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data2/<GROUP>/<USER>/EvoChampion/data/huggingface}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XET_DISABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
export REQUESTS_CONNECT_TIMEOUT=30
export REQUESTS_READ_TIMEOUT=180
export USE_HFD_DATASET_DOWNLOAD="${USE_HFD_DATASET_DOWNLOAD:-1}"
export HFD_SCRIPT_PATH="${HFD_SCRIPT_PATH:-$PWD/hfd.sh}"
export HFD_DATASET_CACHE_DIR="${HFD_DATASET_CACHE_DIR:-$HF_HOME/hfd-datasets}"
export HFD_DOWNLOAD_TOOL="${HFD_DOWNLOAD_TOOL:-aria2c}"
export HFD_DOWNLOAD_THREADS="${HFD_DOWNLOAD_THREADS:-10}"
export HFD_DOWNLOAD_JOBS="${HFD_DOWNLOAD_JOBS:-10}"
export SEARCH_TIMEOUT_SECONDS=180
export SEARCH_DATASET_REPO_LIMIT=16
export SEARCH_FALLBACK_MODE=predefined
export SEARCH_FALLBACK_DATASETS="gsm8k:main:train,openai/gsm8k:main:train,ReasoningTransferability/math_sft_40_k::train,BytedTsinghua-SIA/dapo-math-17k::train,est-ai/math-reasoning-sft::train,RedMod/math_low_medium::train,meta-math/meta_math_qa-40_k::train,microsoft/orca-math-word-problems-200k::train,nouhad/unified-math_medium::train,rasbt/math_full_minus_math500::train,Post-training-Data-Flywheel/camel-ai-math::train,MathLLMs/mm-math_instruct:MM-MathInstruct:train0,mashriram/ultra_data-math:UltraData-Math-L3-Multi-Style-Synthetic:train"

# Online formal run settings. Keep HF networking enabled, but allow cached and
# predefined datasets when the remote API is slow or unavailable.
export MAX_ROUNDS=100
export FILTER_TARGET_QUESTIONS_PER_ROUND=1000
export ROLLOUT_TIMES=5
export SCREENING_ENTRY_MAX_QUESTIONS=1000
export DATASET_WINDOW_SIZE=1000
export DATASET_PROFILE_WINDOW_SIZE=1000
export MAX_PROFILE_WINDOWS_PER_ROUND=10
export MAX_PROFILE_ITEMS_PER_ROUND=10000
export HOLDOUT_EVAL_SIZE=20
export DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS=300
export DATASET_REVIEW_ROWS_TIMEOUT_SECONDS="${DATASET_REVIEW_ROWS_TIMEOUT_SECONDS:-60}"
export DATASET_REVIEW_STREAMING_TIMEOUT_SECONDS="${DATASET_REVIEW_STREAMING_TIMEOUT_SECONDS:-60}"
export DATASET_REVIEW_SAMPLE_ROWS="${DATASET_REVIEW_SAMPLE_ROWS:-3}"
export DATASET_REVIEW_ALLOW_HFD_DURING_REVIEW="${DATASET_REVIEW_ALLOW_HFD_DURING_REVIEW:-0}"
export DATASET_REVIEW_FIRST_ACCEPT_TIMEOUT_SECONDS=300
export DATASET_REVIEW_REPLENISHMENT_WAIT_SECONDS=300
export ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-1024}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-256}"
export LLM_JUDGE_MAX_ITEMS="${LLM_JUDGE_MAX_ITEMS:-0}"
export LLM_JUDGE_BATCH_SIZE="${LLM_JUDGE_BATCH_SIZE:-64}"
export FROZEN_PROBE_EVAL_METHOD=mathbench_opencompass
export FIXED_BENCHMARK_SOURCE=mathbench_opencompass
export MATHBENCH_OPENCOMPASS_ROOT="${MATHBENCH_OPENCOMPASS_ROOT:-/home/<USER>/opencompass}"
export MATHBENCH_OPENCOMPASS_PYTHON="${MATHBENCH_OPENCOMPASS_PYTHON:-$(which python)}"
export MATHBENCH_DATASET="${MATHBENCH_DATASET:-mathbench_gen}"
export MATHBENCH_DATASET_FILTER="${MATHBENCH_DATASET_FILTER:-mathbench-arithmetic-cloze_en}"
export MATHBENCH_SUMMARIZER="${MATHBENCH_SUMMARIZER:-mathbench_v1_2024}"
export MATHBENCH_WORK_DIR="${MATHBENCH_WORK_DIR:-/data2/<GROUP>/<USER>/EvoChampion/mathbench_probe}"
export MATHBENCH_MAX_SEQ_LEN="${MATHBENCH_MAX_SEQ_LEN:-2048}"
export MATHBENCH_MAX_OUT_LEN="${MATHBENCH_MAX_OUT_LEN:-512}"
export MATHBENCH_HF_BATCH_SIZE="${MATHBENCH_HF_BATCH_SIZE:-8}"
export MATHBENCH_NUM_GPUS="${MATHBENCH_NUM_GPUS:-1}"
export MATHBENCH_NO_BATCH_PADDING="${MATHBENCH_NO_BATCH_PADDING:-0}"
export MATHBENCH_MODEL_KWARGS="${MATHBENCH_MODEL_KWARGS:-}"
export MATHBENCH_TOKENIZER_KWARGS="${MATHBENCH_TOKENIZER_KWARGS:-padding_side='left' truncation='left' use_fast=False}"
export MATHBENCH_EXTRA_ARGS="${MATHBENCH_EXTRA_ARGS:--a vllm}"
if [[ -z "${INSTRUCTION_PREFIX:-}" ]]; then
  export INSTRUCTION_PREFIX='Solve the following math problem. Put the final numeric answer in \\boxed{} at the end, for example \\boxed{42}.\n\n'
else
  export INSTRUCTION_PREFIX
fi

# Model artifacts: keep large candidate checkpoints out of /home quota.
export BASE_MODEL_NAME=Qwen/Qwen3-0.6B
export CHAMPION_MODEL_PATH=Qwen/Qwen3-0.6B
export AGENT_BASE_MODEL_NAME=Qwen/Qwen3-0.6B
export CANDIDATE_MODEL_DIR="${CANDIDATE_MODEL_DIR:-/data2/<GROUP>/<USER>/EvoChampion/candidates}"
# 0 disables retention; positive values keep current champion plus N recent rollback/prune candidates.
export CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED=2
mkdir -p "$CANDIDATE_MODEL_DIR"
mkdir -p "$HFD_DATASET_CACHE_DIR"

if [[ "$USE_HFD_DATASET_DOWNLOAD" == "1" && ! -x "$HFD_SCRIPT_PATH" ]]; then
  echo "Installing hfd.sh to $HFD_SCRIPT_PATH"
  wget -q -O "$HFD_SCRIPT_PATH" https://hf-mirror.com/hfd/hfd.sh
  chmod a+x "$HFD_SCRIPT_PATH"
fi

# Keep base and candidate inference on the shared Ray/vLLM actor path. With a
# single cached actor, model switches evict the previous engine cleanly while
# preserving vLLM continuous batching, prefix caching, and cross-worker reuse.
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
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
if (( VLLM_MAX_MODEL_LEN < 8192 )); then
  export VLLM_MAX_MODEL_LEN=8192
fi
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
export VLLM_RAY_INFLIGHT_PER_CALL="${VLLM_RAY_INFLIGHT_PER_CALL:-128}"
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
export RAY_NAMESPACE="${RAY_NAMESPACE:-evochampion-${SLURM_JOB_ID:-local}}"
export VLLM_NO_USAGE_STATS=1
export RAY_process_group_cleanup_enabled=true
export RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper=true

echo "=== Environment ==="
echo "Python: $(which python)"
echo "Conda: $CONDA_DEFAULT_ENV"
echo "HF_HOME: $HF_HOME"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "USE_HFD_DATASET_DOWNLOAD: $USE_HFD_DATASET_DOWNLOAD"
echo "HFD_SCRIPT_PATH: $HFD_SCRIPT_PATH"
echo "HFD_DATASET_CACHE_DIR: $HFD_DATASET_CACHE_DIR"
echo "HFD_DOWNLOAD_TOOL: $HFD_DOWNLOAD_TOOL"
echo "HFD_DOWNLOAD_THREADS: $HFD_DOWNLOAD_THREADS"
echo "HFD_DOWNLOAD_JOBS: $HFD_DOWNLOAD_JOBS"
echo "SEARCH_FALLBACK_MODE: $SEARCH_FALLBACK_MODE"
echo "MAX_ROUNDS: $MAX_ROUNDS"
echo "ROLLOUT_TIMES: $ROLLOUT_TIMES"
echo "ROLLOUT_MAX_NEW_TOKENS: $ROLLOUT_MAX_NEW_TOKENS"
echo "EVAL_MAX_NEW_TOKENS: $EVAL_MAX_NEW_TOKENS"
echo "INFERENCE_BATCH_SIZE: $INFERENCE_BATCH_SIZE"
echo "LLM_JUDGE_MAX_ITEMS: $LLM_JUDGE_MAX_ITEMS"
echo "LLM_JUDGE_BATCH_SIZE: $LLM_JUDGE_BATCH_SIZE"
echo "LLM_JUDGE_MAX_NEW_TOKENS: ${LLM_JUDGE_MAX_NEW_TOKENS:-unset}"
echo "FROZEN_PROBE_EVAL_METHOD: $FROZEN_PROBE_EVAL_METHOD"
echo "MATHBENCH_OPENCOMPASS_ROOT: $MATHBENCH_OPENCOMPASS_ROOT"
echo "MATHBENCH_DATASET: $MATHBENCH_DATASET"
echo "MATHBENCH_DATASET_FILTER: $MATHBENCH_DATASET_FILTER"
echo "MATHBENCH_WORK_DIR: $MATHBENCH_WORK_DIR"
echo "MATHBENCH_MODEL_KWARGS: $MATHBENCH_MODEL_KWARGS"
echo "MATHBENCH_EXTRA_ARGS: $MATHBENCH_EXTRA_ARGS"
echo "DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS: $DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS"
echo "DATASET_REVIEW_ROWS_TIMEOUT_SECONDS: $DATASET_REVIEW_ROWS_TIMEOUT_SECONDS"
echo "DATASET_REVIEW_STREAMING_TIMEOUT_SECONDS: $DATASET_REVIEW_STREAMING_TIMEOUT_SECONDS"
echo "DATASET_REVIEW_SAMPLE_ROWS: $DATASET_REVIEW_SAMPLE_ROWS"
echo "DATASET_REVIEW_ALLOW_HFD_DURING_REVIEW: $DATASET_REVIEW_ALLOW_HFD_DURING_REVIEW"
echo "USE_VLLM: $USE_VLLM"
echo "USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS: $USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS"
echo "VLLM_GPU_MEMORY_UTILIZATION: $VLLM_GPU_MEMORY_UTILIZATION"
echo "VLLM_ENFORCE_EAGER: $VLLM_ENFORCE_EAGER"
echo "VLLM_ENABLE_PREFIX_CACHING: $VLLM_ENABLE_PREFIX_CACHING"
echo "VLLM_ENABLE_CHUNKED_PREFILL: $VLLM_ENABLE_CHUNKED_PREFILL"
echo "VLLM_MAX_MODEL_LEN: $VLLM_MAX_MODEL_LEN"
echo "VLLM_MAX_NUM_SEQS: $VLLM_MAX_NUM_SEQS"
echo "VLLM_MAX_NUM_BATCHED_TOKENS: $VLLM_MAX_NUM_BATCHED_TOKENS"
echo "VLLM_RAY_INFLIGHT_PER_CALL: $VLLM_RAY_INFLIGHT_PER_CALL"
echo "VLLM_RAY_MAX_INFLIGHT_OUTPUT_TOKENS: $VLLM_RAY_MAX_INFLIGHT_OUTPUT_TOKENS"
echo "VLLM_RAY_ACTOR_MAX_CONCURRENCY: $VLLM_RAY_ACTOR_MAX_CONCURRENCY"
echo "VLLM_RAY_GENERATE_TIMEOUT_SECONDS: $VLLM_RAY_GENERATE_TIMEOUT_SECONDS"
echo "VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS: $VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS"
echo "VLLM_NO_USAGE_STATS: $VLLM_NO_USAGE_STATS"
echo "DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR: $DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR"
echo "RAY_process_group_cleanup_enabled: $RAY_process_group_cleanup_enabled"
echo "RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper: $RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Avoid killing unrelated Slurm jobs on shared nodes. Set CLEAN_OWN_GPU_PROCS=1
# only for an intentionally isolated allocation.
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
  src/tools/mathbench_probe.py \
  src/tools/strategy_policy.py

echo "=== Starting main.py ==="
python -u main.py "提高数学能力" \
  --benchmark gsm8k \
  --benchmark-subset main \
  --benchmark-question-key question \
  --benchmark-answer-key answer \
  --benchmark-format gsm8k \
  --benchmark-eval-split test \
  --max-rounds "${MAX_ROUNDS}"

exit_code=$?
echo "=== main.py exited with code: $exit_code at $(date) ==="
exit "$exit_code"
