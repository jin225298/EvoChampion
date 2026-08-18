#!/bin/bash
# Code-domain self-evolution loop on Qwen3-0.6B (LoRA).
# Judges rollouts by EXECUTING the candidate against the problem's tests
# (src/tools/code_execution.py) — no LLM-as-judge, no symbolic math verify.
# Intended for a small smoke run (1 round, ~8 questions).
set -eo pipefail

echo "Job started at $(date)"
echo "Host: $(hostname)"

CONDA_SH="${CONDA_SH:-/home/kang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-agentevolver}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

# ============================================================================
# Code domain + verifier selection
# ============================================================================
export DOMAIN=code
export ANSWER_VERIFIER_TYPE=auto          # auto -> code_exec under DOMAIN=code
export USE_LLM_AS_JUDGE=0                 # never use LLM judge for code
export LLM_JUDGE_MAX_ITEMS=0
export FROZEN_PROBE_EVAL_METHOD=local     # no mathbench/opencompass for code
export USE_GLINER=0                        # code classifier uses keyword fallback

# ============================================================================
# Code smoke dataset (local path under the project repo)
# ============================================================================
export BENCHMARK_DATASET_ID="${BENCHMARK_DATASET_ID:-/home/kang/agents-evolve-formal-new/data/code_smoke}"
export BENCHMARK_SPLIT=train
export BENCHMARK_EVAL_SPLIT=test
export BENCHMARK_QUESTION_KEY=question
export BENCHMARK_ANSWER_KEY=answer
export BENCHMARK_TEST_KEY=test
export BENCHMARK_ENTRY_POINT_KEY=entry_point
export BENCHMARK_FORMAT=generic

# ============================================================================
# Small smoke scale (1 round, 8 questions)
# ============================================================================
export MAX_ROUNDS=1
export FILTER_TARGET_QUESTIONS_PER_ROUND=8
export SCREENING_ENTRY_MAX_QUESTIONS=8
export DATASET_PROFILE_WINDOW_SIZE=8
export MAX_PROFILE_WINDOWS_PER_ROUND=1
export MAX_PROFILE_ITEMS_PER_ROUND=8
export HOLDOUT_EVAL_SIZE=8
export ROLLOUT_TIMES=4
export ROLLOUT_MAX_CONCURRENT=2

# ============================================================================
# Training: LoRA (small data, anti-forgetting)
# ============================================================================
export TRAIN_FINETUNING_TYPE=lora
export LORA_RANK=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.05
export LORA_TARGET_MODULES=q_proj,v_proj
export TRAINING_TIMEOUT_SECONDS=1800

# ============================================================================
# Model + artifacts — MUST live under /data2 in this agent's own folder.
# Folder name = execution folder name (13642-grotto). Never write to /home quota.
# ============================================================================
DATA2_BASE="${DATA2_BASE:-/data2/13642-grotto}"
export BASE_MODEL_NAME=Qwen/Qwen3-0.6B
export CHAMPION_MODEL_PATH=Qwen/Qwen3-0.6B
export AGENT_BASE_MODEL_NAME=Qwen/Qwen3-0.6B
export CANDIDATE_MODEL_DIR="${CANDIDATE_MODEL_DIR:-$DATA2_BASE/code_candidates}"
export CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED=2
mkdir -p "$CANDIDATE_MODEL_DIR"

# ============================================================================
# HF / network cache (also under /data2)
# ============================================================================
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$DATA2_BASE/hf}"
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

# Search: force the local smoke dataset (predefined fallback) so the closed loop
# trains on code_smoke/train.json without depending on the HF API. The benchmark
# (code_smoke) is used for eval via the frozen probe; the same dataset feeds
# training here via the search fallback.
export SEARCH_FALLBACK_MODE=predefined
export SEARCH_FALLBACK_DATASETS="${BENCHMARK_DATASET_ID}::train"
export SEARCH_DATASET_REPO_LIMIT=0
export SEARCH_TIMEOUT_SECONDS=30

# Review / screening — fast for the smoke set.
export DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS=120
export DATASET_REVIEW_SAMPLE_ROWS=3

# Inference scale — small model, small batch.
export ROLLOUT_MAX_NEW_TOKENS=512
export EVAL_MAX_NEW_TOKENS=512
export INFERENCE_BATCH_SIZE=32
export INFERENCE_MAX_NEW_TOKENS=256

# Code judging knobs (read by src/tools/code_execution.py).
export CODE_JUDGE_TIMEOUT_SECONDS="${CODE_JUDGE_TIMEOUT_SECONDS:-10}"
export CODE_JUDGE_MAX_WORKERS="${CODE_JUDGE_MAX_WORKERS:-8}"
export CODE_JUDGE_CPU_SECONDS="${CODE_JUDGE_CPU_SECONDS:-30}"

# Instruction prefix for code: ask for a bare function, no prose.
if [[ -z "${INSTRUCTION_PREFIX:-}" ]]; then
  export INSTRUCTION_PREFIX='Write a Python function that solves the following problem. Output only the function definition (starting with "def"), no explanation, no markdown fences.\n\n'
else
  export INSTRUCTION_PREFIX
fi

# ============================================================================
# vLLM / Ray inference (shared actor path, like run.sh)
# ============================================================================
export USE_VLLM=1
export USE_VLLM_FOR_LOCAL_CHECKPOINTS=1
export USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS=0
export VLLM_MAX_CACHED_ENGINES=1
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION=0
export VLLM_ENFORCE_EAGER=0
export VLLM_ENABLE_PREFIX_CACHING=1
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
export VLLM_RAY_INFLIGHT_PER_CALL="${VLLM_RAY_INFLIGHT_PER_CALL:-32}"
export VLLM_RAY_GENERATE_TIMEOUT_SECONDS=900
export VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS="${VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS:-10}"
export DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR=1
export VLLM_RAY_ACTOR_NUM_GPUS=1
export VLLM_RAY_ACTOR_MAX_RESTARTS=0
export VLLM_RAY_ACTOR_MAX_TASK_RETRIES=0
export VLLM_RAY_ACTOR_MAX_CONCURRENCY="${VLLM_RAY_ACTOR_MAX_CONCURRENCY:-8}"
export RAY_ADDRESS=local
export RAY_NAMESPACE="${RAY_NAMESPACE:-evochampion-code-${SLURM_JOB_ID:-local}}"
export VLLM_NO_USAGE_STATS=1
export RAY_process_group_cleanup_enabled=true
export RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper=true

echo "=== Environment (code domain) ==="
echo "Python: $(which python)"
echo "Conda: ${CONDA_DEFAULT_ENV:-none}"
echo "DOMAIN: $DOMAIN  ANSWER_VERIFIER_TYPE: $ANSWER_VERIFIER_TYPE"
echo "BENCHMARK_DATASET_ID: $BENCHMARK_DATASET_ID"
echo "BENCHMARK_SPLIT: $BENCHMARK_SPLIT  EVAL_SPLIT: $BENCHMARK_EVAL_SPLIT"
echo "BENCHMARK_QUESTION_KEY: $BENCHMARK_QUESTION_KEY  ANSWER_KEY: $BENCHMARK_ANSWER_KEY"
echo "BENCHMARK_TEST_KEY: $BENCHMARK_TEST_KEY  ENTRY_POINT_KEY: $BENCHMARK_ENTRY_POINT_KEY"
echo "TRAIN_FINETUNING_TYPE: $TRAIN_FINETUNING_TYPE"
echo "CANDIDATE_MODEL_DIR: $CANDIDATE_MODEL_DIR"
echo "HF_HOME: $HF_HOME"
echo "MAX_ROUNDS: $MAX_ROUNDS"
echo "ROLLOUT_TIMES: $ROLLOUT_TIMES"
echo "FROZEN_PROBE_EVAL_METHOD: $FROZEN_PROBE_EVAL_METHOD"
echo "USE_LLM_AS_JUDGE: $USE_LLM_AS_JUDGE  LLM_JUDGE_MAX_ITEMS: $LLM_JUDGE_MAX_ITEMS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Static check before launching.
python -u -m py_compile \
  config/settings.py \
  src/harness.py \
  src/models/state.py \
  src/tools/code_execution.py \
  src/tools/model_runner.py \
  src/nodes/evaluator.py \
  src/nodes/strategy_inspector.py

echo "=== Starting main.py (code domain) ==="
python -u main.py "提升代码能力" \
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
