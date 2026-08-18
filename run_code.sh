#!/bin/bash
# Code-domain self-evolution loop on Qwen3-0.6B (smoke milestone).
#
# Runs the full loop end-to-end with execution-based code judging:
#   data load → review → standardize → rollout + difficulty tagging
#   → train (LoRA) → code test evaluation → promote/rollback → next round
#
# Correctness is decided ONLY by executing candidate code against executable
# tests (src/tools/code_execution.py) — never by LLM subjective judgment.
set -eo pipefail

echo "Job started at $(date)"
echo "Host: $(hostname)"

CONDA_SH="${CONDA_SH:-/home/kang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-agentevolver}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

# Resolve paths relative to this script so any clone works (the repo ships its
# own data/code_smoke and run_job_code.sh wrapper).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Domain selector (switches data admission, rollout judging, evaluation,
#     and prompts to the code-domain execution-based loop) ──
export DOMAIN="${DOMAIN:-code}"
export ANSWER_VERIFIER_TYPE="${ANSWER_VERIFIER_TYPE:-auto}"

# ── Workspace on /data2, under this agent's own folder (13644-estuary) ──
# All large artifacts (model checkpoints, HF cache, probe work dirs) live on
# /data2, never under the /home quota. Overridable for the server's group path.
export DATA2_BASE="${DATA2_BASE:-/data2/group_何向南/kang/agents-evolve-formal-new/13644-estuary}"
export HF_HOME="${HF_HOME:-$DATA2_BASE/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XET_DISABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
export REQUESTS_CONNECT_TIMEOUT=30
export REQUESTS_READ_TIMEOUT=180

# Local code smoke dataset (ships in the repo at data/code_smoke). Defaults to
# this clone's own copy; override with BENCHMARK_DATASET_ID to point elsewhere.
export BENCHMARK_DATASET_ID="${BENCHMARK_DATASET_ID:-$SCRIPT_DIR/data/code_smoke}"
export BENCHMARK_SPLIT="${BENCHMARK_SPLIT:-train}"
export BENCHMARK_EVAL_SPLIT="${BENCHMARK_EVAL_SPLIT:-train}"
export BENCHMARK_QUESTION_KEY="${BENCHMARK_QUESTION_KEY:-question}"
export BENCHMARK_ANSWER_KEY="${BENCHMARK_ANSWER_KEY:-answer}"
export BENCHMARK_FORMAT="${BENCHMARK_FORMAT:-generic}"
export BENCHMARK_SUBSET="${BENCHMARK_SUBSET:-main}"

# Frozen probe: evaluate locally via the benchmark test split (no MathBench).
export FROZEN_PROBE_EVAL_METHOD="${FROZEN_PROBE_EVAL_METHOD:-local}"
export FIXED_BENCHMARK_SOURCE="${FIXED_BENCHMARK_SOURCE:-benchmark_eval}"

# No LLM-as-judge: code correctness is decided by execution only.
export USE_LLM_AS_JUDGE=0
export LLM_JUDGE_MAX_ITEMS=0
export LLM_JUDGE_BATCH_SIZE="${LLM_JUDGE_BATCH_SIZE:-16}"

# Smoke-scale evolution (multiple rounds so accuracy can improve over rounds).
export MAX_ROUNDS="${MAX_ROUNDS:-2}"
export FILTER_TARGET_QUESTIONS_PER_ROUND="${FILTER_TARGET_QUESTIONS_PER_ROUND:-24}"
export ROLLOUT_TIMES="${ROLLOUT_TIMES:-3}"
export SCREENING_ENTRY_MAX_QUESTIONS="${SCREENING_ENTRY_MAX_QUESTIONS:-24}"
export DATASET_PROFILE_WINDOW_SIZE="${DATASET_PROFILE_WINDOW_SIZE:-24}"
export MAX_PROFILE_WINDOWS_PER_ROUND="${MAX_PROFILE_WINDOWS_PER_ROUND:-1}"
export MAX_PROFILE_ITEMS_PER_ROUND="${MAX_PROFILE_ITEMS_PER_ROUND:-24}"
export HOLDOUT_EVAL_SIZE="${HOLDOUT_EVAL_SIZE:-8}"
# Allow small training rounds so the smoke dataset does not trigger budget
# exhaustion after the first window.
export DATA_MIN_TRAIN_QUESTIONS_PER_ROUND="${DATA_MIN_TRAIN_QUESTIONS_PER_ROUND:-4}"

# Small eval caps so the smoke loop runs with a handful of questions.
export EVAL_TEST_MAX_ITEMS="${EVAL_TEST_MAX_ITEMS:-40}"
export EVAL_COTEST_MAX_ITEMS="${EVAL_COTEST_MAX_ITEMS:-4}"
export EVAL_MASTERED_MAX_ITEMS="${EVAL_MASTERED_MAX_ITEMS:-8}"
export PROBE_EVAL_MAX_ITEMS="${PROBE_EVAL_MAX_ITEMS:-40}"
export GLOBAL_PROBE_SIZE="${GLOBAL_PROBE_SIZE:-40}"
export FROZEN_PROBE_SIZE="${FROZEN_PROBE_SIZE:-40}"

# Data split ratios (small data → bias toward train).
export TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.70}"
export COTEST_SPLIT_RATIO="${COTEST_SPLIT_RATIO:-0.05}"
export TEST_SPLIT_RATIO="${TEST_SPLIT_RATIO:-0.15}"
export PROBE_SPLIT_RATIO="${PROBE_SPLIT_RATIO:-0.05}"
export LF_VAL_SPLIT_RATIO="${LF_VAL_SPLIT_RATIO:-0.05}"
export LF_VAL_MIN_QUESTIONS="${LF_VAL_MIN_QUESTIONS:-1}"

# Training: LoRA (small data prefers LoRA per the code-domain principle).
export TRAIN_FINETUNING_TYPE="${TRAIN_FINETUNING_TYPE:-lora}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"

# Model artifacts on /data2.
export BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen3-0.6B}"
export CHAMPION_MODEL_PATH="${CHAMPION_MODEL_PATH:-Qwen/Qwen3-0.6B}"
export AGENT_BASE_MODEL_NAME="${AGENT_BASE_MODEL_NAME:-Qwen/Qwen3-0.6B}"
export CANDIDATE_MODEL_DIR="${CANDIDATE_MODEL_DIR:-$DATA2_BASE/code_candidates}"
export CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED=2
mkdir -p "$CANDIDATE_MODEL_DIR"
mkdir -p "$HF_HOME"

# Code rollout instruction prefix (the model should output executable code).
export INSTRUCTION_PREFIX="${INSTRUCTION_PREFIX:-Write a Python function that solves the following problem. Output only the code, no explanation.\n\n}"

# Code execution judge resource limits.
export CODE_EXEC_TIMEOUT_SECONDS="${CODE_EXEC_TIMEOUT_SECONDS:-10}"
export CODE_EXEC_MAX_MEMORY_MB="${CODE_EXEC_MAX_MEMORY_MB:-512}"
export CODE_EXEC_CPU_SECONDS="${CODE_EXEC_CPU_SECONDS:-15}"

# Search: keep HF networking for the searcher, but the local code benchmark is
# always offered as a training-data source (src/tools/hf_search.py) and auto-
# accepted by the reviewer with a passthrough cleaner.
export SEARCH_TIMEOUT_SECONDS=60
export SEARCH_DATASET_REPO_LIMIT="${SEARCH_DATASET_REPO_LIMIT:-8}"
export SEARCH_FALLBACK_MODE="${SEARCH_FALLBACK_MODE:-predefined}"
export SEARCH_FALLBACK_DATASETS="${SEARCH_FALLBACK_DATASETS:-$SCRIPT_DIR/data/code_smoke::train}"
export USE_HFD_DATASET_DOWNLOAD="${USE_HFD_DATASET_DOWNLOAD:-0}"

# Inference: vLLM for fast batched inference.
export ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-512}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
export DATASET_REVIEW_SAMPLE_ROWS="${DATASET_REVIEW_SAMPLE_ROWS:-3}"
export DATASET_REVIEW_ALLOW_HFD_DURING_REVIEW="${DATASET_REVIEW_ALLOW_HFD_DURING_REVIEW:-0}"

# vLLM / Ray actor config (single GPU).
export USE_VLLM=1
export USE_VLLM_FOR_LOCAL_CHECKPOINTS=1
export USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS=0
export VLLM_MAX_CACHED_ENGINES=1
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION=0
export VLLM_ENFORCE_EAGER=0
export VLLM_ENABLE_PREFIX_CACHING=1
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
export VLLM_RAY_INFLIGHT_PER_CALL="${VLLM_RAY_INFLIGHT_PER_CALL:-32}"
export VLLM_RAY_MAX_INFLIGHT_OUTPUT_TOKENS="${VLLM_RAY_MAX_INFLIGHT_OUTPUT_TOKENS:-16384}"
export VLLM_RAY_GENERATE_TIMEOUT_SECONDS=1800
export VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS="${VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS:-8}"
export DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR=1
export VLLM_RAY_ACTOR_NUM_GPUS=1
export VLLM_RAY_ACTOR_MAX_RESTARTS=0
export VLLM_RAY_ACTOR_MAX_TASK_RETRIES=0
export VLLM_RAY_ACTOR_MAX_CONCURRENCY="${VLLM_RAY_ACTOR_MAX_CONCURRENCY:-8}"
export VLLM_RAY_ACTOR_SHUTDOWN_TIMEOUT_SECONDS=45
export RAY_ADDRESS=local
export RAY_NAMESPACE="${RAY_NAMESPACE:-evochampion-code-${SLURM_JOB_ID:-local}}"
export VLLM_NO_USAGE_STATS=1

echo "=== Environment ==="
echo "Python: $(which python)"
echo "Conda: ${CONDA_DEFAULT_ENV:-none}"
echo "DOMAIN: $DOMAIN"
echo "ANSWER_VERIFIER_TYPE: $ANSWER_VERIFIER_TYPE"
echo "BENCHMARK_DATASET_ID: $BENCHMARK_DATASET_ID"
echo "BENCHMARK_FORMAT: $BENCHMARK_FORMAT"
echo "FROZEN_PROBE_EVAL_METHOD: $FROZEN_PROBE_EVAL_METHOD"
echo "USE_LLM_AS_JUDGE: $USE_LLM_AS_JUDGE"
echo "MAX_ROUNDS: $MAX_ROUNDS"
echo "TRAIN_FINETUNING_TYPE: $TRAIN_FINETUNING_TYPE"
echo "CANDIDATE_MODEL_DIR: $CANDIDATE_MODEL_DIR"
echo "HF_HOME: $HF_HOME"
echo "DATA2_BASE: $DATA2_BASE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Static check before launching.
python -u -m py_compile \
  config/settings.py \
  src/harness.py \
  src/models/state.py \
  src/tools/code_execution.py \
  src/tools/dataset_adapter.py \
  src/tools/difficulty_tagger.py \
  src/tools/agent_prompts.py \
  src/nodes/evaluator.py \
  src/nodes/bootstrap.py \
  src/nodes/prompt_designer.py \
  src/nodes/dataset_reviewer.py

echo "=== Starting main.py (code domain) ==="
python -u main.py "提高代码能力" \
  --benchmark "$BENCHMARK_DATASET_ID" \
  --benchmark-subset "$BENCHMARK_SUBSET" \
  --benchmark-question-key "$BENCHMARK_QUESTION_KEY" \
  --benchmark-answer-key "$BENCHMARK_ANSWER_KEY" \
  --benchmark-format "$BENCHMARK_FORMAT" \
  --benchmark-split "$BENCHMARK_SPLIT" \
  --benchmark-eval-split "$BENCHMARK_EVAL_SPLIT" \
  --max-rounds "$MAX_ROUNDS"

exit_code=$?
echo "=== main.py exited with code: $exit_code at $(date) ==="
exit "$exit_code"
