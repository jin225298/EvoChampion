#!/bin/bash
# Code-domain self-evolution smoke run on Qwen3-0.6B.
# Implements the end-to-end code loop: data load -> review -> standardize ->
# rollout difficulty (by test execution) -> train (LoRA) -> code-test eval ->
# promote/rollback -> next round.
#
# Judging is execution-based (src/tools/code_execution.py): candidate code is
# run against the dataset's test in a sandboxed subprocess; 0/1 is decided only
# by test pass/fail, never by LLM subjective equivalence.
set -eo pipefail

echo "Job started at $(date)"
echo "Host: $(hostname)"

# ---------------------------------------------------------------------------
# Environment (per task spec — code domain, server: agentevolver conda env)
# ---------------------------------------------------------------------------
CONDA_SH="${CONDA_SH:-/home/kang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-agentevolver}"
# Deployment name = local execution folder name (per operator clarification).
# Code is cloned from GitHub into the agent's own /data2 folder (never /home).
DEPLOY_NAME="${DEPLOY_NAME:-13645-dendrite}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/data2/group_何向南/kang/${DEPLOY_NAME}}"
PROJECT_DIR="${PROJECT_DIR:-${DEPLOY_ROOT}}"
DATA2_ROOT="${DATA2_ROOT:-${DEPLOY_ROOT}}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

# ---------------------------------------------------------------------------
# Code-domain pipeline settings (hard requirements from the task)
# ---------------------------------------------------------------------------
export DOMAIN=code
export ANSWER_VERIFIER_TYPE=auto
export BENCHMARK_DATASET_ID="${PROJECT_DIR}/data/humaneval"
export BENCHMARK_SPLIT=train
export BENCHMARK_EVAL_SPLIT=test
export BENCHMARK_QUESTION_KEY=question
export BENCHMARK_ANSWER_KEY=answer
export BENCHMARK_FORMAT=generic
export FROZEN_PROBE_EVAL_METHOD=local
export USE_LLM_AS_JUDGE=0
export LLM_JUDGE_MAX_ITEMS=0
export MAX_ROUNDS=1
export FILTER_TARGET_QUESTIONS_PER_ROUND=8
export SCREENING_ENTRY_MAX_QUESTIONS=8
export DATASET_PROFILE_WINDOW_SIZE=8
export MAX_PROFILE_WINDOWS_PER_ROUND=1
export MAX_PROFILE_ITEMS_PER_ROUND=8
export TRAIN_FINETUNING_TYPE=lora
# Candidate models / caches live under the agent's own /data2 folder (never /home).
export CANDIDATE_MODEL_DIR="${DATA2_ROOT}/code_candidates"
export HF_HOME="${HF_HOME:-${DATA2_ROOT}/huggingface}"

# Code-execution judging parameters.
export CODE_JUDGE_TIMEOUT_SECONDS="${CODE_JUDGE_TIMEOUT_SECONDS:-10}"
export CODE_JUDGE_MEMORY_MB="${CODE_JUDGE_MEMORY_MB:-512}"
export CODE_JUDGE_CPU_SECONDS="${CODE_JUDGE_CPU_SECONDS:-15}"
export CODE_JUDGE_MAX_WORKERS="${CODE_JUDGE_MAX_WORKERS:-8}"

# ---------------------------------------------------------------------------
# HF / cache layout (all under /data2)
# ---------------------------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
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

# Code smoke dataset is local; the search short-circuits to it (DOMAIN=code).
# USE_HFD=1 routes the reviewer through load_hf_dataset_with_fallback, which
# loads the local {split}.json via _load_local_json_split (no hfd download for
# local paths — that path is tried first).
export USE_HFD_DATASET_DOWNLOAD="${USE_HFD_DATASET_DOWNLOAD:-1}"
export SEARCH_FALLBACK_MODE=empty
export SEARCH_DATASET_REPO_LIMIT=8

# Instruction prefix for code tasks. HumanEval is completion-based (the model
# sees the function signature and generates the body), so no instruction prefix
# is prepended. Override with INSTRUCTION_PREFIX env var if needed.
if [[ -z "${INSTRUCTION_PREFIX:-}" ]]; then
  export INSTRUCTION_PREFIX=""
else
  export INSTRUCTION_PREFIX
fi

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
export BASE_MODEL_NAME=Qwen/Qwen3-0.6B
export CHAMPION_MODEL_PATH=Qwen/Qwen3-0.6B
export AGENT_BASE_MODEL_NAME=Qwen/Qwen3-0.6B
export CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED=2
mkdir -p "$CANDIDATE_MODEL_DIR"
mkdir -p "$HF_HOME"

# Small rollout/eval token budgets for the smoke run.
export ROLLOUT_TIMES="${ROLLOUT_TIMES:-1}"
export ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-512}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
export LLM_JUDGE_BATCH_SIZE="${LLM_JUDGE_BATCH_SIZE:-64}"
export FROZEN_PROBE_EVAL_METHOD=local
export HOLDOUT_EVAL_SIZE="${HOLDOUT_EVAL_SIZE:-8}"

# ---------------------------------------------------------------------------
# vLLM / Ray (single GPU, shared actor) — adapted from run.sh
# ---------------------------------------------------------------------------
export USE_VLLM=1
export USE_VLLM_FOR_LOCAL_CHECKPOINTS=1
export USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS=0
export VLLM_LOCAL_CHECKPOINT_SUBPROCESS_TIMEOUT_SECONDS=3600
export VLLM_MAX_CACHED_ENGINES=1
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION=0
export VLLM_ENFORCE_EAGER=0
export VLLM_ENABLE_PREFIX_CACHING=1
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
export VLLM_RAY_INFLIGHT_PER_CALL="${VLLM_RAY_INFLIGHT_PER_CALL:-32}"
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
export RAY_process_group_cleanup_enabled=true
export RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper=true

echo "=== Environment (code domain) ==="
echo "Python: $(which python)"
echo "Conda: $CONDA_DEFAULT_ENV"
echo "DOMAIN: $DOMAIN  ANSWER_VERIFIER_TYPE: $ANSWER_VERIFIER_TYPE"
echo "BENCHMARK_DATASET_ID: $BENCHMARK_DATASET_ID"
echo "CANDIDATE_MODEL_DIR: $CANDIDATE_MODEL_DIR"
echo "HF_HOME: $HF_HOME"
echo "MAX_ROUNDS: $MAX_ROUNDS  TRAIN_FINETUNING_TYPE: $TRAIN_FINETUNING_TYPE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [[ "${CLEAN_OWN_GPU_PROCS:-0}" == "1" ]]; then
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  pkill -9 -f "main.py" 2>/dev/null || true
  sleep 2
fi

# Static check of the code-domain modules before running.
python -u -m py_compile \
  config/settings.py \
  src/tools/code_execution.py \
  src/tools/difficulty_tagger.py \
  src/tools/dataset_adapter.py \
  src/tools/agent_prompts.py \
  src/nodes/evaluator.py \
  src/harness.py

echo "=== Starting main.py (code self-evolution) ==="
python -u main.py "提高代码能力" \
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
