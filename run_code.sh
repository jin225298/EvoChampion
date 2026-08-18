#!/bin/bash
# Code domain evolution — self-configuring
# End-to-end code self-evolution loop with test-execution judging
set -eo pipefail

echo "Job started at $(date)"
echo "Host: $(hostname)"

CONDA_SH="${CONDA_SH:-/home/kang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-agentevolver}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

# ── Project directory (auto-detect from script location) ──
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ── Code Domain Configuration ──
export DOMAIN=code
export ANSWER_VERIFIER_TYPE=auto

# ── Benchmark dataset (auto-detect local code_smoke) ──
# Use the code_smoke dataset in the project's data directory
export BENCHMARK_DATASET_ID="${BENCHMARK_DATASET_ID:-$PROJECT_DIR/data/code_smoke}"
export BENCHMARK_SPLIT="${BENCHMARK_SPLIT:-train}"
export BENCHMARK_EVAL_SPLIT="${BENCHMARK_EVAL_SPLIT:-test}"
export BENCHMARK_QUESTION_KEY="${BENCHMARK_QUESTION_KEY:-question}"
export BENCHMARK_ANSWER_KEY="${BENCHMARK_ANSWER_KEY:-answer}"
export BENCHMARK_FORMAT="${BENCHMARK_FORMAT:-generic}"

# ── Evaluation ──
export FROZEN_PROBE_EVAL_METHOD=local
export USE_LLM_AS_JUDGE=0
export LLM_JUDGE_MAX_ITEMS=0

# ── Evolution rounds (auto-scale: small for smoke, larger for full runs) ──
export MAX_ROUNDS="${MAX_ROUNDS:-10}"
export FILTER_TARGET_QUESTIONS_PER_ROUND="${FILTER_TARGET_QUESTIONS_PER_ROUND:-100}"
export SCREENING_ENTRY_MAX_QUESTIONS="${SCREENING_ENTRY_MAX_QUESTIONS:-100}"
export DATASET_PROFILE_WINDOW_SIZE="${DATASET_PROFILE_WINDOW_SIZE:-100}"
export MAX_PROFILE_WINDOWS_PER_ROUND="${MAX_PROFILE_WINDOWS_PER_ROUND:-10}"
export MAX_PROFILE_ITEMS_PER_ROUND="${MAX_PROFILE_ITEMS_PER_ROUND:-1000}"

# ── Training (auto: LoRA for small data, full for large) ──
export TRAIN_FINETUNING_TYPE="${TRAIN_FINETUNING_TYPE:-lora}"

# ── HF cache (auto-detect existing model caches) ──
# Priority: HF_HOME env > project data/huggingface > existing cache on /data2
detect_hf_home() {
    # 1. Check if HF_HOME is already set
    if [[ -n "${HF_HOME:-}" && -d "$HF_HOME/hub" ]]; then
        echo "$HF_HOME"
        return
    fi
    # 2. Check project-local cache
    if [[ -d "$PROJECT_DIR/data/huggingface/hub" ]]; then
        echo "$PROJECT_DIR/data/huggingface"
        return
    fi
    # 3. Search /data2 for existing model caches
    for candidate in \
        "/data2/group_何向南/kang/agents-evolve-formal-new/data/huggingface" \
        "/data2/group_何向南/kang/huggingface-cache" \
        "/data2/group_何向南/kang/EvoChampion/data/huggingface"; do
        if [[ -d "$candidate/hub" ]]; then
            echo "$candidate"
            return
        fi
    done
    # 4. Default to project-local
    echo "$PROJECT_DIR/data/huggingface"
}

export HF_HOME="$(detect_hf_home)"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XET_DISABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
# Offline mode: use cached models, don't try to download
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# ── Search (auto: use local datasets as fallback) ──
export SEARCH_TIMEOUT_SECONDS=60
export SEARCH_DATASET_REPO_LIMIT=8
export SEARCH_FALLBACK_MODE=predefined
# Auto-detect local datasets for search fallback
if [[ -d "$BENCHMARK_DATASET_ID" ]]; then
    export SEARCH_FALLBACK_DATASETS="${SEARCH_FALLBACK_DATASETS:-$BENCHMARK_DATASET_ID::train}"
else
    export SEARCH_FALLBACK_DATASETS="${SEARCH_FALLBACK_DATASETS:-openai/openai_humaneval::train}"
fi

# ── Data cleaner (auto: local for code domain) ──
export DATA_CLEANER_PROVIDER="${DATA_CLEANER_PROVIDER:-local}"

# ── Model artifacts (auto: use project's candidate dir) ──
export BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen3-0.6B}"
export CHAMPION_MODEL_PATH="${CHAMPION_MODEL_PATH:-Qwen/Qwen3-0.6B}"
export AGENT_BASE_MODEL_NAME="${AGENT_BASE_MODEL_NAME:-Qwen/Qwen3-0.6B}"
# Auto-detect candidate model directory
if [[ -n "${CANDIDATE_MODEL_DIR:-}" ]]; then
    :
elif [[ -d "/data2/group_何向南/kang/13643-ferrite/code_candidates" ]]; then
    export CANDIDATE_MODEL_DIR="/data2/group_何向南/kang/13643-ferrite/code_candidates"
else
    export CANDIDATE_MODEL_DIR="$PROJECT_DIR/code_candidates"
fi
export CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED=2
mkdir -p "$CANDIDATE_MODEL_DIR"

# ── Inference ──
export ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-512}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-32}"

# ── Code execution judge ──
export CODE_EXEC_TIMEOUT="${CODE_EXEC_TIMEOUT:-10}"
export CODE_EXEC_MEMORY_MB="${CODE_EXEC_MEMORY_MB:-512}"

# ── vLLM ──
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
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
export VLLM_RAY_ACTOR_NUM_GPUS=1
export VLLM_RAY_ACTOR_MAX_RESTARTS=0
export VLLM_RAY_ACTOR_MAX_TASK_RETRIES=0
export VLLM_RAY_ACTOR_MAX_CONCURRENCY="${VLLM_RAY_ACTOR_MAX_CONCURRENCY:-8}"
export VLLM_RAY_ACTOR_SHUTDOWN_TIMEOUT_SECONDS=45
export RAY_ADDRESS=local
export RAY_NAMESPACE="${RAY_NAMESPACE:-evochampion-code-${SLURM_JOB_ID:-local}}"
export VLLM_NO_USAGE_STATS=1
export DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR=1

# ── Instruction prefix for code ──
if [[ -z "${INSTRUCTION_PREFIX:-}" ]]; then
  export INSTRUCTION_PREFIX='请编写Python函数解决以下问题，只输出代码：\n\n'
else
  export INSTRUCTION_PREFIX
fi

# ── Logging ──
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "=== Environment ==="
echo "Python: $(which python)"
echo "Conda: $CONDA_DEFAULT_ENV"
echo "Project: $PROJECT_DIR"
echo "DOMAIN: $DOMAIN"
echo "ANSWER_VERIFIER_TYPE: $ANSWER_VERIFIER_TYPE"
echo "BENCHMARK_DATASET_ID: $BENCHMARK_DATASET_ID"
echo "BENCHMARK_FORMAT: $BENCHMARK_FORMAT"
echo "MAX_ROUNDS: $MAX_ROUNDS"
echo "TRAIN_FINETUNING_TYPE: $TRAIN_FINETUNING_TYPE"
echo "CANDIDATE_MODEL_DIR: $CANDIDATE_MODEL_DIR"
echo "HF_HOME: $HF_HOME"
echo "DATA_CLEANER_PROVIDER: $DATA_CLEANER_PROVIDER"
echo "SEARCH_FALLBACK_DATASETS: $SEARCH_FALLBACK_DATASETS"
echo "CODE_EXEC_TIMEOUT: $CODE_EXEC_TIMEOUT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ── Static check ──
python -u -m py_compile \
  config/settings.py \
  src/harness.py \
  src/models/state.py \
  src/nodes/strategy_inspector.py \
  src/tools/candidate_cleanup.py \
  src/tools/strategy_policy.py \
  src/tools/code_execution.py

echo "=== Starting main.py ==="
python -u main.py "提高代码生成能力" \
  --benchmark "$BENCHMARK_DATASET_ID" \
  --benchmark-split "$BENCHMARK_SPLIT" \
  --benchmark-eval-split "$BENCHMARK_EVAL_SPLIT" \
  --benchmark-question-key "$BENCHMARK_QUESTION_KEY" \
  --benchmark-answer-key "$BENCHMARK_ANSWER_KEY" \
  --benchmark-format "$BENCHMARK_FORMAT" \
  --max-rounds "${MAX_ROUNDS}"

exit_code=$?
echo "=== main.py exited with code: $exit_code at $(date) ==="
exit "$exit_code"
