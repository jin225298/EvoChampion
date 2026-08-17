#!/usr/bin/env bash
#SBATCH --partition=L40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=evochampion-code
#SBATCH --output=/data2/group_何向南/kang/13645-dendrite/log/code_job_%j.out
#SBATCH --error=/data2/group_何向南/kang/13645-dendrite/log/code_job_%j.err
#SBATCH --mem=128G

set -Eeuo pipefail

# Deployment name = local execution folder name (per operator clarification).
# Code is cloned from GitHub into the agent's own /data2 folder (never /home).
DEPLOY_NAME="${DEPLOY_NAME:-13645-dendrite}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/data2/group_何向南/kang/${DEPLOY_NAME}}"
PROJECT_DIR="${PROJECT_DIR:-${DEPLOY_ROOT}}"
ENV_FILE="${PROJECT_DIR}/.env"
RUN_SCRIPT="${PROJECT_DIR}/run_code.sh"
CONDA_SH="/home/kang/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="agentevolver"

usage() {
  cat <<USAGE
Usage:
  sbatch ${0##*/}          Submit and run the code-domain loop on a Slurm GPU node
  bash ${0##*/} --dry-run  Validate paths, environment, and syntax only
  bash ${0##*/}            Run directly in the current shell environment
USAGE
}

log() {
  printf '[run_job_code] %s\n' "$*"
}

fail() {
  printf '[run_job_code] ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  printf '[run_job_code] FAILED at line %s with exit code %s\n' "${BASH_LINENO[0]}" "$exit_code" >&2
  exit "$exit_code"
}
trap on_error ERR

DRY_RUN=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  usage >&2
  fail "unknown argument: $1"
fi

[[ -d "$PROJECT_DIR" ]] || fail "project directory not found: $PROJECT_DIR"
cd "$PROJECT_DIR"
mkdir -p log artifacts

[[ -f "$RUN_SCRIPT" ]] || fail "run script not found: $RUN_SCRIPT"
[[ -r "$RUN_SCRIPT" ]] || fail "run script is not readable: $RUN_SCRIPT"

# Export variables from .env when present (optional; the code run is self-contained).
if [[ -f "$ENV_FILE" ]]; then
  [[ -r "$ENV_FILE" ]] || fail "environment file is not readable: $ENV_FILE"
  set +u
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  set -u
else
  log "environment file not found at $ENV_FILE; using script defaults"
fi

# Cluster-reachable proxy so GPU nodes can reach Hugging Face / dataset APIs.
export http_proxy="${http_proxy:-http://<PROXY_IP>:1081}"
export https_proxy="${https_proxy:-http://<PROXY_IP>:1081}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export PYTHONUNBUFFERED=1

if [[ -f "$CONDA_SH" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV" || fail "failed to activate conda env: $CONDA_ENV"
  fi
  set -u
else
  log "Conda profile not found at $CONDA_SH; using current PATH"
fi

command -v bash >/dev/null 2>&1 || fail "bash is not available"
command -v python >/dev/null 2>&1 || fail "python is not available after environment setup"

# Syntax + compile checks (the same gate run.sh performs, plus the new code modules).
bash -n "$RUN_SCRIPT"
python -m py_compile \
  main.py \
  config/settings.py \
  src/harness.py \
  src/models/state.py \
  src/tools/code_execution.py \
  src/tools/difficulty_tagger.py \
  src/tools/dataset_adapter.py \
  src/tools/agent_prompts.py \
  src/nodes/evaluator.py

log "project: $PROJECT_DIR"
log "host: $(hostname)"
log "python: $(command -v python)"
log "conda env: ${CONDA_DEFAULT_ENV:-none}"
log "DOMAIN: ${DOMAIN:-unset}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  log "slurm job: $SLURM_JOB_ID"
else
  log "not running inside a Slurm allocation"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "dry-run checks passed"
  exit 0
fi

exec bash "$RUN_SCRIPT"
