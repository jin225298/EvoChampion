#!/usr/bin/env bash
#SBATCH --partition=L40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=evochampion-code
#SBATCH --output=/data2/group_何向南/kang/13644-estuary/log/code_job_%j.out
#SBATCH --error=/data2/group_何向南/kang/13644-estuary/log/code_job_%j.err
#SBATCH --mem=128G

set -Eeuo pipefail

# Code-domain smoke run. Wraps run_code.sh with the same dry-run / sbatch
# contract as run_job.sh. PROJECT_DIR defaults to the Slurm submit dir (the
# deployment dir when sbatch is invoked from the clone) or this script's dir;
# override for a different deployment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"
ENV_FILE="${PROJECT_DIR}/.env"
RUN_SCRIPT="${PROJECT_DIR}/run_code.sh"
CONDA_SH="${CONDA_SH:-/home/kang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-agentevolver}"

usage() {
  cat <<USAGE
Usage:
  sbatch ${0##*/}          Submit and run the code smoke loop on a Slurm GPU node
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

# Export variables from .env for Python/config loaders when present.
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

# Cluster-reachable proxy for Hugging Face / dataset search APIs on GPU nodes.
# Overridable; defaults to the shared cluster proxy used by run_job.sh.
export http_proxy="${http_proxy:-http://10.0.0.1:1081}"
export https_proxy="${https_proxy:-http://10.0.0.1:1081}"
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
PYTHON="${PYTHON:-$(command -v python || command -v python3 || true)}"
[[ -n "$PYTHON" ]] || fail "python is not available after environment setup"

bash -n "$RUN_SCRIPT"
"$PYTHON" -m py_compile main.py config/settings.py src/harness.py src/models/state.py \
  src/tools/code_execution.py src/tools/dataset_adapter.py src/tools/difficulty_tagger.py \
  src/tools/agent_prompts.py src/nodes/evaluator.py src/nodes/bootstrap.py \
  src/nodes/prompt_designer.py src/nodes/dataset_reviewer.py

log "project: $PROJECT_DIR"
log "host: $(hostname)"
log "python: $PYTHON"
log "conda env: ${CONDA_DEFAULT_ENV:-none}"
log "http_proxy=${http_proxy:-unset}"
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
