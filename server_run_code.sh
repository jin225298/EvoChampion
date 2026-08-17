#!/usr/bin/env bash
# Server-side helper for the code-domain self-evolution run.
# Run this ONCE after cloning the repo from GitHub into /data2. It pulls the
# latest branch, validates (dry-run), and submits the Slurm job.
#
# Usage (on the GPU server, after `ssh ustc`):
#   source /home/kang/miniconda3/etc/profile.d/conda.sh
#   conda activate agentevolver
#   # First-time clone (only once):
#   #   DEPLOY=/data2/group_何向南/kang/13645-dendrite
#   #   mkdir -p "$(dirname "$DEPLOY")"
#   #   git clone -b dendrite-13645 https://github.com/jin225298/EvoChampion.git "$DEPLOY"
#   #   cd "$DEPLOY"
#   bash server_run_code.sh            # pull + dry-run + sbatch
#   bash server_run_code.sh --dry-run  # pull + dry-run only (no sbatch)
set -Eeuo pipefail

DEPLOY="${DEPLOY:-/data2/group_何向南/kang/13645-dendrite}"
BRANCH="dendrite-13645"

cd "$DEPLOY"

echo "[server_run_code] pulling $BRANCH ..."
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

mkdir -p log artifacts

ACTION="submit"
if [[ "${1:-}" == "--dry-run" ]]; then
  ACTION="dry-run"
fi

echo "[server_run_code] dry-run validation ..."
bash run_job_code.sh --dry-run

if [[ "$ACTION" == "dry-run" ]]; then
  echo "[server_run_code] dry-run only; not submitting."
  exit 0
fi

echo "[server_run_code] submitting Slurm job ..."
JOB_OUT=$(sbatch run_job_code.sh)
echo "$JOB_OUT"
JOB_ID=$(echo "$JOB_OUT" | grep -oE '[0-9]+' | head -1)
echo ""
echo "=== Submitted job $JOB_ID ==="
echo "Monitor with:"
echo "  tail -f $DEPLOY/log/code_job_${JOB_ID}.out"
echo "  tail -f $DEPLOY/log/code_job_${JOB_ID}.err"
echo "  squeue -j $JOB_ID"
