#!/usr/bin/env bash
# Server setup for the code-domain self-evolution smoke run.
# Run on the GPU server (ustc) after `git pull grotto-13642`.
# Usage: bash scripts/setup_code_server.sh
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kang/agents-evolve-formal-new}"
cd "$PROJECT_DIR"

echo "=== 1. Update code (grotto-13642 branch) ==="
git fetch origin
git checkout grotto-13642 2>/dev/null || git checkout -b grotto-13642 origin/grotto-13642
git pull origin grotto-13642

echo "=== 2. Install run scripts (gitignored names) ==="
cp scripts/code_run.sh run_code.sh
cp scripts/code_job.sh run_job_code.sh
chmod +x run_code.sh run_job_code.sh

echo "=== 3. Ensure code_smoke dataset is present (should come from git) ==="
if [[ ! -f data/code_smoke/train.json ]]; then
  echo "WARNING: data/code_smoke/train.json not found. Creating minimal placeholder."
  mkdir -p data/code_smoke
  cat > data/code_smoke/train.json <<'JSON'
[
  {"question":"Write a function add(a, b) that returns a + b.","answer":"def add(a, b):\n    return a + b","test":"def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0","entry_point":"add"}
]
JSON
  cat > data/code_smoke/test.json <<'JSON'
[
  {"question":"Write a function subtract(a, b) that returns a - b.","answer":"def subtract(a, b):\n    return a - b","test":"def check(candidate):\n    assert candidate(5, 3) == 2\n    assert candidate(0, 5) == -5","entry_point":"subtract"}
]
JSON
fi
echo "  train.json: $(python3 -c "import json;print(len(json.load(open('data/code_smoke/train.json'))))" 2>/dev/null || echo '?') items"
echo "  test.json:  $(python3 -c "import json;print(len(json.load(open('data/code_smoke/test.json'))))" 2>/dev/null || echo '?') items"

echo "=== 4. Create /data2 workspace (this agent's own folder) ==="
mkdir -p /data2/13642-grotto/code_candidates /data2/13642-grotto/hf

echo "=== 5. Activate conda ==="
source /home/kang/miniconda3/etc/profile.d/conda.sh
conda activate agentevolver
python --version
which llamafactory-cli || echo "WARNING: llamafactory-cli not in PATH"

echo "=== 6. Static check (py_compile) ==="
python -m py_compile \
  main.py config/settings.py \
  src/harness.py src/models/state.py \
  src/tools/code_execution.py src/tools/model_runner.py \
  src/nodes/evaluator.py src/nodes/strategy_inspector.py \
  src/tools/strategy_policy.py
echo "  py_compile OK"

echo "=== 7. Run code_execution unit tests (no GPU needed) ==="
python tests/test_code_execution.py

echo "=== 8. Dry-run the Slurm job ==="
bash run_job_code.sh --dry-run

echo ""
echo "=== Setup complete. Next steps: ==="
echo "  sbatch run_job_code.sh"
echo "  tail -f log/code_job_*.out"
echo "  tail -f log/code_job_*.err"
