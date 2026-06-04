#!/bin/bash
#SBATCH -J tale_bigbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${REPO_DIR}/experiments/task_search/bigbench_boolean_expressions.py"
