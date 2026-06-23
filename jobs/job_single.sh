#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: single-GPU greedy layer pruning
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=tale_prune_single
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/prune_single_%j.out
#SBATCH --error=logs/prune_single_%j.err

# ─── Edit these variables ──────────────────────────────────────────────────
MODEL_PATH="/path/to/model"          # Local path or HuggingFace model ID
DATASET="arc_challenge"              # See greedy_prune.py --help for options
DATA_PATH="/path/to/dataset.csv"     # Local CSV file
EVAL_MODE="custom"                   # 'custom' or 'lmeval'
THRESHOLD=0.08                       # Max accuracy drop (0.08 = 8%)
LIMIT=290                            # Max samples per eval (set to 0 for all)
OUTPUT="pruning_results_single.json"

# ─── Container / environment ──────────────────────────────────────────────
CONTAINER="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"
PYTHON_ENV="/path/to/python/env"     # PYTHONUSERBASE or conda env path
CODE_DIR="/path/to/TALE-pruning"     # Path to this repository

# ─── Sanity check ─────────────────────────────────────────────────────────
mkdir -p logs
echo "====================================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU          : $(nvidia-smi --list-gpus | head -1)"
echo "Model        : $MODEL_PATH"
echo "Dataset      : $DATASET"
echo "Eval mode    : $EVAL_MODE"
echo "Threshold    : $THRESHOLD"
echo "Limit        : $LIMIT"
echo "====================================================="

# ─── Run ──────────────────────────────────────────────────────────────────
apptainer exec \
    --env PYTHONUSERBASE="$PYTHON_ENV" \
    --bind /tmpdir,/work \
    --nv \
    "$CONTAINER" \
    python "$CODE_DIR/greedy_prune.py" \
        --model-path  "$MODEL_PATH" \
        --dataset     "$DATASET" \
        --data-path   "$DATA_PATH" \
        --eval        "$EVAL_MODE" \
        --threshold   "$THRESHOLD" \
        --limit       "$LIMIT" \
        --output      "$OUTPUT"

echo "Job finished with exit code $?"
