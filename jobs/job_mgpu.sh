#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job: multi-node / multi-GPU greedy layer pruning
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=tale_prune_mgpu
#SBATCH --partition=small
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks=8                   # Total tasks = nodes × gpus-per-node
#SBATCH --ntasks-per-node=2          # Tasks per node
#SBATCH --cpus-per-task=8            # CPU cores per task
#SBATCH --gres=gpu:2                 # GPUs per node
#SBATCH --time=48:00:00
#SBATCH --output=logs/prune_mgpu_%j.out
#SBATCH --error=logs/prune_mgpu_%j.err

# ─── Edit these variables ──────────────────────────────────────────────────
MODEL_PATH="/path/to/model"          # Local path or HuggingFace model ID
DATASET="bigbench"                   # See greedy_prune_mgpu.py --help
DATA_PATH="/path/to/dataset.csv"     # Local CSV file
THRESHOLD=0.08                       # Max accuracy drop
LIMIT=250                            # Max samples per eval call
NODES=4                              # Must match --nodes above
GPUS_PER_NODE=2                      # Must match --ntasks-per-node above
OUTPUT="pruning_results_mgpu.json"

# ─── Container / environment ──────────────────────────────────────────────
CONTAINER="/work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif"
PYTHON_ENV="/path/to/python/env"     # PYTHONUSERBASE or conda env path
CODE_DIR="/path/to/TALE-pruning"     # Path to this repository

# ─── Distributed environment variables ────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(( SLURM_NNODES * GPUS_PER_NODE ))
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1

# NCCL tuning
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_TIMEOUT=1800000
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# ─── Sanity check ─────────────────────────────────────────────────────────
mkdir -p logs
echo "====================================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Nodes        : $SLURM_JOB_NUM_NODES"
echo "Total tasks  : $SLURM_NTASKS"
echo "Master node  : $MASTER_ADDR:$MASTER_PORT"
echo "Node list    : $SLURM_JOB_NODELIST"
echo "Model        : $MODEL_PATH"
echo "Dataset      : $DATASET"
echo "Threshold    : $THRESHOLD"
echo "Limit        : $LIMIT"
echo "====================================================="

srun nvidia-smi --list-gpus

# ─── Run ──────────────────────────────────────────────────────────────────
srun apptainer exec \
    --env PYTHONUSERBASE="$PYTHON_ENV" \
    --env MASTER_ADDR="$MASTER_ADDR" \
    --env MASTER_PORT="$MASTER_PORT" \
    --env WORLD_SIZE="$WORLD_SIZE" \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --bind /tmpdir,/work \
    --nv \
    "$CONTAINER" \
    python "$CODE_DIR/greedy_prune_mgpu.py" \
        --model-path    "$MODEL_PATH" \
        --dataset       "$DATASET" \
        --data-path     "$DATA_PATH" \
        --threshold     "$THRESHOLD" \
        --limit         "$LIMIT" \
        --nodes         "$NODES" \
        --gpus-per-node "$GPUS_PER_NODE" \
        --output        "$OUTPUT"

echo "Job finished with exit code $?"
