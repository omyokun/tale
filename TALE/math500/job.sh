#!/bin/bash
  
#SBATCH -J inference             # Job name
#SBATCH -p small                # Partition name
#SBATCH --nodes=1                # Minimum number of nodes
#SBATCH --gres=gpu:1             # Number of GPUs per node
#SBATCH --ntasks 1               # no of tasks
#SBATCH --time=96:00:00           # hh:mm:ss

# Bind the required directories and run the training script


apptainer exec --env "PYTHONUSERBASE=${MYENVS}/in-context-learning" --bind /tmpdir,/work --nv /work/conteneurs/calmip/custom_users/cuquantum_arrayfire.sif python /tmpdir/m24047nmmr/pruning/llama/math500/greedy.py
