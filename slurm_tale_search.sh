#!/bin/bash
#SBATCH --job-name=tale_search
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=tale_search_%j.log
#SBATCH --error=tale_search_%j.err
#SBATCH --hint=nomultithread

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a Hugging Face id or local model path.}"
: "${TASK:?Set TASK, for example arc_easy.}"
: "${DATA_PATH:?Set DATA_PATH to the prepared task CSV.}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/tale_search/${TASK}}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/job_${JOB_ID}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${RUN_OUTPUT_DIR}"

echo "Job name: ${SLURM_JOB_NAME:-tale_search}"
echo "Job id: ${JOB_ID}"
echo "Mode: tale_search"
echo "Nodes: ${SLURM_NNODES:-1}"
echo "Task: ${TASK}"
echo "Model path: ${MODEL_PATH}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${RUN_OUTPUT_DIR}"
echo "Repository dir: ${REPO_DIR}"

if [[ -n "${MODULES:-}" ]]; then
  module purge
  for module_name in ${MODULES}; do
    module load "${module_name}"
  done
fi

CMD=(
  "${PYTHON_BIN}"
  -m tale.search
  --model "${MODEL_PATH}"
  --task "${TASK}"
  --data-path "${DATA_PATH}"
  --output-dir "${RUN_OUTPUT_DIR}"
)

if [[ -n "${NUM_SAMPLES:-}" ]]; then
  CMD+=(--num-samples "${NUM_SAMPLES}")
fi

if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  CMD+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi

if [[ -n "${EPSILON:-}" ]]; then
  CMD+=(--epsilon "${EPSILON}")
fi

if [[ -n "${MAX_DROP_LAYERS:-}" ]]; then
  CMD+=(--max-drop-layers "${MAX_DROP_LAYERS}")
fi

if [[ -n "${INITIAL_DROP_LAYERS_0IDX:-}" ]]; then
  CMD+=(--initial-drop-layers-0idx "${INITIAL_DROP_LAYERS_0IDX}")
fi

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi

LAUNCH_CMD=(srun)
if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
  BIND_PATHS="${BIND_PATHS:-${REPO_DIR}:${REPO_DIR}}"
  LAUNCH_CMD+=(apptainer exec --nv --bind "${BIND_PATHS}" --env PYTHONPATH="${PYTHONPATH}" "${CONTAINER_IMAGE}")
fi
LAUNCH_CMD+=("${CMD[@]}")

printf "Arguments:"
printf " %q" "${CMD[@]}"
printf "\n"

{
  printf "#!/bin/bash\n"
  printf "cd %q\n" "${REPO_DIR}"
  printf "export PYTHONPATH=%q\n" "${PYTHONPATH}"
  printf "%q " "${LAUNCH_CMD[@]}"
  printf "\n"
} > "${RUN_OUTPUT_DIR}/command.sh"

"${LAUNCH_CMD[@]}"
