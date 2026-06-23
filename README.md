# TALE — Greedy Layer Pruning for LLMs

Greedy algorithm that finds transformer layers that can be removed from a
causal language model while keeping accuracy within a user-specified threshold
of the full-model baseline.

Supports **4 models**, **9 datasets**, and **2 evaluation backends** out of the
box.  Both single-GPU and multi-node/multi-GPU (SLURM + NCCL) modes are
provided.

---

## Repository layout

```
TALE-pruning/
├── greedy_prune.py          # Single-GPU entry point
├── greedy_prune_mgpu.py     # Multi-node/multi-GPU entry point (SLURM)
├── src/
│   ├── model.py             # ModifiedModel wrapper + in-place patch helpers
│   ├── datasets.py          # Dataset loaders (CSV → dicts)
│   └── evaluate.py          # Custom eval, lm-eval wrapper, greedy algorithm
├── jobs/
│   ├── job_single.sh        # SLURM job template (1 GPU)
│   └── job_mgpu.sh          # SLURM job template (multi-node, multi-GPU)
└── requirements.txt
```

---

## 1. Installation

```bash
# Clone / copy the repository
cd TALE-pruning

# Install dependencies
pip install -r requirements.txt

# (Optional) lm-evaluation-harness — only needed for --eval lmeval
pip install lm-eval
```

---

## 2. Supported models

Pass a HuggingFace model ID, a local path, or one of the built-in shortcuts:

| Shortcut  | Model                                      |
|-----------|--------------------------------------------|
| `llama`   | `meta-llama/Llama-3.1-8B-Instruct`         |
| `qwen`    | `Qwen/Qwen2.5-0.5B-Instruct`              |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3`       |
| `lucie`   | `OpenLLM-France/Lucie-7B-Instruct`         |

Any other Llama-style model (same architecture: `model.embed_tokens` /
`model.layers` / `model.norm` / `lm_head`) will also work.

---

## 3. Supported datasets

| `--dataset`     | Evaluation type | Expected CSV columns                                   |
|-----------------|-----------------|--------------------------------------------------------|
| `arc_challenge` | custom / lmeval | `question`, `choice_label`, `choice_text`, `answer_key` |
| `arc_easy`      | custom / lmeval | same as arc_challenge                                  |
| `boolq`         | custom / lmeval | `question`, `passage`, `answer_label`                  |
| `bigbench`      | custom / lmeval | `input`, `target`                                      |
| `common_qa`     | custom / lmeval | `question`, `choice_A`…`choice_E`, `answer_key`        |
| `gsm8k`         | custom / lmeval | `input`, `target`                                      |
| `mmlu`          | custom / lmeval | `question`, `choice_A`…`choice_D`, `answer`, `subject` |
| `winogrande`    | custom / lmeval | `sentence`, `option1`, `option2`, `answer`             |
| `math500`       | custom only     | `problem` (or `input`), `solution` (or `target`)       |

---

## 4. Single-GPU usage

### Custom evaluation (fastest, no extra dependencies)

```bash
python greedy_prune.py \
    --model-path llama \
    --dataset arc_challenge \
    --data-path /data/arc_challenge_validation.csv \
    --eval custom \
    --threshold 0.08 \
    --limit 290 \
    --output results_arc_challenge.json
```

### lm-evaluation-harness evaluation (standardised benchmarks)

```bash
python greedy_prune.py \
    --model-path mistral \
    --dataset mmlu \
    --eval lmeval \
    --threshold 0.05 \
    --limit 200 \
    --output results_mmlu.json
```

### All arguments

| Argument        | Default                    | Description                                      |
|-----------------|----------------------------|--------------------------------------------------|
| `--model-path`  | *(required)*               | HuggingFace ID, local path, or shortcut name     |
| `--dataset`     | *(required)*               | Dataset name (see table above)                   |
| `--data-path`   | `None`                     | Path to CSV (required for `--eval custom`)       |
| `--eval`        | `custom`                   | `custom` or `lmeval`                             |
| `--num-fewshot` | `0`                        | Few-shot examples (lmeval only)                  |
| `--threshold`   | `0.08`                     | Max accuracy drop allowed (8 percentage points)  |
| `--limit`       | `None` (all)               | Max samples per evaluation call                  |
| `--output`      | `pruning_results.json`     | Where to save results                            |

---

## 5. Multi-GPU usage (SLURM)

### Direct launch via srun

```bash
srun --nodes=4 --ntasks=8 --ntasks-per-node=2 --gres=gpu:2 \
    python greedy_prune_mgpu.py \
        --model-path llama \
        --dataset bigbench \
        --data-path /data/bigbench_boolean_expressions.csv \
        --threshold 0.08 \
        --limit 250 \
        --nodes 4 \
        --gpus-per-node 2 \
        --output results_bigbench_mgpu.json
```

### Via SLURM job script

1. Edit `jobs/job_mgpu.sh` — set `MODEL_PATH`, `DATASET`, `DATA_PATH`,
   `CONTAINER`, `PYTHON_ENV`, and `CODE_DIR`.
2. Submit:

```bash
sbatch jobs/job_mgpu.sh
```

### Multi-GPU arguments (additional)

| Argument          | Default | Description                                    |
|-------------------|---------|------------------------------------------------|
| `--nodes`         | `4`     | Number of SLURM nodes                          |
| `--gpus-per-node` | `2`     | GPUs per node                                  |

The total number of processes is `nodes × gpus-per-node`.  SLURM environment
variables (`SLURM_PROCID`, `SLURM_NTASKS`, etc.) are read automatically.

---

## 6. Output format

Both scripts write a JSON file:

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "arc_challenge",
  "eval_mode": "custom",
  "threshold": 0.08,
  "limit": 290,
  "baseline_acc": 0.7241,
  "dropped_layers": [14, 22, 27],
  "layers_kept": 29,
  "total_layers": 32,
  "compression": "9.4%",
  "iterations": {
    "iteration_1": {
      "best_layer": 14,
      "best_accuracy": 0.7207,
      "dropped_so_far": [14]
    }
  }
}
```

---

## 7. How the algorithm works

1. **Baseline** — evaluate the full model (all layers kept).
2. **Threshold** — set the minimum acceptable accuracy to `baseline − threshold`.
3. **Iterate**:
   - For every remaining layer, evaluate the model with that layer removed.
   - Permanently drop the layer whose removal yields the highest accuracy,
     provided it is still above the threshold.
4. **Stop** when no single-layer removal keeps accuracy above the threshold.

The algorithm is greedy (not globally optimal), but finds good layer-drop
sets quickly.  A threshold of `0.08` (8 pp) typically removes 10–25% of
layers with negligible real-world impact.

---

## 8. Adding a new dataset

1. Add a loader function in `src/datasets.py` that returns the standard dict
   (`items`, `format_chat`, `get_answer`, `extract_pred`, `max_new_tokens`).
2. Register it in `DATASET_LOADERS`.
3. Optionally add an entry to `LMEVAL_TASK_MAP` for lm-eval support.

---

## 9. Reproducing paper results

The original experiments used the following settings:

| Model    | Dataset         | Eval    | Limit | Threshold |
|----------|-----------------|---------|-------|-----------|
| LLaMA-3.1-8B | ARC-Challenge | custom | 290 | 0.08 |
| Mistral-7B   | ARC-Easy      | custom | 290 | 0.08 |
| Qwen-0.5B    | BoolQ         | custom | 1000 | 0.08 |
| Lucie-7B     | Winogrande    | custom | 1267 | 0.08 |
| LLaMA-3.1-8B | BigBench      | custom (mgpu) | 250 | 0.08 |

All models are evaluated with 0-shot prompting.
