#!/usr/bin/env python3
"""
Greedy Layer Pruning — Single GPU
==================================
Iteratively drops transformer layers from a causal LM while keeping accuracy
within a user-specified threshold of the full-model baseline.

Supports two evaluation modes:
  custom  – fast project-specific greedy decoding (no extra harness needed)
  lmeval  – uses lm-evaluation-harness for standardised benchmark scores

Usage examples
--------------
# Custom evaluation on ARC-Challenge (Llama-3.1-8B):
python greedy_prune.py \\
    --model-path meta-llama/Llama-3.1-8B-Instruct \\
    --dataset arc_challenge \\
    --data-path /data/arc_challenge_validation.csv \\
    --eval custom \\
    --threshold 0.08 \\
    --limit 290

# lm-eval evaluation on MMLU (Mistral-7B):
python greedy_prune.py \\
    --model-path mistralai/Mistral-7B-Instruct-v0.3 \\
    --dataset mmlu \\
    --eval lmeval \\
    --threshold 0.05 \\
    --limit 200

Supported datasets (--dataset)
-------------------------------
arc_challenge, arc_easy, boolq, bigbench, common_qa, gsm8k, mmlu, winogrande, math500

Notes
-----
- --data-path is required when --eval custom is used.
- --data-path is ignored when --eval lmeval (lm-eval fetches its own data).
- model names can be HuggingFace Hub IDs or local directory paths.
"""
import argparse
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.datasets import load_dataset, LMEVAL_TASK_MAP, DATASET_LOADERS
from src.evaluate import custom_evaluate, lmeval_evaluate, greedy_layer_dropping
from src.model import ModifiedModel, patch_model_inplace, restore_model_inplace


# ─── Preset model name shortcuts ─────────────────────────────────────────────
MODEL_SHORTCUTS = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":    "Qwen/Qwen2.5-0.5B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "lucie":   "OpenLLM-France/Lucie-7B-Instruct",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Greedy layer pruning for causal LLMs (single GPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Model ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--model-path", required=True,
        help=(
            "HuggingFace model name / local path. "
            f"Shortcuts: {list(MODEL_SHORTCUTS)}"
        ),
    )
    # ── Dataset ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--dataset", required=True, choices=list(DATASET_LOADERS),
        help="Dataset to evaluate on.",
    )
    p.add_argument(
        "--data-path", default=None,
        help="Path to the local dataset CSV file (required for --eval custom).",
    )
    # ── Evaluation ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--eval", default="custom", choices=["custom", "lmeval"],
        help="Evaluation backend.",
    )
    p.add_argument(
        "--num-fewshot", type=int, default=0,
        help="Few-shot examples for lmeval mode.",
    )
    # ── Pruning ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--threshold", type=float, default=0.08,
        help="Maximum accuracy drop allowed (e.g. 0.08 = 8 percentage points).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N samples per evaluation call.",
    )
    # ── Output ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output", default="pruning_results.json",
        help="Where to save the pruning results (JSON).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve model path ────────────────────────────────────────────────────
    model_path = MODEL_SHORTCUTS.get(args.model_path, args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Model       : {model_path}")
    print(f"Dataset     : {args.dataset}")
    print(f"Eval mode   : {args.eval}")
    print(f"Threshold   : {args.threshold:.0%}")
    print(f"Limit       : {args.limit or 'all'}")

    # ── Validate arguments ────────────────────────────────────────────────────
    if args.eval == "custom" and args.data_path is None:
        print("Error: --data-path is required when using --eval custom", file=sys.stderr)
        sys.exit(1)
    if args.eval == "lmeval" and args.dataset not in LMEVAL_TASK_MAP:
        print(
            f"Error: dataset '{args.dataset}' has no lm-eval mapping. "
            f"Use --eval custom or choose from: {list(LMEVAL_TASK_MAP)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load model & tokenizer ────────────────────────────────────────────────
    print("\nLoading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model…")
    model_base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model_base.eval()

    total_layers = len(model_base.model.layers)
    print(f"Model layers: {total_layers}")

    # ── Build evaluation function ─────────────────────────────────────────────
    if args.eval == "custom":
        dataset_info = load_dataset(args.dataset, args.data_path)

        def eval_fn(drop_set):
            pruned = ModifiedModel(model_base, delete_indices=drop_set, device=device)
            pruned.eval()
            acc, dur = custom_evaluate(
                pruned, dataset_info, tokenizer, device, limit=args.limit
            )
            del pruned
            torch.cuda.empty_cache()
            return acc, dur

    else:  # lmeval
        task_name = LMEVAL_TASK_MAP[args.dataset]

        def eval_fn(drop_set):
            patched, orig = patch_model_inplace(model_base, drop_set)
            acc, dur = lmeval_evaluate(
                patched, task_name, tokenizer, device,
                limit=args.limit, num_fewshot=args.num_fewshot,
            )
            restore_model_inplace(model_base, orig)
            torch.cuda.empty_cache()
            return acc, dur

    # ── Run greedy pruning ────────────────────────────────────────────────────
    dropped, results, baseline_acc = greedy_layer_dropping(
        eval_fn=eval_fn,
        total_layers=total_layers,
        threshold=args.threshold,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "model":         model_path,
        "dataset":       args.dataset,
        "eval_mode":     args.eval,
        "threshold":     args.threshold,
        "limit":         args.limit,
        "baseline_acc":  baseline_acc,
        "dropped_layers": sorted(dropped),
        "layers_kept":   total_layers - len(dropped),
        "total_layers":  total_layers,
        "compression":   f"{len(dropped)/total_layers:.1%}",
        "iterations":    results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print(f"Layers to drop  : {sorted(dropped)}")


if __name__ == "__main__":
    main()
