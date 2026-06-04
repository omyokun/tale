"""Benchmark inference latency and throughput for baseline and TALE-pruned models."""

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tale.modeling import (
    ModifiedDecoderModel,
    get_model_dtype,
    get_model_root,
    maybe_sync,
)
from tale.tasks import TASK_NAMES, build_chat_prompt, load_task_dataset, tokenize_chat


# Example:
# python experiments/inference_benchmark.py --model <MODEL_PATH_OR_HF_ID> --model-name "Llama-3.1-8B-Instruct" --task arc_easy --data-path data/arc/arc_easy_validation.csv --drop-layers-0idx 18,19,20,28,31


SUMMARY_FIELDS = [
    "timestamp_utc",
    "model_name",
    "model_path",
    "task",
    "variant",
    "drop_layers_0idx",
    "drop_layers_1idx",
    "num_model_layers",
    "num_samples",
    "max_new_tokens",
    "warmup_samples",
    "avg_first_token_latency_s",
    "p50_first_token_latency_s",
    "p90_first_token_latency_s",
    "p95_first_token_latency_s",
    "avg_throughput_tokens_per_s",
    "aggregate_throughput_tokens_per_s",
    "avg_output_tokens",
    "total_output_tokens",
    "total_generation_time_s",
]


COMPARISON_FIELDS = [
    "model_name",
    "task",
    "drop_layers_0idx",
    "drop_layers_1idx",
    "baseline_latency_s",
    "pruned_latency_s",
    "latency_change_pct",
    "baseline_throughput_tokens_per_s",
    "pruned_throughput_tokens_per_s",
    "throughput_change_pct",
    "speedup_ratio",
]


def parse_layer_list(raw_value: str) -> List[int]:
    if not raw_value:
        return []
    layers = [int(value.strip()) for value in raw_value.split(",") if value.strip()]
    return sorted(set(layers))


def to_1_index(layers_0idx: Sequence[int]) -> List[int]:
    return [int(layer) + 1 for layer in layers_0idx]


def to_0_index(layers_1idx: Sequence[int]) -> List[int]:
    return [int(layer) - 1 for layer in layers_1idx]


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return get_model_dtype(device)
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def validate_drop_layers(drop_layers_0idx: Sequence[int], num_layers: int) -> None:
    invalid = [idx for idx in drop_layers_0idx if idx < 0 or idx >= num_layers]
    if invalid:
        raise ValueError(
            f"Invalid 0-index drop layers {invalid} for model with {num_layers} layers."
        )


def build_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("experiments.inference_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def generate_with_timing(
    model_instance: nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[int, float, float]:
    current_ids = input_ids
    first_token_latency_s = 0.0
    output_tokens = 0

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_ids: set[int] = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_token_ids = {int(token_id) for token_id in eos_token_id}
    else:
        eos_token_ids = {int(eos_token_id)}

    with torch.no_grad():
        maybe_sync(device)
        generation_start = time.perf_counter()

        for step in range(max_new_tokens):
            maybe_sync(device)
            step_start = time.perf_counter()

            outputs = model_instance(current_ids)
            next_token = torch.argmax(outputs["logits"][:, -1, :], dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            output_tokens += 1

            maybe_sync(device)
            step_end = time.perf_counter()

            if step == 0:
                first_token_latency_s = step_end - step_start

            if eos_token_ids and int(next_token.item()) in eos_token_ids:
                break

        maybe_sync(device)
        generation_time_s = time.perf_counter() - generation_start

    return output_tokens, first_token_latency_s, generation_time_s


def run_benchmark(
    model_instance: nn.Module,
    tokenizer,
    task: str,
    records: List[Dict[str, Any]],
    num_samples: int,
    max_new_tokens: int,
    warmup_samples: int,
    device: torch.device,
    progress_label: str,
) -> Dict[str, float]:
    selected_records = records[: min(num_samples, len(records))]

    for warmup_record in selected_records[: min(warmup_samples, len(selected_records))]:
        chat = build_chat_prompt(task, warmup_record)
        input_ids = tokenize_chat(tokenizer, chat, device)
        generate_with_timing(
            model_instance=model_instance,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    first_token_latencies: List[float] = []
    throughputs: List[float] = []
    output_tokens_list: List[int] = []
    generation_times: List[float] = []

    for record in tqdm(selected_records, desc=progress_label):
        chat = build_chat_prompt(task, record)
        input_ids = tokenize_chat(tokenizer, chat, device)
        output_tokens, first_latency_s, generation_time_s = generate_with_timing(
            model_instance=model_instance,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        throughput = output_tokens / generation_time_s if generation_time_s > 0 else 0.0
        first_token_latencies.append(first_latency_s)
        throughputs.append(throughput)
        output_tokens_list.append(output_tokens)
        generation_times.append(generation_time_s)

    total_output_tokens = int(sum(output_tokens_list))
    total_generation_time_s = float(sum(generation_times))

    return {
        "avg_first_token_latency_s": float(np.mean(first_token_latencies))
        if first_token_latencies
        else 0.0,
        "p50_first_token_latency_s": float(np.percentile(first_token_latencies, 50))
        if first_token_latencies
        else 0.0,
        "p90_first_token_latency_s": float(np.percentile(first_token_latencies, 90))
        if first_token_latencies
        else 0.0,
        "p95_first_token_latency_s": float(np.percentile(first_token_latencies, 95))
        if first_token_latencies
        else 0.0,
        "avg_throughput_tokens_per_s": float(np.mean(throughputs))
        if throughputs
        else 0.0,
        "aggregate_throughput_tokens_per_s": float(
            total_output_tokens / total_generation_time_s
        )
        if total_generation_time_s > 0
        else 0.0,
        "avg_output_tokens": float(np.mean(output_tokens_list))
        if output_tokens_list
        else 0.0,
        "total_output_tokens": total_output_tokens,
        "total_generation_time_s": total_generation_time_s,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct_change(baseline: float, updated: float) -> float:
    if baseline == 0:
        return 0.0
    return ((updated - baseline) / baseline) * 100.0


def build_comparison_row(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    variants = {row["variant"]: row for row in summary_rows}
    base_row = variants.get("base")
    pruned_row = variants.get("pruned")
    if base_row is None or pruned_row is None:
        return None

    baseline_latency_s = float(base_row["avg_first_token_latency_s"])
    pruned_latency_s = float(pruned_row["avg_first_token_latency_s"])
    baseline_throughput = float(base_row["aggregate_throughput_tokens_per_s"])
    pruned_throughput = float(pruned_row["aggregate_throughput_tokens_per_s"])
    speedup_ratio = (
        pruned_throughput / baseline_throughput if baseline_throughput > 0 else 0.0
    )

    return {
        "model_name": pruned_row["model_name"],
        "task": pruned_row["task"],
        "drop_layers_0idx": pruned_row["drop_layers_0idx"],
        "drop_layers_1idx": pruned_row["drop_layers_1idx"],
        "baseline_latency_s": baseline_latency_s,
        "pruned_latency_s": pruned_latency_s,
        "latency_change_pct": pct_change(baseline_latency_s, pruned_latency_s),
        "baseline_throughput_tokens_per_s": baseline_throughput,
        "pruned_throughput_tokens_per_s": pruned_throughput,
        "throughput_change_pct": pct_change(baseline_throughput, pruned_throughput),
        "speedup_ratio": speedup_ratio,
    }


def run(args: argparse.Namespace) -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    logger = build_logger(run_dir)

    device = select_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)
    records = load_task_dataset(args.task, args.data_path)
    num_samples = args.num_samples or len(records)

    if args.drop_layers_0idx and args.drop_layers_1idx:
        raise ValueError("Use only one of --drop-layers-0idx or --drop-layers-1idx.")
    drop_layers_0idx = parse_layer_list(args.drop_layers_0idx)
    if args.drop_layers_1idx:
        drop_layers_0idx = to_0_index(parse_layer_list(args.drop_layers_1idx))

    logger.info("Using device: %s", device)
    logger.info("Using dtype: %s", dtype)
    logger.info("Loaded %d records from %s", len(records), args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model_base = model_base.to(device)
    model_base.eval()

    num_layers = len(get_model_root(model_base).layers)
    validate_drop_layers(drop_layers_0idx, num_layers)
    model_name = args.model_name or args.model

    variants = [("base", []), ("pruned", drop_layers_0idx)]
    if args.skip_base:
        variants = [variant for variant in variants if variant[0] != "base"]
    if args.skip_pruned or not drop_layers_0idx:
        variants = [variant for variant in variants if variant[0] != "pruned"]

    summary_rows: List[Dict[str, Any]] = []
    for variant_name, variant_drop_layers in variants:
        logger.info(
            "Running %s | task=%s | samples=%d | max_new_tokens=%d | drop_layers=%s",
            variant_name,
            args.task,
            num_samples,
            args.max_new_tokens,
            variant_drop_layers,
        )
        model_instance = ModifiedDecoderModel(
            original_model=model_base,
            delete_indices=variant_drop_layers,
            device=device,
        )
        model_instance.eval()
        summary = run_benchmark(
            model_instance=model_instance,
            tokenizer=tokenizer,
            task=args.task,
            records=records,
            num_samples=num_samples,
            max_new_tokens=args.max_new_tokens,
            warmup_samples=args.warmup_samples,
            device=device,
            progress_label=f"{args.task}:{variant_name}",
        )
        row = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "model_path": args.model,
            "task": args.task,
            "variant": variant_name,
            "drop_layers_0idx": json.dumps(variant_drop_layers),
            "drop_layers_1idx": json.dumps(to_1_index(variant_drop_layers)),
            "num_model_layers": num_layers,
            "num_samples": num_samples,
            "max_new_tokens": args.max_new_tokens,
            "warmup_samples": args.warmup_samples,
            **summary,
        }
        summary_rows.append(row)
        write_csv(run_dir / "results.csv", summary_rows, SUMMARY_FIELDS)

        del model_instance
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison_row = build_comparison_row(summary_rows)
    if comparison_row is not None:
        write_csv(run_dir / "comparison.csv", [comparison_row], COMPARISON_FIELDS)

    logger.info("Results written to %s", run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency and throughput for baseline and TALE-pruned models."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--model-name", default="", help="Display name used in output tables.")
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--data-path", required=True, help="Prepared task CSV path.")
    parser.add_argument("--drop-layers-0idx", default="", help="Comma-separated 0-indexed layer ids.")
    parser.add_argument("--drop-layers-1idx", default="", help="Comma-separated 1-indexed layer ids.")
    parser.add_argument("--num-samples", type=int, default=None, help="Defaults to all rows in the CSV.")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/inference_benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-pruned", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
