import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tale.modeling import (
    ModifiedDecoderModel,
    get_model_dtype,
    get_model_root,
    maybe_sync,
)
from tale.tasks import (
    TASK_NAMES,
    build_chat_prompt,
    load_task_dataset,
    tokenize_chat,
)


CHOICE_LABELS = {
    "arc_easy": ["A", "B", "C", "D"],
    "arc_challenge": ["A", "B", "C", "D"],
    "boolq": ["A", "B"],
    "mmlu": ["A", "B", "C", "D"],
    "commonqa": ["A", "B", "C", "D", "E"],
    "winogrande": ["1", "2"],
}


def parse_layer_list(raw_value: str) -> Set[int]:
    if not raw_value:
        return set()
    return {int(value.strip()) for value in raw_value.split(",") if value.strip()}


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


def target_for_record(task: str, record: Dict[str, Any]) -> str:
    if task == "math500":
        return str(record["target"])
    return str(record["target"]).strip()


def normalize_numeric_answer(text: str) -> str:
    text = str(text).strip()
    boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed.group(1)
    final = re.search(r"####\s*([^\n]+)", text)
    if final:
        text = final.group(1)
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return text.strip().strip(".")


def normalize_target(task: str, target: str) -> str:
    if task == "bigbench":
        return target.strip().lower()
    if task in {"gsm8k_hard", "math500"}:
        return normalize_numeric_answer(target)
    if task == "winogrande":
        return target.strip().replace(".0", "")
    return target.strip().upper()


def extract_choice(text: str, labels: Sequence[str]) -> str:
    clean = text.strip().upper()
    if clean[:1] in labels:
        return clean[:1]
    pattern = r"\b(" + "|".join(re.escape(label) for label in labels) + r")\b"
    match = re.search(pattern, clean)
    if match:
        return match.group(1)
    return "INVALID"


def extract_prediction(task: str, text: str) -> str:
    if task in CHOICE_LABELS:
        return extract_choice(text, CHOICE_LABELS[task])
    if task == "bigbench":
        match = re.search(r"\b(true|false)\b", text.lower())
        return match.group(1) if match else "invalid"
    if task in {"gsm8k_hard", "math500"}:
        return normalize_numeric_answer(text)
    return text.strip()


def generate_text(
    model_instance,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    current_ids = input_ids
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_ids: Set[int] = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_token_ids = {int(token_id) for token_id in eos_token_id}
    else:
        eos_token_ids = {int(eos_token_id)}

    with torch.no_grad():
        for _ in range(max_new_tokens):
            maybe_sync(device)
            outputs = model_instance(current_ids)
            next_token = torch.argmax(outputs["logits"][:, -1, :], dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            if eos_token_ids and int(next_token.item()) in eos_token_ids:
                break

    generated = current_ids[:, input_ids.shape[1] :]
    if generated.numel() == 0:
        return ""
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def evaluate_accuracy(
    model_base,
    tokenizer,
    task: str,
    records: List[Dict[str, Any]],
    drop_layers_0idx: Set[int],
    num_samples: int,
    max_new_tokens: int,
    device: torch.device,
    label: str,
) -> Dict[str, Any]:
    selected_records = records[: min(num_samples, len(records))]
    model_instance = ModifiedDecoderModel(
        original_model=model_base,
        delete_indices=sorted(drop_layers_0idx),
        device=device,
    )
    model_instance.eval()

    correct = 0
    predictions: List[Dict[str, Any]] = []
    start_time = time.time()

    for idx, record in enumerate(tqdm(selected_records, desc=label)):
        chat = build_chat_prompt(task, record)
        input_ids = tokenize_chat(tokenizer, chat, device)
        output_text = generate_text(
            model_instance=model_instance,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        prediction = extract_prediction(task, output_text)
        target = normalize_target(task, target_for_record(task, record))
        is_correct = prediction == target
        correct += int(is_correct)
        predictions.append(
            {
                "index": idx,
                "prediction": prediction,
                "target": target,
                "correct": is_correct,
            }
        )

    elapsed_s = time.time() - start_time
    total = len(selected_records)
    del model_instance
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "elapsed_s": elapsed_s,
        "drop_layers_0idx": sorted(drop_layers_0idx),
        "predictions": predictions,
    }


def write_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def run_search(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)

    records = load_task_dataset(args.task, args.data_path)
    num_samples = args.num_samples or len(records)
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
    dropped_layers = parse_layer_list(args.initial_drop_layers_0idx)
    invalid = [layer for layer in dropped_layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Invalid 0-index layer ids for this model: {invalid}")

    accepted_configs: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    baseline = evaluate_accuracy(
        model_base=model_base,
        tokenizer=tokenizer,
        task=args.task,
        records=records,
        drop_layers_0idx=dropped_layers,
        num_samples=num_samples,
        max_new_tokens=args.max_new_tokens,
        device=device,
        label="baseline",
    )

    baseline_accuracy = float(baseline["accuracy"])
    threshold_accuracy = baseline_accuracy - args.epsilon
    initial_config = {
        "iteration": 0,
        "accepted_layer_0idx": None,
        "accuracy": baseline_accuracy,
        "correct": baseline["correct"],
        "total": baseline["total"],
        "elapsed_s": baseline["elapsed_s"],
        "drop_layers_0idx": sorted(dropped_layers),
    }
    accepted_configs.append(initial_config)
    best_config = dict(initial_config)
    bsba_config = dict(initial_config)

    stop_reason = "all layers tested"
    iteration = 1

    while len(dropped_layers) < num_layers:
        if args.max_drop_layers is not None and len(dropped_layers) >= args.max_drop_layers:
            stop_reason = "max drop layers reached"
            break

        testable_layers = [layer for layer in range(num_layers) if layer not in dropped_layers]
        iteration_results = []

        for layer in testable_layers:
            candidate_drop_layers = set(dropped_layers)
            candidate_drop_layers.add(layer)
            result = evaluate_accuracy(
                model_base=model_base,
                tokenizer=tokenizer,
                task=args.task,
                records=records,
                drop_layers_0idx=candidate_drop_layers,
                num_samples=num_samples,
                max_new_tokens=args.max_new_tokens,
                device=device,
                label=f"iter{iteration}:drop{layer}",
            )
            row = {
                "iteration": iteration,
                "candidate_layer_0idx": layer,
                "accuracy": float(result["accuracy"]),
                "correct": result["correct"],
                "total": result["total"],
                "elapsed_s": result["elapsed_s"],
                "drop_layers_0idx": json.dumps(result["drop_layers_0idx"]),
            }
            iteration_results.append(row)
            candidate_rows.append(row)

        best_candidate = max(iteration_results, key=lambda item: item["accuracy"])
        if float(best_candidate["accuracy"]) < threshold_accuracy:
            stop_reason = "best candidate fell below threshold"
            break

        accepted_layer = int(best_candidate["candidate_layer_0idx"])
        dropped_layers.add(accepted_layer)
        accepted_config = {
            "iteration": iteration,
            "accepted_layer_0idx": accepted_layer,
            "accuracy": float(best_candidate["accuracy"]),
            "correct": int(best_candidate["correct"]),
            "total": int(best_candidate["total"]),
            "elapsed_s": float(best_candidate["elapsed_s"]),
            "drop_layers_0idx": sorted(dropped_layers),
        }
        accepted_configs.append(accepted_config)

        if accepted_config["accuracy"] > best_config["accuracy"]:
            best_config = dict(accepted_config)
        if accepted_config["accuracy"] >= baseline_accuracy:
            bsba_config = dict(accepted_config)

        iteration += 1

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model": args.model,
        "task": args.task,
        "data_path": args.data_path,
        "num_model_layers": num_layers,
        "num_samples": num_samples,
        "max_new_tokens": args.max_new_tokens,
        "epsilon": args.epsilon,
        "baseline_accuracy": baseline_accuracy,
        "threshold_accuracy": threshold_accuracy,
        "final_accuracy": accepted_configs[-1]["accuracy"],
        "final_drop_layers_0idx": accepted_configs[-1]["drop_layers_0idx"],
        "best_accuracy": best_config["accuracy"],
        "best_drop_layers_0idx": best_config["drop_layers_0idx"],
        "bsba_accuracy": bsba_config["accuracy"],
        "bsba_drop_layers_0idx": bsba_config["drop_layers_0idx"],
        "accepted_iterations": len(accepted_configs) - 1,
        "stop_reason": stop_reason,
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "accepted_configs.json", accepted_configs)
    write_csv(output_dir / "candidate_results.csv", candidate_rows)
    write_json(output_dir / "baseline_predictions.json", baseline["predictions"])

    del model_base
    del tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TALE greedy task-aware layer elimination.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model path.")
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="outputs/tale_search")
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-drop-layers", type=int, default=None)
    parser.add_argument("--initial-drop-layers-0idx", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
