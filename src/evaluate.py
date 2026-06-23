"""
Evaluation helpers for greedy layer pruning.

Two evaluation modes
--------------------
custom  – project-specific greedy decoding; no external harness needed.
lmeval  – uses lm-evaluation-harness (pip install lm-eval).

Core algorithm
--------------
greedy_layer_dropping(eval_fn, total_layers, threshold)
  Accepts any callable eval_fn(drop_set) → (accuracy, seconds) so it works
  identically for both evaluation modes and for single-/multi-GPU setups.
"""
import time
import torch
from tqdm import tqdm


# ─── Greedy decoding ──────────────────────────────────────────────────────────

def greedy_generate(model, input_ids, max_new_tokens=1):
    """Run greedy (argmax) decoding for max_new_tokens steps."""
    current = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(current)
            next_tok = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
            current = torch.cat([current, next_tok], dim=1)
    return current[:, input_ids.shape[1]:]


# ─── Custom evaluation ────────────────────────────────────────────────────────

def custom_evaluate(model, dataset_info, tokenizer, device, limit=None):
    """
    Evaluate model using project-specific greedy decoding.

    Parameters
    ----------
    model        : a ModifiedModel (or any model exposing {"logits": ...} in forward)
    dataset_info : dict returned by datasets.load_dataset()
    tokenizer    : HuggingFace tokenizer
    device       : torch.device
    limit        : if set, only evaluate the first `limit` samples

    Returns
    -------
    (accuracy: float, elapsed_seconds: float)
    """
    items          = dataset_info["items"]
    format_chat    = dataset_info["format_chat"]
    get_answer     = dataset_info["get_answer"]
    extract_pred   = dataset_info["extract_pred"]
    max_new_tokens = dataset_info["max_new_tokens"]

    if limit:
        items = items[:limit]

    correct = 0
    t0 = time.time()

    for item in tqdm(items, desc="Evaluating", leave=False):
        chat     = format_chat(item)
        text     = tokenizer.apply_chat_template(chat, tokenize=False,
                                                  add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        gen_ids   = greedy_generate(model, input_ids, max_new_tokens=max_new_tokens)
        predicted = extract_pred(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
        expected  = get_answer(item)

        if str(predicted).strip().lower() == str(expected).strip().lower():
            correct += 1

    accuracy = correct / len(items) if items else 0.0
    return accuracy, time.time() - t0


# ─── lm-eval evaluation ───────────────────────────────────────────────────────

def lmeval_evaluate(model, task_name, tokenizer, device, limit=None, num_fewshot=0):
    """
    Evaluate using lm-evaluation-harness.

    Requires:  pip install lm-eval

    Parameters
    ----------
    model       : standard HuggingFace causal LM (use patch_model_inplace, not ModifiedModel)
    task_name   : lm-eval task string, e.g. "arc_challenge"
    tokenizer   : HuggingFace tokenizer
    device      : torch.device
    limit       : max samples (None = all)
    num_fewshot : number of few-shot examples

    Returns
    -------
    (accuracy: float, elapsed_seconds: float)
    """
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            "lm-eval is not installed. Run:  pip install lm-eval"
        ) from e

    lm = HFLM(
        pretrained=model,
        backend="causal",
        tokenizer=tokenizer,
        batch_size=1,
        device=str(device),
    )

    t0 = time.time()
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=1,
    )
    elapsed = time.time() - t0

    accuracy = _extract_lmeval_accuracy(results, task_name)
    if accuracy is None:
        raise RuntimeError(
            f"Could not extract accuracy from lm-eval results for task '{task_name}'. "
            f"Available keys: {list(results.get('results', {}).keys())}"
        )
    return accuracy, elapsed


def _extract_lmeval_accuracy(results, task_name):
    all_results = results.get("results", results)

    if task_name in all_results:
        task_res = all_results[task_name]
    else:
        subtasks = {k: v for k, v in all_results.items() if task_name in k}
        if subtasks:
            accs = []
            for tr in subtasks.values():
                for key in ("acc_norm,none", "acc_norm", "acc,none", "acc",
                            "exact_match,none", "exact_match"):
                    if key in tr:
                        accs.append(tr[key])
                        break
            return sum(accs) / len(accs) if accs else None
        return None

    for key in ("acc_norm,none", "acc_norm", "acc,none", "acc",
                "exact_match,none", "exact_match"):
        if key in task_res:
            return task_res[key]
    return None


# ─── Greedy layer-dropping algorithm ─────────────────────────────────────────

def greedy_layer_dropping(eval_fn, total_layers, threshold=0.08):
    """
    Iteratively find transformer layers that can be removed while keeping
    accuracy within `threshold` of the full-model baseline.

    At each iteration every remaining layer is tested individually; the one
    whose removal hurts accuracy the least (while staying above the threshold)
    is permanently dropped.  The loop stops when no single-layer removal meets
    the threshold.

    Parameters
    ----------
    eval_fn      : callable(drop_set: set[int]) → (accuracy: float, seconds: float)
                   Must handle an empty set (baseline) and any subset of layers.
    total_layers : int  — number of layers in the full model
    threshold    : float — maximum allowed accuracy drop from baseline (default 0.08)

    Returns
    -------
    dropped      : set[int]   — final set of dropped layer indices
    all_results  : dict       — per-iteration diagnostics
    baseline_acc : float      — accuracy of the full model
    """
    print("=" * 65)
    print("Getting baseline accuracy (no layers dropped)…")
    baseline_acc, baseline_dur = eval_fn(set())
    threshold_acc = baseline_acc - threshold

    print(f"  Total layers    : {total_layers}")
    print(f"  Baseline acc    : {baseline_acc:.4f}")
    print(f"  Threshold acc   : {threshold_acc:.4f}  (baseline − {threshold:.0%})")
    print(f"  Baseline time   : {baseline_dur:.1f}s")
    print("=" * 65)

    dropped     = set()
    all_results = {}
    iteration   = 1

    while True:
        testable = set(range(total_layers)) - dropped
        if not testable:
            print("No more layers to test. Done.")
            break

        print(f"\n─── Iteration {iteration}  |  dropped so far: "
              f"{sorted(dropped) if dropped else '∅'} ───")

        iter_results  = {}
        iter_timings  = {}

        for layer_idx in tqdm(sorted(testable), desc=f"Iter {iteration}"):
            candidate = dropped | {layer_idx}
            acc, dur  = eval_fn(candidate)
            iter_results[layer_idx] = acc
            iter_timings[layer_idx] = dur
            print(f"  drop layer {layer_idx:2d}  →  acc={acc:.4f}  ({dur:.1f}s)")

        best_idx, best_acc = max(iter_results.items(), key=lambda x: x[1])
        print(f"\n  Best candidate : layer {best_idx}  acc={best_acc:.4f}"
              f"  threshold={threshold_acc:.4f}")

        if best_acc >= threshold_acc:
            dropped.add(best_idx)
            all_results[f"iteration_{iteration}"] = {
                "best_layer":     best_idx,
                "best_accuracy":  best_acc,
                "layer_results":  iter_results,
                "layer_timings":  iter_timings,
                "dropped_so_far": sorted(dropped),
            }
            speedup = baseline_dur / iter_timings[best_idx] if iter_timings[best_idx] > 0 else 0
            print(f"  ✅  Dropping layer {best_idx}  "
                  f"(total dropped={len(dropped)}, speedup≈{speedup:.2f}×)")
            iteration += 1
        else:
            print(f"  ❌  {best_acc:.4f} < threshold {threshold_acc:.4f}. Stopping.")
            break

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"  Layers dropped  : {sorted(dropped) if dropped else '∅'}")
    print(f"  Layers kept     : {total_layers - len(dropped)} / {total_layers}")
    print(f"  Compression     : {len(dropped)/total_layers:.1%}")

    if dropped:
        final_acc, final_dur = eval_fn(dropped)
        speedup = baseline_dur / final_dur if final_dur > 0 else float("inf")
        acc_drop = baseline_acc - final_acc
        print(f"  Final accuracy  : {final_acc:.4f}  (drop={acc_drop:.4f}, "
              f"{acc_drop/baseline_acc:.1%})")
        print(f"  Speedup         : {speedup:.2f}×")
    else:
        print("  No layers could be dropped within the threshold.")

    if all_results:
        print("\nIteration summary:")
        for name, data in all_results.items():
            print(f"  {name}: layer {data['best_layer']}  "
                  f"acc={data['best_accuracy']:.4f}  "
                  f"dropped={data['dropped_so_far']}")

    print("=" * 65)
    return dropped, all_results, baseline_acc
