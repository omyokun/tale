#!/usr/bin/env python3
"""
Greedy Layer Pruning — Multi-Node / Multi-GPU (SLURM + NCCL)
=============================================================
Distributed version of the greedy pruning algorithm.  Transformer layers are
spread across all available GPUs using PyTorch distributed (NCCL backend).
The greedy decision loop runs only on rank 0; all ranks participate in every
forward pass through broadcast synchronisation.

Typical launch (via srun inside a SLURM job):
    srun python greedy_prune_mgpu.py \\
        --model-path meta-llama/Llama-3.1-8B-Instruct \\
        --dataset bigbench \\
        --data-path /data/bigbench_boolean_expressions.csv \\
        --threshold 0.08 \\
        --limit 250 \\
        --nodes 4 \\
        --gpus-per-node 2

Environment variables set by SLURM (read automatically):
    SLURM_PROCID   – global rank
    SLURM_NTASKS   – world size
    SLURM_LOCALID  – local rank within node
    SLURM_NODEID   – node index
    MASTER_ADDR    – master node hostname
    MASTER_PORT    – master node port (default 29500)

The script falls back to single-process mode when SLURM vars are absent,
which is useful for local testing.
"""
import argparse
import datetime
import gc
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    _LEGACY_MASK = True
except ImportError:
    _LEGACY_MASK = False

from src.datasets import load_dataset, DATASET_LOADERS

# ─── Model shortcuts ──────────────────────────────────────────────────────────
MODEL_SHORTCUTS = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "qwen":    "Qwen/Qwen2.5-0.5B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "lucie":   "OpenLLM-France/Lucie-7B-Instruct",
}


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Greedy layer pruning — distributed multi-GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True,
                   help="HuggingFace model ID or local path.")
    p.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS),
                   help="Dataset name.")
    p.add_argument("--data-path", required=True,
                   help="Path to local dataset CSV.")
    p.add_argument("--threshold", type=float, default=0.08,
                   help="Max accuracy drop allowed.")
    p.add_argument("--limit", type=int, default=250,
                   help="Max samples per evaluation call.")
    p.add_argument("--nodes", type=int, default=4,
                   help="Number of SLURM nodes (informational, also sets NODES).")
    p.add_argument("--gpus-per-node", type=int, default=2,
                   help="GPUs per node.")
    p.add_argument("--output", default="pruning_results_mgpu.json",
                   help="Output JSON path (written by rank 0 only).")
    return p.parse_args()


# ─── Distributed setup ────────────────────────────────────────────────────────

def setup_distributed():
    """Initialise NCCL process group from SLURM environment variables."""
    if "SLURM_PROCID" not in os.environ:
        print("SLURM environment not detected — running in single-process mode.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, 0, device

    rank       = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    node_id    = int(os.environ.get("SLURM_NODEID", 0))

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30),
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}/{world_size}] node={node_id} local_rank={local_rank} "
          f"device={device}")
    return rank, world_size, local_rank, node_id, device


# ─── Distributed model ────────────────────────────────────────────────────────

class DistributedModifiedModel(nn.Module):
    """
    Spread transformer layers across all ranks (GPUs).  Each rank owns a
    contiguous shard of the kept layers.  Forward passes are coordinated
    via broadcast; only rank 0 holds embeddings and the LM head.
    """

    def __init__(self, original_model, delete_indices, rank, world_size,
                 node_id, gpus_per_node):
        super().__init__()
        self.rank         = rank
        self.world_size   = world_size
        self.node_id      = node_id
        self.local_rank   = rank % gpus_per_node
        self.primary_dev  = f"cuda:{self.local_rank}"
        self.config       = original_model.config
        self.vocab_size   = original_model.config.vocab_size

        # Embedding and head live on primary device of every rank (but only
        # rank 0 uses them for real computation).
        self.embed_tokens = original_model.model.embed_tokens.to(self.primary_dev)
        self.norm         = original_model.model.norm.to(self.primary_dev)
        self.lm_head      = original_model.lm_head.to(self.primary_dev)

        # Filter layers
        kept = [(i, l) for i, l in enumerate(original_model.model.layers)
                if i not in set(delete_indices)]
        self.kept_layers = kept

        if rank == 0:
            print(f"  Original layers : {len(original_model.model.layers)}")
            print(f"  Dropped layers  : {sorted(delete_indices)}")
            print(f"  Kept layers     : {len(kept)}")

        # ── Distribute layers across ranks ─────────────────────────────────────
        n = len(kept)
        base, rem = divmod(n, world_size)
        if rank < rem:
            start = rank * (base + 1)
            count = base + 1
        else:
            start = rank * base + rem
            count = base
        end = start + count

        self.layer_device_map = {}
        self.distributed_layers = nn.ModuleDict()
        for idx in range(start, end):
            if idx < n:
                orig_idx, layer = kept[idx]
                key = f"layer_{idx}"
                self.distributed_layers[key] = layer.to(self.primary_dev)
                self.layer_device_map[idx] = self.primary_dev

        # Broadcast the complete layer→rank mapping
        if world_size > 1:
            if rank == 0:
                mapping = {}
                for r in range(world_size):
                    b2, r2 = divmod(n, world_size)
                    s = r * (b2 + 1) if r < r2 else r * b2 + r2
                    c = (b2 + 1) if r < r2 else b2
                    for i in range(s, s + c):
                        if i < n:
                            mapping[i] = r
                container = [mapping]
            else:
                container = [None]
            dist.broadcast_object_list(container, src=0)
            self.global_layer_rank_map = container[0]
        else:
            self.global_layer_rank_map = {i: 0 for i in range(n)}

    def forward(self, input_ids, attention_mask=None, **kwargs):
        primary_dev = self.primary_dev
        n = len(self.kept_layers)

        if self.rank == 0:
            input_ids = input_ids.to(primary_dev)
            batch_size, seq_len = input_ids.shape

            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_len), device=primary_dev)
            else:
                attention_mask = attention_mask.to(primary_dev)

            position_ids = (
                torch.arange(seq_len, device=primary_dev)
                .unsqueeze(0).expand(batch_size, -1)
            )
            hidden_states = self.embed_tokens(input_ids)

            if _LEGACY_MASK:
                causal_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_len), hidden_states, 0
                )
            else:
                causal_mask = attention_mask
        else:
            hidden_states = None
            causal_mask   = None
            position_ids  = None

        # ── Sequential layer forward with broadcast sync ───────────────────────
        for global_idx in range(n):
            target_rank = self.global_layer_rank_map.get(global_idx, -1)

            # On the first step, broadcast tensor shapes so non-0 ranks can
            # allocate buffers.
            if global_idx == 0 and self.world_size > 1:
                if self.rank == 0:
                    info = [list(hidden_states.shape), list(causal_mask.shape)]
                else:
                    info = [None, None]
                dist.broadcast_object_list(info, src=0)
                if self.rank != 0:
                    h_shape, m_shape = info
                    hidden_states = torch.zeros(h_shape, dtype=torch.float16,
                                                device=primary_dev)
                    causal_mask   = torch.zeros(m_shape, dtype=torch.float16,
                                                device=primary_dev)
                    seq_len = h_shape[1]
                    position_ids = (
                        torch.arange(seq_len, device=primary_dev)
                        .unsqueeze(0).expand(h_shape[0], -1)
                    )

            # Compute on the owning rank
            if target_rank == self.rank:
                key   = f"layer_{global_idx}"
                layer = self.distributed_layers[key]
                layer_out = layer(
                    hidden_states=hidden_states.to(primary_dev),
                    attention_mask=causal_mask.to(primary_dev),
                    position_ids=position_ids.to(primary_dev),
                )
                hidden_states = layer_out[0]

            # Broadcast result to all ranks
            if self.world_size > 1:
                buf = (hidden_states.detach().contiguous().to(primary_dev)
                       if target_rank == self.rank
                       else torch.zeros_like(hidden_states).to(primary_dev))
                dist.broadcast(buf, src=target_rank)
                hidden_states = buf

        # ── Final norm + head (rank 0 only) ───────────────────────────────────
        if self.rank == 0:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return {"logits": logits}
        else:
            dummy = torch.zeros((1, 1, self.vocab_size), device=primary_dev)
            return {"logits": dummy}


# ─── Evaluation ───────────────────────────────────────────────────────────────

def distributed_evaluate(model, dataset_info, tokenizer, primary_dev,
                         rank, world_size, limit=None):
    """
    Run evaluation across all ranks.  Rank 0 loads data and decodes;
    all other ranks participate in forward passes.

    Returns (accuracy, elapsed_seconds) — meaningful only on rank 0.
    """
    items          = dataset_info["items"]
    format_chat    = dataset_info["format_chat"]
    get_answer     = dataset_info["get_answer"]
    extract_pred   = dataset_info["extract_pred"]
    max_new_tokens = dataset_info["max_new_tokens"]

    if limit:
        items = items[:limit]
    n = len(items)

    # Broadcast the number of items so all ranks iterate the same count
    if world_size > 1:
        count_tensor = torch.tensor([n], device=primary_dev)
        dist.broadcast(count_tensor, src=0)
        n = int(count_tensor.item())

    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()

    correct = 0
    t0 = time.time()

    for i in range(n):
        if rank == 0:
            item   = items[i]
            chat   = format_chat(item)
            text   = tokenizer.apply_chat_template(chat, tokenize=False,
                                                    add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(primary_dev)
        else:
            input_ids = torch.ones((1, 10), dtype=torch.long, device=primary_dev)

        # Greedy decoding — all ranks participate
        current_ids = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(current_ids)

                if rank == 0:
                    next_tok = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
                    current_ids = torch.cat([current_ids, next_tok], dim=1)
                    seq_len_t = torch.tensor([current_ids.shape[1]], device=primary_dev)
                else:
                    seq_len_t = torch.tensor([0], device=primary_dev)

                if world_size > 1:
                    dist.broadcast(seq_len_t, src=0)
                    new_len = int(seq_len_t.item())
                    if rank != 0 and new_len > current_ids.shape[1]:
                        pad = torch.ones(
                            (1, new_len - current_ids.shape[1]),
                            dtype=torch.long, device=primary_dev
                        )
                        current_ids = torch.cat([current_ids, pad], dim=1)

        if rank == 0:
            gen_ids   = current_ids[:, input_ids.shape[1]:]
            predicted = extract_pred(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
            expected  = get_answer(items[i])
            if str(predicted).strip().lower() == str(expected).strip().lower():
                correct += 1

    if world_size > 1:
        dist.barrier()

    accuracy = correct / n if n > 0 else 0.0
    return accuracy, time.time() - t0


# ─── Greedy algorithm ─────────────────────────────────────────────────────────

def greedy_layer_dropping_distributed(
    model_base, dataset_info, tokenizer, primary_dev,
    rank, world_size, node_id, gpus_per_node,
    total_layers, threshold=0.08, limit=None,
):
    """Distributed greedy layer-dropping loop."""

    def make_model(drop_set):
        m = DistributedModifiedModel(
            model_base, drop_set, rank, world_size, node_id, gpus_per_node
        )
        m.eval()
        return m

    def eval_model(drop_set):
        m = make_model(drop_set)
        acc, dur = distributed_evaluate(
            m, dataset_info, tokenizer, primary_dev, rank, world_size, limit
        )
        del m
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
        gc.collect()
        if world_size > 1:
            dist.barrier()
        return acc, dur

    # Baseline
    if rank == 0:
        print("=" * 65)
        print("Baseline evaluation (no layers dropped)…")
    baseline_acc, baseline_dur = eval_model(set())
    threshold_acc = baseline_acc - threshold

    if rank == 0:
        print(f"  Total layers    : {total_layers}")
        print(f"  Baseline acc    : {baseline_acc:.4f}")
        print(f"  Threshold acc   : {threshold_acc:.4f}  (−{threshold:.0%})")
        print(f"  Baseline time   : {baseline_dur:.1f}s")
        print("=" * 65)

    dropped     = set()
    all_results = {}
    iteration   = 1

    while True:
        testable = set(range(total_layers)) - dropped
        if not testable:
            if rank == 0:
                print("No more layers to test. Done.")
            break

        if rank == 0:
            print(f"\n─── Iteration {iteration}  |  dropped so far: "
                  f"{sorted(dropped) if dropped else '∅'} ───")

        iter_results = {}
        iter_timings = {}

        for layer_idx in sorted(testable):
            if rank == 0:
                print(f"  Testing drop of layer {layer_idx}…")
            candidate = dropped | {layer_idx}
            acc, dur  = eval_model(candidate)
            if rank == 0:
                iter_results[layer_idx] = acc
                iter_timings[layer_idx] = dur
                print(f"  drop layer {layer_idx:2d}  →  acc={acc:.4f}  ({dur:.1f}s)")

        # Rank 0 decides; broadcasts result to all ranks
        if rank == 0:
            best_idx, best_acc = max(iter_results.items(), key=lambda x: x[1])
            should_continue = best_acc >= threshold_acc
            decision = [int(should_continue), best_idx]
        else:
            decision = [0, 0]

        if world_size > 1:
            dist.broadcast_object_list(decision, src=0)

        should_continue, best_idx = bool(decision[0]), decision[1]

        if should_continue:
            dropped.add(best_idx)
            if rank == 0:
                best_acc = iter_results[best_idx]
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
            if rank == 0:
                best_acc = iter_results.get(best_idx, 0)
                print(f"  ❌  {best_acc:.4f} < threshold {threshold_acc:.4f}. Stopping.")
            break

    # Final eval
    if rank == 0:
        print("\n" + "=" * 65)
        print("FINAL RESULTS")
        print("=" * 65)
        print(f"  Layers dropped  : {sorted(dropped) if dropped else '∅'}")
        print(f"  Layers kept     : {total_layers - len(dropped)} / {total_layers}")
        print(f"  Compression     : {len(dropped)/total_layers:.1%}")

    if dropped:
        final_acc, final_dur = eval_model(dropped)
        if rank == 0:
            speedup  = baseline_dur / final_dur if final_dur > 0 else float("inf")
            acc_drop = baseline_acc - final_acc
            print(f"  Final accuracy  : {final_acc:.4f}  (drop={acc_drop:.4f})")
            print(f"  Speedup         : {speedup:.2f}×")

    if rank == 0:
        print("=" * 65)

    return dropped, all_results, baseline_acc


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Distributed init ──────────────────────────────────────────────────────
    rank, world_size, local_rank, node_id, primary_dev = setup_distributed()

    # ── Resolve model path ────────────────────────────────────────────────────
    model_path = MODEL_SHORTCUTS.get(args.model_path, args.model_path)

    if rank == 0:
        print(f"Model        : {model_path}")
        print(f"Dataset      : {args.dataset}")
        print(f"Data path    : {args.data_path}")
        print(f"Threshold    : {args.threshold:.0%}")
        print(f"Limit        : {args.limit}")
        print(f"Nodes        : {args.nodes}  |  GPUs/node: {args.gpus_per_node}")
        print(f"World size   : {world_size}")

    # ── Load data (rank 0 loads and broadcasts) ───────────────────────────────
    if rank == 0:
        dataset_info = load_dataset(args.dataset, args.data_path)
        payload = [dataset_info]
    else:
        payload = [None]

    if world_size > 1:
        dist.barrier()
        dist.broadcast_object_list(payload, src=0)

    dataset_info = payload[0]

    # ── Load tokenizer ────────────────────────────────────────────────────────
    if rank == 0:
        print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if world_size > 1:
        dist.barrier()

    # ── Load model (staggered to avoid OOM) ──────────────────────────────────
    print(f"[Rank {rank}] loading model (delay={rank*10}s)…")
    time.sleep(rank * 10)
    model_base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    print(f"[Rank {rank}] model loaded")

    if world_size > 1:
        dist.barrier()

    total_layers = len(model_base.model.layers)
    if rank == 0:
        print(f"Model layers : {total_layers}")

    # ── Run pruning ───────────────────────────────────────────────────────────
    dropped, results, baseline_acc = greedy_layer_dropping_distributed(
        model_base    = model_base,
        dataset_info  = dataset_info,
        tokenizer     = tokenizer,
        primary_dev   = primary_dev,
        rank          = rank,
        world_size    = world_size,
        node_id       = node_id,
        gpus_per_node = args.gpus_per_node,
        total_layers  = total_layers,
        threshold     = args.threshold,
        limit         = args.limit,
    )

    # ── Save results (rank 0 only) ────────────────────────────────────────────
    if rank == 0:
        output = {
            "model":          model_path,
            "dataset":        args.dataset,
            "eval_mode":      "custom_mgpu",
            "threshold":      args.threshold,
            "limit":          args.limit,
            "nodes":          args.nodes,
            "gpus_per_node":  args.gpus_per_node,
            "world_size":     world_size,
            "baseline_acc":   baseline_acc,
            "dropped_layers": sorted(dropped),
            "layers_kept":    total_layers - len(dropped),
            "total_layers":   total_layers,
            "compression":    f"{len(dropped)/total_layers:.1%}",
            "iterations":     results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")
        print(f"Layers to drop  : {sorted(dropped)}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
    gc.collect()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
