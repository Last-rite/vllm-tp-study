#!/usr/bin/env python3
"""Offline vLLM benchmark for tensor-parallel scaling.

One process = one (model, TP, engine options) configuration. Inside the
process we sweep batch sizes and, for each batch, time two runs:

  * prefill run : output_len = 1          -> time ~= prefill (TTFT of the batch)
  * e2e run     : output_len = O          -> prefill + (O-1) decode steps

so that  decode step time (TPOT) = (t_e2e - t_prefill) / (O - 1).

Every finished measurement is appended to a CSV immediately, and the sweep
stops by itself before a soft time budget, so a job that is killed by the
6-minute Slurm limit still leaves all completed rows on disk.

Example:
  python harness/bench_tp.py --config configs/sweep_main.json \
      --model Qwen/Qwen3-8B --tp 2 --platform h100 --tag main
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform as pyplatform
import random
import socket
import sys
import time

T_START = time.perf_counter()


def elapsed():
    return time.perf_counter() - T_START


def log(msg):
    print(f"[bench {elapsed():7.1f}s] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="JSON sweep config")
    p.add_argument("--model", help="override model in config (HF id or local path)")
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--platform", required=True,
                   help="label, e.g. h100 / rtx4090 / rtx4060 (used in file names)")
    p.add_argument("--tag", default="main",
                   help="experiment tag, e.g. main / no_car / eager / profile")
    p.add_argument("--enforce-eager", action="store_true",
                   help="disable CUDA graphs and torch.compile (ablation)")
    p.add_argument("--disable-custom-all-reduce", action="store_true",
                   help="use NCCL all-reduce instead of vLLM's custom kernel (ablation)")
    p.add_argument("--batches", type=str, default=None,
                   help="override batch list, comma separated, e.g. 8,16")
    p.add_argument("--reps", type=int, default=None, help="override repetitions")
    p.add_argument("--time-budget", type=float, default=None,
                   help="soft limit in seconds from process start (default: config)")
    p.add_argument("--outdir", default="results/raw")
    return p.parse_args()


def kv_bytes_per_token(hf):
    layers = hf.get("num_hidden_layers")
    heads = hf.get("num_attention_heads")
    kv_heads = hf.get("num_key_value_heads", heads)
    head_dim = hf.get("head_dim") or hf["hidden_size"] // heads
    return 2 * layers * kv_heads * head_dim  # elements; x dtype bytes later


def estimate_params(hf):
    """Parameter count for a dense Llama/Qwen-style decoder (no biases)."""
    h = hf["hidden_size"]
    L = hf["num_hidden_layers"]
    heads = hf["num_attention_heads"]
    kv_heads = hf.get("num_key_value_heads", heads)
    head_dim = hf.get("head_dim") or h // heads
    inter = hf["intermediate_size"]
    vocab = hf["vocab_size"]
    attn = h * heads * head_dim * 2 + h * kv_heads * head_dim * 2  # q,o + k,v
    mlp = 3 * h * inter                                            # gate, up, down
    norms = 2 * h
    per_layer = attn + mlp + norms
    embed = vocab * h
    lm_head = 0 if hf.get("tie_word_embeddings", False) else vocab * h
    return {
        "params_total": L * per_layer + embed + lm_head + h,
        "params_nonembed": L * per_layer,   # matmul weights used per token
        "params_lm_head": vocab * h,        # also a matmul per token
    }


def kv_capacity_tokens(llm):
    """Best-effort: number of KV-cache tokens the engine allocated."""
    paths = [
        ("llm_engine", "vllm_config", "cache_config"),
        ("llm_engine", "cache_config"),
    ]
    for path in paths:
        obj = llm
        try:
            for name in path:
                obj = getattr(obj, name)
            blocks = obj.num_gpu_blocks
            bs = obj.block_size
            if blocks:
                return int(blocks) * int(bs)
        except Exception:
            continue
    return None


def make_prompts(batch, input_len, vocab, seed):
    rng = random.Random(seed)
    lo, hi = 1000, max(1001, vocab - 1000)
    return [{"prompt_token_ids": [rng.randrange(lo, hi) for _ in range(input_len)]}
            for _ in range(batch)]


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    model = args.model or cfg["model"]
    batches = ([int(x) for x in args.batches.split(",")] if args.batches
               else cfg["batches"])
    reps = args.reps or cfg.get("reps", 3)
    input_len = cfg["input_len"]
    output_len = cfg["output_len"]
    budget = args.time_budget or cfg.get("time_budget_s", 270)
    dtype = cfg.get("dtype", "bfloat16")
    seed = cfg.get("seed", 0)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    short_model = model.rstrip("/").split("/")[-1]
    base = f"{args.platform}_{short_model}_tp{args.tp}_{args.tag}_{stamp}"
    csv_path = os.path.join(args.outdir, base + ".csv")
    meta_path = os.path.join(args.outdir, base + ".meta.json")

    log(f"model={model} tp={args.tp} batches={batches} reps={reps} "
        f"I={input_len} O={output_len} budget={budget}s")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tp,
        dtype=dtype,
        max_model_len=input_len + output_len + 16,
        max_num_seqs=max(batches),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        seed=seed,
        trust_remote_code=False,
    )
    init_s = time.perf_counter() - t0
    log(f"engine ready in {init_s:.1f}s")

    try:
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(model).to_dict()
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: could not read HF config ({e})")
        hf = {}
    dbytes = 2 if dtype in ("bfloat16", "float16", "half") else 4
    kv_cap = kv_capacity_tokens(llm)
    meta = {
        "model": model, "tp": args.tp, "platform": args.platform, "tag": args.tag,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "dtype": dtype, "dtype_bytes": dbytes,
        "input_len": input_len, "output_len": output_len,
        "batches": batches, "reps": reps, "init_s": init_s,
        "kv_capacity_tokens": kv_cap,
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(i)
                      for i in range(torch.cuda.device_count())],
        "host": socket.gethostname(), "python": pyplatform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_config": {k: hf.get(k) for k in (
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "intermediate_size",
            "vocab_size", "tie_word_embeddings") if k in hf},
    }
    if hf:
        meta.update(estimate_params(hf))
        meta["kv_elems_per_token"] = kv_bytes_per_token(hf)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"meta -> {meta_path}  (kv capacity tokens: {kv_cap})")

    vocab = hf.get("vocab_size", 32000)
    fields = ["timestamp", "platform", "gpu_name", "tp", "model", "tag",
              "enforce_eager", "disable_custom_all_reduce", "batch",
              "input_len", "output_len", "rep", "t_prefill_s", "t_e2e_s",
              "tpot_s", "e2e_out_tok_s", "decode_tok_s", "elapsed_job_s"]
    fcsv = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    writer.writeheader()
    fcsv.flush()

    def run(prompts, out_len):
        sp = SamplingParams(max_tokens=out_len, min_tokens=out_len,
                            ignore_eos=True, temperature=0.0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt_s = time.perf_counter() - t
        got = sum(len(o.outputs[0].token_ids) for o in outs)
        if got != out_len * len(prompts):
            log(f"WARNING: expected {out_len * len(prompts)} tokens, got {got}")
        return dt_s

    # global warm-up (CUDA graphs, kernels, allocator)
    run(make_prompts(min(batches), input_len, vocab, seed + 999), 8)
    log("warm-up done")

    last_cost = 0.0
    stopped = False
    for b in batches:
        need = b * (input_len + output_len)
        if kv_cap and need > 0.95 * kv_cap:
            log(f"SKIP batch {b}: needs {need} KV tokens > 95% of capacity {kv_cap}")
            continue
        run(make_prompts(b, input_len, vocab, seed + 500 + b), 8)  # per-shape warm-up
        for r in range(reps):
            if elapsed() + 1.5 * last_cost > budget:
                log(f"time budget reached ({elapsed():.0f}s); stopping sweep")
                stopped = True
                break
            t_m = time.perf_counter()
            prompts = make_prompts(b, input_len, vocab, seed + 1000 * r + b)
            tp_ = run(prompts, 1)
            te = run(prompts, output_len)
            tpot = (te - tp_) / (output_len - 1)
            row = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "platform": args.platform, "gpu_name": meta["gpu_names"][0],
                "tp": args.tp, "model": short_model, "tag": args.tag,
                "enforce_eager": int(args.enforce_eager),
                "disable_custom_all_reduce": int(args.disable_custom_all_reduce),
                "batch": b, "input_len": input_len, "output_len": output_len,
                "rep": r, "t_prefill_s": f"{tp_:.6f}", "t_e2e_s": f"{te:.6f}",
                "tpot_s": f"{tpot:.6f}",
                "e2e_out_tok_s": f"{b * output_len / te:.2f}",
                "decode_tok_s": f"{b / tpot:.2f}",
                "elapsed_job_s": f"{elapsed():.1f}",
            }
            writer.writerow(row)
            fcsv.flush()
            os.fsync(fcsv.fileno())
            last_cost = time.perf_counter() - t_m
            log(f"B={b:4d} rep={r} prefill={tp_ * 1e3:8.1f}ms "
                f"tpot={tpot * 1e3:7.2f}ms e2e={b * output_len / te:9.1f} tok/s")
        if stopped:
            break
    fcsv.close()
    log(f"done -> {csv_path}")
    # vLLM worker processes sometimes keep the interpreter alive; exit hard.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
