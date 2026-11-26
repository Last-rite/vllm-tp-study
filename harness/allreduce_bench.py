#!/usr/bin/env python3
"""All-reduce microbenchmark with torch.distributed (NCCL backend).

Uses the same NCCL library that vLLM uses, so no separate nccl-tests build is
needed. Spawns one process per GPU inside a single Slurm task.

Outputs one CSV row per message size:
  bytes, time_us (mean over iterations, max over ranks), algbw_GBps, busbw_GBps
where busbw = algbw * 2(p-1)/p (same definition as nccl-tests).

Also records whether GPU peer-to-peer access is available (important for the
PCIe RTX 4090 vs NVLink H100 comparison).

Example:
  python harness/allreduce_bench.py --ngpus 2 --platform h100
"""
import argparse
import csv
import datetime as dt
import json
import os
import socket
import time


def worker(rank, world, args, out_rows):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dtype = torch.bfloat16
    esize = 2
    sizes = [2 ** k for k in range(args.min_exp, args.max_exp + 1)]
    rows = []
    for nbytes in sizes:
        n = max(1, nbytes // esize)
        x = torch.ones(n, dtype=dtype, device="cuda")
        iters = args.iters if nbytes <= (1 << 24) else max(5, args.iters // 4)
        for _ in range(args.warmup):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            dist.all_reduce(x)
        end.record()
        torch.cuda.synchronize()
        t_us = start.elapsed_time(end) * 1e3 / iters
        t = torch.tensor([t_us], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        t_us = t.item()
        if rank == 0:
            algbw = nbytes / (t_us * 1e-6) / 1e9
            busbw = algbw * 2 * (world - 1) / world
            rows.append({"bytes": n * esize, "time_us": f"{t_us:.3f}",
                         "algbw_GBps": f"{algbw:.3f}", "busbw_GBps": f"{busbw:.3f}"})
            print(f"{n * esize:>12d} B  {t_us:10.2f} us  busbw {busbw:8.2f} GB/s",
                  flush=True)
    if rank == 0:
        out_rows.extend(rows)
        with open(args._tmp, "w") as f:
            json.dump(rows, f)
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ngpus", type=int, default=2)
    p.add_argument("--platform", required=True)
    p.add_argument("--min-exp", type=int, default=10)   # 1 KiB
    p.add_argument("--max-exp", type=int, default=28)   # 256 MiB
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--outdir", default="results/raw")
    args = p.parse_args()

    import torch
    import torch.multiprocessing as mp
    world = args.ngpus
    assert torch.cuda.device_count() >= world, "not enough visible GPUs"
    p2p = {}
    for i in range(world):
        for j in range(world):
            if i != j:
                p2p[f"{i}->{j}"] = bool(torch.cuda.can_device_access_peer(i, j))
    print("peer access:", p2p, flush=True)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    os.makedirs(args.outdir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(args.outdir, f"allreduce_{args.platform}_p{world}_{stamp}")
    args._tmp = base + ".tmp.json"

    t0 = time.time()
    mp.spawn(worker, args=(world, args, []), nprocs=world, join=True)
    with open(args._tmp) as f:
        rows = json.load(f)
    os.remove(args._tmp)

    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["platform", "p", "bytes", "time_us",
                                          "algbw_GBps", "busbw_GBps"])
        w.writeheader()
        for r in rows:
            w.writerow({"platform": args.platform, "p": world, **r})
    meta = {"platform": args.platform, "p": world, "peer_access": p2p,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(world)],
            "torch": torch.__version__, "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "host": socket.gethostname(), "wall_s": time.time() - t0,
            "step": os.environ.get("VTP_STEP"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": os.environ.get("VTP_GIT"), "log": os.environ.get("VTP_LOG"),
            "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    with open(base + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done -> {base}.csv  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
