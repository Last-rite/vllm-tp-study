#!/usr/bin/env python3
"""Turn Nsight Systems reports into a kernel-time breakdown CSV.

For each .nsys-rep it runs `nsys stats --report cuda_gpu_kern_sum` (needs nsys
on the machine where you run this; the free Nsight Systems GUI install on the
laptop works) and groups kernels by name into GEMM / Attention /
Communication / Other.

  python analysis/nsys_breakdown.py results/nsys/*_cuda_gpu_kern_sum.csv   # exported in the job
  python analysis/nsys_breakdown.py results/nsys/*.nsys-rep                # needs local nsys
  -> results/kernel_breakdown.csv   (run, category, time_ms)

Check the printed top kernels once: kernel names change across versions, so
adjust PATTERNS if a big kernel lands in "Other".
"""
import io
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = [
    ("Communication", r"nccl|allreduce|all_reduce|cross_device_reduce|custom_ar|one_shot|two_shot"),
    ("Attention", r"flash|fmha|attn|attention|paged|reshape_and_cache"),
    ("GEMM", r"gemm|cutlass|sm90_xmma|sm80_xmma|ampere_.*gemm|cublas|matmul|nvjet|wgmma"),
]


def category(name):
    n = name.lower()
    for cat, pat in PATTERNS:
        if re.search(pat, n):
            return cat
    return "Other"


def stats(rep):
    if rep.endswith(".csv"):  # already exported by the job (nsys stats --format csv)
        out = open(rep).read()
    else:
        out = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                              "--force-export=true", rep], capture_output=True, text=True,
                             check=True).stdout
    start = out.find("Time (%)")
    if start < 0:
        start = out.find("Time(%)")
    df = pd.read_csv(io.StringIO(out[start:]))
    tcol = [c for c in df.columns if c.startswith("Total Time")][0]
    df["time_ms"] = df[tcol] / 1e6
    return df.rename(columns={"Name": "name"})[["name", "time_ms"]]


def main():
    reps = sys.argv[1:]
    if not reps:
        sys.exit(__doc__)
    rows = []
    for rep in reps:
        run = re.sub(r"(_cuda_gpu_kern_sum)?\.(nsys-rep|csv)$", "", os.path.basename(rep))
        df = stats(rep)
        df["category"] = df["name"].map(category)
        print(f"\n== {run}: top kernels")
        print(df.sort_values("time_ms", ascending=False).head(8)[["category", "time_ms", "name"]]
              .to_string(index=False, max_colwidth=70))
        for cat, t in df.groupby("category")["time_ms"].sum().items():
            rows.append({"run": run, "category": cat, "time_ms": t})
    out = pd.DataFrame(rows)
    path = os.path.join(ROOT, "results", "kernel_breakdown.csv")
    out.to_csv(path, index=False)
    print("\nwrote", path)
    print(out.pivot_table(index="run", columns="category", values="time_ms").round(1))


if __name__ == "__main__":
    main()
