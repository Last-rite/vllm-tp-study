#!/usr/bin/env python3
"""Communication-aware analytical model for vLLM tensor parallelism.

    T_step(B, p) = max(T_comp, T_mem) + T_comm

    T_comp = 2 * (P_nonembed + P_lm_head) * N_tok / (p * F_peak * eta_c)
    T_mem  = (W_bytes + KV_bytes(B, ctx)) / (p * BW_hbm * eta_m)
    T_comm = 2 * L * T_allreduce(N_tok * h * s, p)            (0 when p = 1)
    T_allreduce(n, p) = a_p + 2(p-1)/p * n / beta_p            (fitted from allreduce_bench)

N_tok = B for a decode step and B * input_len for a prefill.
eta_c is calibrated on TP=1 prefill, eta_m on TP=1 decode; TP>1 is predicted
(out of sample). Writes results/model_predictions.csv and results/model_params.json.

  python analysis/model.py                     # all platforms found in results/raw
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "results", "raw")


def load_bench(tags=("main",)):
    rows = []
    for path in glob.glob(os.path.join(RAW, "*.csv")):
        if os.path.basename(path).startswith("allreduce_"):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        meta_path = path[:-4] + ".meta.json"
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        df["source"] = os.path.basename(path)
        for k in ("params_nonembed", "params_lm_head", "params_total",
                  "kv_elems_per_token", "dtype_bytes"):
            df[k] = meta.get(k, np.nan)
        hf = meta.get("hf_config", {})
        df["hidden_size"] = hf.get("hidden_size", np.nan)
        df["num_layers"] = hf.get("num_hidden_layers", np.nan)
        rows.append(df)
    if not rows:
        sys.exit(f"no benchmark CSVs in {RAW}")
    df = pd.concat(rows, ignore_index=True)
    if tags:
        df = df[df["tag"].isin(tags)]
    return df


def load_allreduce():
    rows = [pd.read_csv(p) for p in glob.glob(os.path.join(RAW, "allreduce_*.csv"))]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_allreduce(ar, max_bytes=16 << 20):
    """Least squares on t = a + c * n (relative error weighting) per (platform, p)."""
    params = {}
    for (plat, p), g in ar.groupby(["platform", "p"]):
        if p < 2:
            continue
        g = g[g["bytes"] <= max_bytes].groupby("bytes", as_index=False)["time_us"].median()
        n = g["bytes"].to_numpy(float)
        t = g["time_us"].to_numpy(float) * 1e-6
        w = 1.0 / t
        A = np.vstack([np.ones_like(n), n]).T * w[:, None]
        (a, c), *_ = np.linalg.lstsq(A, t * w, rcond=None)
        beta_bus = 2 * (p - 1) / p / c  # bytes/s
        params[f"{plat}:p{int(p)}"] = {"alpha_us": a * 1e6, "beta_bus_GBps": beta_bus / 1e9}
    return params


def t_allreduce(nbytes, p, ar):
    if p == 1 or ar is None:
        return 0.0
    return ar["alpha_us"] * 1e-6 + 2 * (p - 1) / p * nbytes / (ar["beta_bus_GBps"] * 1e9)


def work_terms(row, phase):
    s = row["dtype_bytes"]
    P = row["params_nonembed"] + row["params_lm_head"]
    if phase == "prefill":
        n_tok = row["batch"] * row["input_len"]
        kv = 0.0
    else:
        n_tok = row["batch"]
        ctx = row["input_len"] + row["output_len"] / 2
        kv = row["batch"] * ctx * row["kv_elems_per_token"] * s
    flops = 2 * P * n_tok
    mem = row["params_total"] * s + kv
    msg = n_tok * row["hidden_size"] * s
    return flops, mem, msg


def main():
    plat_spec = json.load(open(os.path.join(ROOT, "configs", "platforms.json")))
    df = load_bench()
    ar_params = fit_allreduce(load_allreduce()) if len(load_allreduce()) else {}

    # median over repetitions and jobs
    keys = ["platform", "tp", "batch", "input_len", "output_len"]
    num = ["t_prefill_s", "tpot_s", "params_nonembed", "params_lm_head", "params_total",
           "kv_elems_per_token", "dtype_bytes", "hidden_size", "num_layers"]
    agg = df.groupby(keys, as_index=False)[num].median()
    std = df.groupby(keys)[["t_prefill_s", "tpot_s"]].std().reset_index()
    agg = agg.merge(std, on=keys, suffixes=("", "_std"))

    out, params = [], {"allreduce": ar_params, "platforms": {}}
    for plat, g in agg.groupby("platform"):
        spec = plat_spec.get(plat)
        if spec is None:
            print(f"skip {plat}: not in configs/platforms.json")
            continue
        F = spec["peak_bf16_tflops"] * 1e12
        BW = spec["hbm_GBps"] * 1e9
        base = g[g["tp"] == 1]
        if base.empty:
            print(f"skip {plat}: no TP=1 data to calibrate")
            continue
        # eta_c from prefill at TP=1 (largest batches are most compute bound)
        big = base[base["batch"] >= base["batch"].quantile(0.5)]
        eta_c = np.median([work_terms(r, "prefill")[0] / (F * r["t_prefill_s"])
                           for _, r in big.iterrows()])
        # eta_m from decode at TP=1, small batches (memory bound)
        small = base[base["batch"] <= 8]
        if small.empty:
            small = base
        eta_m = np.median([work_terms(r, "decode")[1] / (BW * r["tpot_s"])
                           for _, r in small.iterrows()])
        params["platforms"][plat] = {"eta_c": eta_c, "eta_m": eta_m,
                                     "peak_tflops": F / 1e12, "hbm_GBps": BW / 1e9}
        for _, r in g.iterrows():
            p = int(r["tp"])
            ar = ar_params.get(f"{plat}:p{p}")
            if p > 1 and ar is None:
                print(f"WARNING {plat} tp={p}: no all-reduce fit, comm term = 0")
            L = r["num_layers"]
            for phase, meas in (("prefill", r["t_prefill_s"]), ("decode", r["tpot_s"])):
                fl, mem, msg = work_terms(r, phase)
                tc = fl / (p * F * eta_c)
                tm = mem / (p * BW * eta_m)
                tcomm = 2 * L * t_allreduce(msg, p, ar)
                pred = max(tc, tm) + tcomm
                out.append({"platform": plat, "tp": p, "batch": int(r["batch"]),
                            "phase": phase, "measured_s": meas,
                            "measured_std_s": r[f"{'t_prefill_s' if phase == 'prefill' else 'tpot_s'}_std"],
                            "pred_s": pred, "pred_comp_s": tc, "pred_mem_s": tm,
                            "pred_comm_s": tcomm,
                            "bound": "compute" if tc >= tm else "memory",
                            "calibration": p == 1,
                            "ape": abs(pred - meas) / meas})
    res = pd.DataFrame(out)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    res.to_csv(os.path.join(ROOT, "results", "model_predictions.csv"), index=False)
    summ = (res.groupby(["platform", "phase", "calibration"])["ape"].mean() * 100).round(1)
    params["mape_percent"] = {" / ".join(map(str, k)): v for k, v in summ.items()}
    with open(os.path.join(ROOT, "results", "model_params.json"), "w") as f:
        json.dump(params, f, indent=2, default=float)
    print(json.dumps(params, indent=2, default=float))
    print("\nMAPE (%) by platform / phase / calibration-set:\n", summ.to_string())


if __name__ == "__main__":
    main()
