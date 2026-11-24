#!/usr/bin/env python3
"""Produce every paper figure and table from results/.

Figures are drawn at their printed size (single column = 3.33 in) with fonts
>= 8 pt, as the ACM/PPoPP template requires, and written to figures/ and to
<paper>/figures/ if --paper is given.

  python analysis/model.py                       # first: fits the model
  python analysis/make_figures.py --paper ../template
"""
import argparse
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from model import ROOT, load_allreduce, load_bench  # noqa: E402

COL_W = 3.33
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.2,
    "lines.markersize": 4, "axes.grid": True, "grid.alpha": 0.3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
STYLE = {"h100": dict(color="#1f77b4", marker="o", label="H100 (NVLink)"),
         "rtx4090": dict(color="#d62728", marker="s", label="RTX 4090 (PCIe)"),
         "rtx4060": dict(color="#7f7f7f", marker="^", label="RTX 4060")}
FIG = os.path.join(ROOT, "figures")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def med(df, keys, col):
    g = df.groupby(keys)[col]
    return g.median().rename("med").to_frame().join(g.std().rename("std")).reset_index()


def fig_allreduce(ar):
    if ar.empty:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    for (plat, p), g in ar.groupby(["platform", "p"]):
        if p < 2:
            continue
        g = g.groupby("bytes", as_index=False)["busbw_GBps"].median()
        st = STYLE.get(plat, {})
        ax.plot(g["bytes"], g["busbw_GBps"], marker=st.get("marker"), color=st.get("color"),
                linestyle="-" if p == 2 else "--", label=f"{st.get('label', plat)}, {p} GPUs")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Message size (bytes)")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.legend(loc="upper left")
    save(fig, "allreduce_bw.pdf")


def fig_strong(df, batch):
    d = df[df["batch"] == batch]
    if d.empty:
        return
    d = d.assign(thr=d["decode_tok_s"])
    m = med(d, ["platform", "tp"], "thr")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.9))
    for plat, g in m.groupby("platform"):
        st = STYLE.get(plat, {})
        g = g.sort_values("tp")
        base = g.loc[g["tp"] == 1, "med"]
        if base.empty:
            continue
        sp = g["med"] / base.iloc[0]
        a1.errorbar(g["tp"], sp, yerr=g["std"] / base.iloc[0], capsize=2,
                    color=st.get("color"), marker=st.get("marker"), label=st.get("label", plat))
        a2.plot(g["tp"], sp / g["tp"] * 100, color=st.get("color"), marker=st.get("marker"))
    tps = sorted(m["tp"].unique())
    a1.plot(tps, tps, "k:", lw=0.8, label="ideal")
    a1.set_xlabel("TP degree $p$")
    a1.set_ylabel("Decode speedup $S(p)$")
    a1.set_xticks(tps)
    a2.set_xlabel("TP degree $p$")
    a2.set_ylabel("Efficiency $E(p)$ (%)")
    a2.set_xticks(tps)
    a2.set_ylim(0, 110)
    a1.legend(loc="lower right", fontsize=7)
    fig.tight_layout(w_pad=0.6)
    save(fig, "strong_scaling.pdf")


def fig_weak(df, per_gpu):
    rows = []
    for (plat, tp), g in df.groupby(["platform", "tp"]):
        g = g[g["batch"] == per_gpu * tp]
        if not g.empty:
            rows.append({"platform": plat, "tp": tp, "tpot_ms": g["tpot_s"].median() * 1e3,
                         "std": g["tpot_s"].std() * 1e3})
    if not rows:
        return
    m = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    for plat, g in m.groupby("platform"):
        st = STYLE.get(plat, {})
        g = g.sort_values("tp")
        ax.errorbar(g["tp"], g["tpot_ms"], yerr=g["std"], capsize=2, color=st.get("color"),
                    marker=st.get("marker"), label=st.get("label", plat))
    ax.set_xlabel(f"TP degree $p$ (batch = {per_gpu}$\\times p$)")
    ax.set_ylabel("Time per output token (ms)")
    ax.set_xticks(sorted(m["tp"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend()
    save(fig, "weak_scaling.pdf")


def fig_model():
    path = os.path.join(ROOT, "results", "model_predictions.csv")
    if not os.path.exists(path):
        return
    r = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for (plat, cal), g in r.groupby(["platform", "calibration"]):
        st = STYLE.get(plat, {})
        ax.scatter(g["measured_s"] * 1e3, g["pred_s"] * 1e3, s=12, color=st.get("color"),
                   marker=st.get("marker"), facecolors="none" if cal else st.get("color"),
                   label=f"{st.get('label', plat)} ({'calib., TP=1' if cal else 'predicted, TP>1'})")
    lo = min(r["measured_s"].min(), r["pred_s"].min()) * 1e3 * 0.8
    hi = max(r["measured_s"].max(), r["pred_s"].max()) * 1e3 * 1.2
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured time (ms)")
    ax.set_ylabel("Predicted time (ms)")
    ax.legend(fontsize=6.5, loc="upper left")
    save(fig, "model_vs_measured.pdf")


def fig_breakdown():
    path = os.path.join(ROOT, "results", "kernel_breakdown.csv")
    if not os.path.exists(path):
        return
    b = pd.read_csv(path)  # columns: run, category, time_ms
    piv = b.pivot_table(index="run", columns="category", values="time_ms", aggfunc="sum").fillna(0)
    share = piv.div(piv.sum(axis=1), axis=0) * 100
    order = [c for c in ["GEMM", "Attention", "Communication", "Other"] if c in share.columns]
    fig, ax = plt.subplots(figsize=(COL_W, 0.4 + 0.35 * len(share)))
    left = np.zeros(len(share))
    colors = {"GEMM": "#4c72b0", "Attention": "#55a868", "Communication": "#c44e52", "Other": "#bbbbbb"}
    for c in order:
        ax.barh(share.index, share[c], left=left, color=colors[c], label=c)
        left += share[c].to_numpy()
    ax.set_xlabel("Share of GPU kernel time (%)")
    ax.set_xlim(0, 100)
    ax.legend(ncol=4, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.grid(axis="y", visible=False)
    save(fig, "kernel_breakdown.pdf")


def tab_ablation(all_df, batch):
    d = all_df[(all_df["batch"] == batch) & (all_df["tag"].isin(["main", "no_car", "eager"]))]
    if d.empty:
        return
    m = d.groupby(["platform", "tp", "tag"])["tpot_s"].median().unstack("tag") * 1e3
    names = {"main": "default", "no_car": "NCCL all-reduce", "eager": "no CUDA graphs"}
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Platform & $p$ & default & NCCL AR & no graphs \\",
             r" & & \multicolumn{3}{c}{TPOT (ms), batch " + str(batch) + r"} \\", r"\midrule"]
    for (plat, tp), r in m.iterrows():
        def cell(t):
            if t not in r or pd.isna(r[t]):
                return "--"
            if t == "main":
                return f"{r[t]:.2f}"
            return f"{r[t]:.2f} ({r[t] / r['main']:.2f}$\\times$)" if "main" in r else f"{r[t]:.2f}"
        lines.append(f"{STYLE.get(plat, {}).get('label', plat)} & {tp} & {cell('main')} & "
                     f"{cell('no_car')} & {cell('eager')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(FIG, "tab_ablation.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab_ablation.tex  (", ", ".join(names.values()), ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", help="paper folder; figures are copied to <paper>/figures")
    ap.add_argument("--strong-batch", type=int, default=32)
    ap.add_argument("--weak-per-gpu", type=int, default=8)
    a = ap.parse_args()
    all_df = load_bench(tags=None)
    main_df = all_df[all_df["tag"] == "main"]
    fig_allreduce(load_allreduce())
    fig_strong(main_df, a.strong_batch)
    fig_weak(main_df, a.weak_per_gpu)
    fig_model()
    fig_breakdown()
    tab_ablation(all_df, a.strong_batch)
    if a.paper:
        dst = os.path.join(a.paper, "figures")
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(FIG):
            if f.endswith((".pdf", ".tex")):
                shutil.copy(os.path.join(FIG, f), dst)
        print("copied figures to", dst)


if __name__ == "__main__":
    main()
