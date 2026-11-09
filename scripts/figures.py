"""Sensitivity figure (distillation weight and bridge tolerance) from a reports/summary.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# Okabe-Ito palette, with distinct markers so it also reads in greyscale
STYLE = {
    "cosine":     ("#0072B2", "o", "-"),
    "mse":        ("#E69F00", "s", "-"),
    "smooth_l1":  ("#009E73", "^", "-"),
    "relational": ("#D55E00", "D", "--"),
}
LABEL = {"cosine": "cosine", "mse": "MSE",
         "smooth_l1": "smooth-$L_1$", "relational": "relational"}


def _rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.8,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _panel(ax, frame, xcol, baseline, *, logx=False, xlabel="", title=""):
    for loss, (colour, marker, dash) in STYLE.items():
        d = frame[frame["distill_loss"] == loss].sort_values(xcol)
        if d.empty:
            continue
        x = d[xcol].to_numpy(dtype=float)
        y = d["test_macro_f1_mean"].to_numpy(dtype=float)
        ax.plot(x, y, color=colour, marker=marker, linestyle=dash,
                label=LABEL[loss], zorder=3, clip_on=False)
        if "test_macro_f1_std" in d:
            sd = d["test_macro_f1_std"].to_numpy(dtype=float) / np.sqrt(max(len(x), 1))
            ax.fill_between(x, y - sd, y + sd, color=colour, alpha=0.13,
                            linewidth=0, zorder=2)
    if np.isfinite(baseline):
        ax.axhline(baseline, color="0.35", linestyle=(0, (4, 3)), linewidth=1.0,
                   zorder=1, label="audio only")
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(sorted(frame[xcol].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.8", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, help="directory holding summary.csv")
    parser.add_argument("--encoder", default="CNN")
    parser.add_argument("--view", default="ct_clinical")
    parser.add_argument("--age", default="5", help="tolerance held fixed in the weight panel")
    parser.add_argument("--weight", type=float, default=1.0,
                        help="weight held fixed in the tolerance panel")
    parser.add_argument("--seeds", type=int, default=None,
                        help="keep only cells with exactly this many seeds")
    args = parser.parse_args()

    reports = Path(args.reports)
    summary = pd.read_csv(reports / "summary.csv")
    summary["age_tolerance"] = summary["age_tolerance"].astype(str)

    base = summary[(summary.method == "baseline_audio") & (summary.encoder == args.encoder)]
    baseline = float(base["test_macro_f1_mean"].iloc[0]) if len(base) else np.nan

    d = summary[(summary.method == "distilled") & (summary.encoder == args.encoder)
                & (summary.view == args.view) & (summary.ablation == "none")]
    if args.seeds is not None:
        d = d[d["seeds"] == args.seeds]

    _rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)

    _panel(axes[0], d[d.age_tolerance == args.age], "knob", baseline, logx=True,
           xlabel=r"auxiliary weight $w$", title="(a) objective $\\times$ weight")
    tol = d[np.isclose(d["knob"], args.weight)].copy()
    tol["age_num"] = pd.to_numeric(tol["age_tolerance"], errors="coerce")
    _panel(axes[1], tol.rename(columns={"age_num": "tol"}), "tol", baseline,
           xlabel=r"bridge tolerance $\tau$ (years)", title="(b) matching tolerance")
    axes[1].set_xticks([3, 5, 7])

    axes[0].set_ylabel("patient macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.06), handlelength=2.0,
               columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(reports / f"fig_sensitivity.{ext}", bbox_inches="tight")
    print(f"wrote {reports/'fig_sensitivity.pdf'} and .png")


if __name__ == "__main__":
    main()
