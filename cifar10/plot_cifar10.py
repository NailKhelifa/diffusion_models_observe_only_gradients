"""
Figure 3: error decomposition and KL-bound estimates during CIFAR-10 training.

Reads <runs_root>/ch<c>/seed<s>/hist.json written by train_cifar10.py and plots
median + inter-quartile range over seeds:
  left : full error E|r|^2, gradient part (critic estimate), solenoidal part (difference)
  right: K * full ("Girsanov bound") and K * gradient part ("Ours"), with
         K = 0.5 * (T - t0) * mean_s sigma_s^2.

Usage:  python plot_cifar10.py --runs_root logs_v24 --ch 96 --seeds 45 47 49 52
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Noise schedule of train_cifar10.py: VP-SDE, linear beta from 0.1 to 20, s in [t0, T].
BETA_MIN, BETA_MAX, T_SDE, T0 = 0.1, 20.0, 1.0, 1e-3

def sigma(t):
    int_beta = BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2
    return np.sqrt(1.0 - np.exp(-int_beta) + 1e-8)


def _set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.dpi": 200,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "grid.alpha"    : 0.3,
        "grid.linewidth": 0.5,
        "axes.grid"      : True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def plot_combined_metrics_from_hist(hists_per_seed, out_path):
    """Plots both two-phase decomposition and KL bounds side-by-side."""
    _set_paper_style()
    h0 = hists_per_seed[0]
    epochs = np.array(h0["eval_epoch"] if "eval_epoch" in h0 else h0["fid_epoch"])

    full_mat = np.array([h["dsm_full_l2"] for h in hists_per_seed])
    grad_mat = np.array([h["dsm_grad_l2"] for h in hists_per_seed])
    sol_mat = np.maximum(full_mat - grad_mat, 0.0)

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.5))

    # ── left panel: decomposition ───────────────────────────────────────────
    ax = axs[0]
    for mat, color, label in [
        (full_mat, "#444444", r"Total $\|e\|^2$"),
        (grad_mat, "#E76F51", r"Gradient $\|\Pi_{\mathcal{G}} e\|^2$"),
        (sol_mat, "#0D695E", r"Solenoidal $\|\Pi_{\mathcal{G}^\perp} e\|^2$"),
    ]:
        med = np.median(mat, axis=0)
        lo, hi = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
        ax.plot(epochs, med, lw=2.2, color=color, label=label)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"$\mathbb{E}_{s,x_s}[\,\cdot\,]$")
    ax.grid(True)
    ax.legend(loc="best")

    # ── right panel: KL-bound estimates ─────────────────────────────────────
    ax = axs[1]
    ts = np.linspace(T0, T_SDE, 256)
    K = 0.5 * (T_SDE - T0) * float(np.mean(sigma(ts) ** 2))
    bound_full = K * full_mat
    bound_grad = K * grad_mat
    for mat, color, label in [
        (bound_full, "#C0392B", r"Girsanov bound"),
        (bound_grad, "#0D695E", r"Ours"),
    ]:
        med = np.median(mat, axis=0)
        lo, hi = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
        ax.plot(epochs, med, lw=2.2, color=color, label=label)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.18)
    ax.fill_between(epochs, np.median(bound_grad, 0), np.median(bound_full, 0),
                    color="#888888", alpha=0.12, label="Inobservability gap")
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("KL upper-bound estimate")
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="logs_v24")
    p.add_argument("--ch", type=int, default=96)
    p.add_argument("--seeds", type=int, nargs="+", default=[45, 47, 49, 52])
    p.add_argument("--out_dir", default="./figures_reproduced")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hists = []
    for s in args.seeds:
        with open(Path(args.runs_root) / f"ch{args.ch}" / f"seed{s}" / "hist.json") as f:
            hists.append(json.load(f))
    plot_combined_metrics_from_hist(hists, out_dir / "combined_metrics")
    print(f"wrote {out_dir / 'combined_metrics'}.pdf/.png")


if __name__ == "__main__":
    main()
