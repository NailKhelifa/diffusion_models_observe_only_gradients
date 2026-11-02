"""
Figure 2: Spearman correlation between FID and the full error / gradient part,
mean +- std over seeds, per model capacity.
Reads the table written by diagnostic_fmnist.py.

Usage:  python plot_spearman.py [--cache-dir ./diagnostic_cache] [--out-dir ./figures_reproduced]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── plot style (matches the rest of the paper) ─────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 200,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
})
COL_FULL = "#E63946"
COL_GRAD = "#2A9D8F"

# ── paths ─────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--cache-dir", default="./diagnostic_cache")
_p.add_argument("--out-dir", default="./figures_reproduced")
_args = _p.parse_args()
CACHE = Path(_args.cache_dir) / "spearman_table_score_dsm.json"
OUT_DIR = Path(_args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load + sort by capacity (tiny → small → full) ─────────────────────────
tbl = json.loads(CACHE.read_text())
order = {"tiny": 0, "small": 1, "full": 2}
tbl = sorted([r for r in tbl if "reason" not in r],
             key=lambda r: order.get(r["capacity"], 99))

caps = [r["capacity"] for r in tbl]

# means ± std across seeds (per-seed stats are stored in the JSON)
full_mean = np.array([np.mean(r["rho_full_per_seed"]) for r in tbl])
full_std  = np.array([np.std(r["rho_full_per_seed"], ddof=1) for r in tbl])
grad_mean = np.array([np.mean(r["rho_grad_per_seed"]) for r in tbl])
grad_std  = np.array([np.std(r["rho_grad_per_seed"], ddof=1) for r in tbl])

# ── plot ──────────────────────────────────────────────────────────────────
x = np.arange(len(caps))
w = 0.36

fig, ax = plt.subplots(figsize=(7.5, 5.0))

ax.bar(x - w/2, full_mean, w, yerr=full_std, color=COL_FULL,
       capsize=4, error_kw=dict(elinewidth=1.0, capthick=1.0, ecolor="0.2"),
       label=r"$\rho(\mathrm{FID},\,\mathcal{E}_{\mathrm{full}})$")
ax.bar(x + w/2, grad_mean, w, yerr=grad_std, color=COL_GRAD,
       capsize=4, error_kw=dict(elinewidth=1.0, capthick=1.0, ecolor="0.2"),
       label=r"$\rho(\mathrm{FID},\,\mathcal{E}_{\mathrm{grad}})$")

ax.set_xticks(x)
ax.set_xticklabels(caps)
ax.set_xlabel("Model capacity")
ax.set_ylabel(r"Spearman $\rho$")

# Tighten y-axis so the meaningful range is visible.
y_lo = max(0.0, min(full_mean - full_std).item() - 0.05)
ax.set_ylim(y_lo, 1.02)
ax.grid(True, axis="y")

# Legend ABOVE the plot, two columns -> never overlaps the bars.
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    handlelength=1.4,
    columnspacing=1.6,
    borderaxespad=0,
)

plt.tight_layout()
for ext in ("pdf", "png"):
    out = OUT_DIR / f"spearman_bars_score_dsm_v2.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
    print(f"saved -> {out}")
plt.close(fig)