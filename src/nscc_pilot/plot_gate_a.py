"""Gate A hook figure: compression ratio (x) vs Recall@FPR (y), faceted by
attack span, one line per detector (Protocol I). Plus a Protocol I-vs-II delta
panel to decompose info-loss vs representation-mismatch."""
from __future__ import annotations

from collections import defaultdict
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .schema import SPANS


def _avg(rows, **filt):
    v = [r["recall_at_fpr"] for r in rows
         if all(r[k] == val for k, val in filt.items())
         and r["recall_at_fpr"] == r["recall_at_fpr"]]
    return sum(v) / len(v) if v else float("nan")


def plot_hook_figure(rows: List[dict], out_dir: str) -> str:
    detectors = sorted({r["detector"] for r in rows})
    ratios = sorted({r["ratio"] for r in rows if r["compressor"] != "none"})
    x = ratios

    fig, axes = plt.subplots(1, len(SPANS), figsize=(4.2 * len(SPANS), 3.6),
                             sharey=True)
    for ax, span in zip(axes, SPANS):
        for d in detectors:
            ys = [_avg(rows, detector=d, protocol="I", span=span, ratio=r)
                  for r in ratios]
            ax.plot(x, ys, marker="o", label=d)
            base = _avg(rows, detector=d, protocol="I", span=span,
                        compressor="none", ratio=1.0)
            if base == base:
                ax.axhline(base, ls=":", lw=0.8, alpha=0.4)
        ax.set_title(f"span = {span}")
        ax.set_xlabel("tokens kept (compression ratio)")
        ax.invert_xaxis()
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Recall @ FPR=0.05")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Gate A hook: safety recall vs compression, by attack time-span")
    fig.tight_layout()
    path = f"{out_dir}/hook_figure.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path
