"""
plot_results.py - Generate comparison figures from experiment CSVs.

Reads results/baseline.csv and results/imrfit.csv, then produces:
  1. Throughput vs Epoch (img/s)
  2. RMW Ratio vs Epoch
  3. Displacement D(e) vs Epoch (IMR-Fit only)
  4. Delta RMW per Epoch

Figures are saved to results/figures/ as PNG.

Usage::

    python experiments/plot_results.py
    python experiments/plot_results.py --results-dir /path/to/results --dpi 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"[plot] Warning: {path} not found, skipping.")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(records: List[Dict], key: str, dtype=float) -> List:
    return [dtype(r[key]) for r in records if key in r]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(results_dir: Path = RESULTS_DIR, dpi: int = 150) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for Linux servers
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_csv(results_dir / "baseline.csv")
    imrfit = load_csv(results_dir / "imrfit.csv")

    if not baseline and not imrfit:
        print("[plot] No result files found. Run the experiments first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Figure 1: Throughput vs Epoch
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    if baseline:
        ax.plot(col(baseline, "epoch", int), col(baseline, "throughput_img_s"),
                label="Baseline (naive IMR)", marker="o", linewidth=2)
    if imrfit:
        ax.plot(col(imrfit, "epoch", int), col(imrfit, "throughput_img_s"),
                label="IMR-Fit", marker="s", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Throughput (images/s)")
    ax.set_title("Training Throughput vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = figures_dir / "throughput_vs_epoch.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[plot] Saved {out}")

    # -----------------------------------------------------------------------
    # Figure 2: RMW Ratio vs Epoch
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    if baseline:
        ax.plot(col(baseline, "epoch", int), col(baseline, "rmw_ratio"),
                label="Baseline", marker="o", linewidth=2, color="tab:orange")
    if imrfit:
        ax.plot(col(imrfit, "epoch", int), col(imrfit, "rmw_ratio"),
                label="IMR-Fit", marker="s", linewidth=2, color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMW Ratio (rmw / writes)")
    ax.set_title("Read-Modify-Write Ratio vs Epoch")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = figures_dir / "rmw_ratio_vs_epoch.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[plot] Saved {out}")

    # -----------------------------------------------------------------------
    # Figure 3: Displacement D(e) vs Epoch  (IMR-Fit only)
    # -----------------------------------------------------------------------
    if imrfit and "displacement_before" in imrfit[0]:
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = col(imrfit, "epoch", int)
        d_before = col(imrfit, "displacement_before")
        d_after = col(imrfit, "displacement_after")
        ax.plot(epochs, d_before, label="D(e) before migration",
                marker="^", linewidth=2, linestyle="--", color="tab:red")
        ax.plot(epochs, d_after, label="D(e) after migration",
                marker="v", linewidth=2, color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Displacement D(e) = misplaced / total")
        ax.set_title("Block Displacement vs Epoch (IMR-Fit)")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = figures_dir / "displacement_vs_epoch.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"[plot] Saved {out}")

    # -----------------------------------------------------------------------
    # Figure 4: Delta RMW per Epoch
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    if baseline:
        ax.bar(
            [e - 0.2 for e in col(baseline, "epoch", int)],
            col(baseline, "delta_rmw"),
            width=0.35, label="Baseline", color="tab:orange", alpha=0.7,
        )
    if imrfit:
        ax.bar(
            [e + 0.2 for e in col(imrfit, "epoch", int)],
            col(imrfit, "delta_rmw"),
            width=0.35, label="IMR-Fit", color="tab:blue", alpha=0.7,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMW operations (delta)")
    ax.set_title("Per-Epoch RMW Operations")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = figures_dir / "delta_rmw_vs_epoch.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[plot] Saved {out}")

    print(f"\n[plot] All figures saved to {figures_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot IMR-Fit experiment results")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                   help="Directory containing baseline.csv and imrfit.csv")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_all(results_dir=args.results_dir, dpi=args.dpi)
