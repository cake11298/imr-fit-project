"""plot_results.py — Module 6 plotter.

Reads a finished run directory (default `results/latest/`) and emits five
publication-ready figures into `<run_dir>/figures/`:

    Figure 1 — S(b) violin plot, three scenarios side by side.
    Figure 2 — Throughput vs epoch, four strategies.
    Figure 3 — RMW count bar chart, scenario × strategy.
    Figure 4 — D(e) convergence curve.
    Figure 5 — Z(b) bimodal histogram (text vs image sizes).

All figures are rendered at 300 DPI.

Usage::

    python plot_results.py --results-dir results/latest
    python plot_results.py --results-dir results/run_20260427T140000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive — works on headless dev boxes

import matplotlib.pyplot as plt

try:
    import seaborn as sns          # optional, only used for nicer styling
    _HAS_SEABORN = True
except Exception:  # pragma: no cover
    _HAS_SEABORN = False

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


SCENARIOS = ["scenario_a", "scenario_b", "scenario_c"]
STRATEGIES = ["cmr_baseline", "naive_imr", "tracklace", "imrfit"]
STRATEGY_LABELS = {
    "cmr_baseline": "CMR Baseline",
    "naive_imr":    "Naive IMR",
    "tracklace":    "TrackLace",
    "imrfit":       "IMR-Fit (4D)",
}
STRATEGY_COLORS = {
    "cmr_baseline": "#4daf4a",
    "naive_imr":    "#e41a1c",
    "tracklace":    "#377eb8",
    "imrfit":       "#984ea3",
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_placement_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _load_run(run_dir: Path) -> Dict[str, Any]:
    """Discover all relevant artefacts under a run directory."""
    out: Dict[str, Any] = {
        "summary": _load_json(run_dir / "summary.json"),
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        decisions = _load_placement_jsonl(
            run_dir / f"03_{scenario}_placement.jsonl"
        )
        analyze_summary = _load_json(run_dir / f"03_{scenario}_summary.json")
        replay = _load_json(run_dir / f"04_{scenario}_replay.json")
        out["scenarios"][scenario] = {
            "decisions": decisions,
            "analyze": analyze_summary,
            "replay": replay,
        }
    return out


# ---------------------------------------------------------------------------
# Figure 1: S(b) violin plot
# ---------------------------------------------------------------------------


def fig1_score_violin(run: Dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    data: List[List[float]] = []
    labels: List[str] = []
    for scenario in SCENARIOS:
        rows = run["scenarios"][scenario]["decisions"]
        scores = [r["S"] for r in rows] if rows else []
        data.append(scores or [0.0])
        labels.append(scenario.split("_")[-1].upper())

    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for pc, color in zip(parts["bodies"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"Scenario {l}" for l in labels])
    ax.set_ylabel("S(b) score")
    ax.set_title("Figure 1: S(b) distribution by scenario")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate variance to highlight the killer figure point.
    for i, scenario in enumerate(SCENARIOS):
        sv = (run["scenarios"][scenario]["analyze"] or {}).get("score_variance", {})
        if sv:
            ax.text(i + 1, 0.05, f"var={sv.get('variance', 0):.3f}",
                    ha="center", fontsize=8, color="dimgray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: throughput vs epoch
# ---------------------------------------------------------------------------


def fig2_throughput_vs_epoch(run: Dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        replay = run["scenarios"][scenario]["replay"] or {}
        ax.set_title(f"Scenario {scenario.split('_')[-1].upper()}")
        for strategy in STRATEGIES:
            rec = replay.get(strategy)
            if not rec:
                continue
            y = rec.get("epoch_throughput") or []
            x = list(range(1, len(y) + 1))
            ax.plot(x, y,
                    label=STRATEGY_LABELS[strategy],
                    color=STRATEGY_COLORS[strategy],
                    linewidth=1.6)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Throughput (MB/s)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Figure 2: Throughput convergence per epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: RMW count bar chart
# ---------------------------------------------------------------------------


def fig3_rmw_bars(run: Dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    n_scen = len(SCENARIOS)
    n_strat = len(STRATEGIES)
    bar_width = 0.18
    x_base = list(range(n_scen))
    for i, strategy in enumerate(STRATEGIES):
        ys = []
        for scenario in SCENARIOS:
            replay = run["scenarios"][scenario]["replay"] or {}
            rec = replay.get(strategy) or {}
            ys.append(rec.get("rmw_count", 0))
        offsets = [x + (i - n_strat / 2) * bar_width + bar_width / 2
                   for x in x_base]
        ax.bar(offsets, ys, bar_width,
               label=STRATEGY_LABELS[strategy],
               color=STRATEGY_COLORS[strategy])

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"Scenario {s.split('_')[-1].upper()}"
                        for s in SCENARIOS])
    ax.set_ylabel("RMW count")
    ax.set_title("Figure 3: RMW operations per scenario × strategy")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: D(e) convergence curve
# ---------------------------------------------------------------------------


def fig4_displacement(run: Dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    for scenario in SCENARIOS:
        replay = run["scenarios"][scenario]["replay"] or {}
        rec = replay.get("imrfit") or {}
        ys = rec.get("displacement") or []
        if not ys:
            continue
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, marker="o", markersize=3,
                label=f"Scenario {scenario.split('_')[-1].upper()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("D(e) — fraction misplaced")
    ax.set_title("Figure 4: IMR-Fit displacement convergence")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Z(b) bimodal histogram
# ---------------------------------------------------------------------------


def fig5_z_bimodal(run: Dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    # Pull Z values from scenario_a (typically the richest dataset).
    rows = run["scenarios"]["scenario_a"]["decisions"] \
        or run["scenarios"]["scenario_c"]["decisions"]
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    z_values = [r["Z"] for r in rows]
    ax.hist(z_values, bins=40, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Z(b) — mean I/O size / block size")
    ax.set_ylabel("Number of blocks")
    ax.set_title("Figure 5: Z(b) bimodal distribution (text vs image)")
    ax.grid(True, alpha=0.3, axis="y")

    # Mark expected modes if visible
    for mark, label in [(2_048 / (128 * 1024 * 1024), "text ~2 KB"),
                        (1 * 1024 * 1024 / (128 * 1024 * 1024), "image ~1 MB")]:
        if 0 <= mark <= 1:
            ax.axvline(mark, color="darkred", linestyle="--", linewidth=1,
                       alpha=0.8)
            ax.text(mark, ax.get_ylim()[1] * 0.9, label,
                    rotation=90, va="top", ha="right",
                    fontsize=8, color="darkred")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results/latest")
    p.add_argument("--out-dir", default=None,
                   help="Where to write figures (default: <results-dir>/figures)")
    args = p.parse_args(argv)

    run_dir = Path(args.results_dir).resolve()
    if not run_dir.exists():
        print(f"[plot] not found: {run_dir}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir or (run_dir / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if _HAS_SEABORN:
        sns.set_context("paper", font_scale=1.05)
        sns.set_style("whitegrid")

    run = _load_run(run_dir)
    figs = [
        ("figure1_score_violin.png",       fig1_score_violin),
        ("figure2_throughput_vs_epoch.png", fig2_throughput_vs_epoch),
        ("figure3_rmw_bars.png",           fig3_rmw_bars),
        ("figure4_displacement.png",       fig4_displacement),
        ("figure5_z_bimodal.png",          fig5_z_bimodal),
    ]
    for name, fn in figs:
        path = out_dir / name
        try:
            fn(run, path)
            print(f"[plot] wrote {path}", file=sys.stderr)
        except Exception as exc:  # pragma: no cover
            print(f"[plot] {name} failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
