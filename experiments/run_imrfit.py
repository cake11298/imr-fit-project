"""
run_imrfit.py - Full IMR-Fit pipeline experiment.

Per-epoch loop:
  1. Wrap DataLoader with DataLoaderProfiler
  2. Train one epoch (mock forward pass)
  3. Compute S(b) for all profiled blocks via Scorer
  4. Generate MigrationPlan (budget-constrained)
  5. Apply plan (update in-memory placement map)
  6. Poll IMRSimMonitor for RMW stats
  7. Record metrics to results/imrfit.csv

Usage::

    python experiments/run_imrfit.py --epochs 10 --dry-run
    python experiments/run_imrfit.py --epochs 20 \\
        --data-root /mnt/imrsim/imagenet \\
        --budget 5 --theta 0.55
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from imrfit.monitor import IMRSimMonitor
from imrfit.profiler import DataLoaderProfiler
from imrfit.scheduler import MigrationScheduler
from imrfit.scorer import BlockPlacement, Scorer, ScorerConfig
from workloads.dataloader_factory import make_dataloader
from workloads.synthetic_cv import SyntheticCVConfig, generate_dataset


RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_imrfit(
    epochs: int = 10,
    batch_size: int = 64,
    data_root: str = "/mnt/imrsim/imagenet",
    dry_run: bool = False,
    n_classes: int = 10,
    m_images: int = 100,
    # Scorer hyperparameters
    w_freq: float = 0.35,
    w_seq: float = 0.30,
    w_size: float = 0.15,
    w_rec: float = 0.20,
    theta: float = 0.55,
    recency_lambda: float = 1.0,
    # Scheduler
    budget: int = 10,
    verbose: bool = True,
) -> List[dict]:
    """
    Run the full IMR-Fit experiment.

    Returns a list of per-epoch metric dicts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    use_mock = dry_run or not os.path.exists(data_root)

    # --- Dataset preparation ---
    if not os.path.exists(data_root) and not dry_run:
        if verbose:
            print(f"[imrfit] Dataset not found at {data_root}, generating...")
        cfg = SyntheticCVConfig(root=data_root, n_classes=n_classes, m_images=m_images)
        generate_dataset(cfg)
    elif dry_run and verbose:
        print("[imrfit] DRY RUN — using mock dataloader, no real device I/O")

    base_loader = make_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        use_mock=use_mock,
    )

    scorer_cfg = ScorerConfig(
        w_freq=w_freq, w_seq=w_seq, w_size=w_size, w_rec=w_rec,
        theta=theta, recency_lambda=recency_lambda,
    )
    scorer = Scorer(scorer_cfg)
    scheduler = MigrationScheduler(budget=budget, theta=theta)
    monitor = IMRSimMonitor(dry_run=dry_run)

    # Current placement map: block_id -> BlockPlacement
    current_placements: Dict[int, BlockPlacement] = {}

    records: List[dict] = []
    if verbose:
        print(f"[imrfit] Starting {epochs} epochs, batch_size={batch_size}")
        print(f"         Scorer: w_freq={w_freq}, w_seq={w_seq}, "
              f"w_size={w_size}, w_rec={w_rec}, theta={theta}")
        print(f"         Migration budget: {budget} blocks/epoch")

    monitor.poll()  # baseline snapshot

    for epoch in range(1, epochs + 1):
        epoch_start = time.monotonic()

        # --- Step 1: Profile this epoch ---
        profiler = DataLoaderProfiler(
            base_loader,
            mount_point="/mnt/imrsim",
            recency_lambda=recency_lambda,
        )

        batches_processed = 0
        for batch in profiler:
            batches_processed += 1
            if not dry_run:
                time.sleep(0.001)  # simulate compute

        epoch_elapsed = time.monotonic() - epoch_start

        # --- Step 2: Score all profiled blocks ---
        metrics = profiler.compute_scores_input()
        score_results = scorer.score_all(metrics)
        summary = scorer.score_summary(score_results)

        # --- Step 3: Displacement before migration ---
        d_before = Scorer.displacement(score_results, current_placements)

        # --- Step 4: Migration plan ---
        plan = scheduler.plan(epoch, score_results, current_placements)
        current_placements = MigrationScheduler.apply_plan(plan, current_placements)

        # --- Step 5: Displacement after migration ---
        d_after = Scorer.displacement(score_results, current_placements)

        # --- Step 6: Device stats ---
        stats = monitor.poll()
        delta = monitor.delta()

        throughput_imgs_sec = (
            len(base_loader.dataset) / epoch_elapsed
            if hasattr(base_loader, "dataset")
            else batches_processed * batch_size / epoch_elapsed
        )

        record = {
            "epoch": epoch,
            "strategy": "imrfit",
            "elapsed_s": round(epoch_elapsed, 4),
            "throughput_img_s": round(throughput_imgs_sec, 2),
            "total_rmw": stats.total_rmw,
            "total_writes": stats.total_writes,
            "rmw_ratio": round(stats.overall_rmw_ratio, 4),
            "delta_rmw": delta.total_rmw if delta else 0,
            "displacement_before": round(d_before, 4),
            "displacement_after": round(d_after, 4),
            "blocks_scored": summary.get("total_blocks", 0),
            "top_blocks": summary.get("top_blocks", 0),
            "bottom_blocks": summary.get("bottom_blocks", 0),
            "migrations_issued": len(plan.operations),
            "promote_to_top": len(plan.to_top),
            "demote_to_bottom": len(plan.to_bottom),
        }
        records.append(record)

        if verbose:
            print(
                f"  epoch {epoch:3d}/{epochs} | "
                f"{throughput_imgs_sec:7.1f} img/s | "
                f"RMW: {record['rmw_ratio']:.3f} | "
                f"D: {d_before:.3f}->{d_after:.3f} | "
                f"mig: {len(plan.operations)}"
            )

    # --- Write CSV ---
    csv_path = RESULTS_DIR / "imrfit.csv"
    _write_csv(csv_path, records)
    if verbose:
        print(f"[imrfit] Results saved to {csv_path}")
        mig_total = scheduler.total_migrations()
        print(f"[imrfit] Total migrations across all epochs: {mig_total}")

    return records


def _write_csv(path: Path, records: List[dict]) -> None:
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IMR-Fit full pipeline experiment")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-root", default="/mnt/imrsim/imagenet")
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--images-per-class", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--w-freq", type=float, default=0.35)
    p.add_argument("--w-seq", type=float, default=0.30)
    p.add_argument("--w-size", type=float, default=0.15)
    p.add_argument("--w-rec", type=float, default=0.20)
    p.add_argument("--theta", type=float, default=0.55)
    p.add_argument("--lambda", type=float, default=1.0, dest="recency_lambda")
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_imrfit(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_root=args.data_root,
        dry_run=args.dry_run,
        n_classes=args.classes,
        m_images=args.images_per_class,
        w_freq=args.w_freq,
        w_seq=args.w_seq,
        w_size=args.w_size,
        w_rec=args.w_rec,
        theta=args.theta,
        recency_lambda=args.recency_lambda,
        budget=args.budget,
        verbose=not args.quiet,
    )
