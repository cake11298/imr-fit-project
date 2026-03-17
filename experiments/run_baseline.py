"""
run_baseline.py - Baseline experiment: naive IMR with no block placement optimisation.

All blocks are placed on BOTTOM tracks (worst case RMW), simulating a vanilla
IMR drive with no intelligence.  Results are written to results/baseline.csv.

Usage::

    python experiments/run_baseline.py --epochs 10 --dry-run
    python experiments/run_baseline.py --epochs 20 --data-root /mnt/imrsim/imagenet
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from imrfit.monitor import IMRSimMonitor
from workloads.dataloader_factory import make_dataloader
from workloads.synthetic_cv import SyntheticCVConfig, generate_dataset


RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_baseline(
    epochs: int = 10,
    batch_size: int = 64,
    data_root: str = "/mnt/imrsim/imagenet",
    dry_run: bool = False,
    n_classes: int = 10,
    m_images: int = 100,
    imrsim_util: str = "~/IMRSim/imrsim_util/imrsim_util",
    device: str = "/dev/mapper/imrsim",
    verbose: bool = True,
) -> List[dict]:
    """
    Run the baseline (no-optimisation) experiment.

    Returns a list of per-epoch metric dicts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    use_mock = dry_run or not os.path.exists(data_root)

    # --- Dataset preparation ---
    if not os.path.exists(data_root) and not dry_run:
        if verbose:
            print(f"[baseline] Dataset not found at {data_root}, generating...")
        cfg = SyntheticCVConfig(root=data_root, n_classes=n_classes, m_images=m_images)
        generate_dataset(cfg)
    elif dry_run:
        if verbose:
            print(f"[baseline] DRY RUN — using mock dataloader, no real device I/O")

    loader = make_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        use_mock=use_mock,
    )
    monitor = IMRSimMonitor(
        device=device,
        imrsim_util=os.path.expanduser(imrsim_util),
        dry_run=dry_run,
    )

    records: List[dict] = []
    if verbose:
        print(f"[baseline] Starting {epochs} epochs, batch_size={batch_size}")
        print(f"           Placement strategy: ALL BOTTOM (naive IMR)")
        if not dry_run:
            print(f"           imrsim_util: {os.path.expanduser(imrsim_util)}")
            print(f"           device:      {device}")

    for epoch in range(1, epochs + 1):
        # Reset IMRSim counters before each epoch so per-epoch delta is clean
        monitor.reset_stats()

        epoch_start = time.monotonic()
        batches_processed = 0

        for batch in loader:
            batches_processed += 1
            # Simulate some compute time (in a real run, this is the DNN forward pass)
            if not dry_run:
                time.sleep(0.001)

        epoch_elapsed = time.monotonic() - epoch_start
        # Poll after epoch to capture RMW counts accumulated during this epoch
        stats = monitor.poll()
        delta = monitor.delta()

        throughput_imgs_sec = (
            len(loader.dataset) / epoch_elapsed if hasattr(loader, "dataset") else
            batches_processed * batch_size / epoch_elapsed
        )

        record = {
            "epoch": epoch,
            "strategy": "baseline",
            "elapsed_s": round(epoch_elapsed, 4),
            "throughput_img_s": round(throughput_imgs_sec, 2),
            # With reset_stats() before each epoch, stats.total_* == per-epoch counts
            "epoch_rmw": stats.total_rmw,
            "epoch_writes": stats.total_writes,
            "rmw_ratio": round(stats.overall_rmw_ratio, 4),
            "displacement": 1.0,  # Baseline: all blocks misplaced (all on BOTTOM)
        }
        records.append(record)

        if verbose:
            print(
                f"  epoch {epoch:3d}/{epochs} | "
                f"{throughput_imgs_sec:7.1f} img/s | "
                f"RMW ratio: {record['rmw_ratio']:.3f} | "
                f"epoch_rmw: {record['epoch_rmw']}"
            )

    # --- Write CSV ---
    csv_path = RESULTS_DIR / "baseline.csv"
    _write_csv(csv_path, records)
    if verbose:
        print(f"[baseline] Results saved to {csv_path}")

    return records


def _write_csv(path: Path, records: List[dict]) -> None:
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IMR-Fit baseline experiment (no optimisation)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-root", default="/mnt/imrsim/imagenet")
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--images-per-class", type=int, default=100)
    p.add_argument("--imrsim-util", default="~/IMRSim/imrsim_util/imrsim_util",
                   help="Path to imrsim_util binary")
    p.add_argument("--device", default="/dev/mapper/imrsim",
                   help="IMRSim device-mapper path")
    p.add_argument("--dry-run", action="store_true",
                   help="Use mock dataloader; no real device required")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_baseline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_root=args.data_root,
        dry_run=args.dry_run,
        n_classes=args.classes,
        m_images=args.images_per_class,
        imrsim_util=args.imrsim_util,
        device=args.device,
        verbose=not args.quiet,
    )
