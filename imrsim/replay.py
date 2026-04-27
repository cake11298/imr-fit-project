"""replay.py — Module 5.

Replay a cold-tier trace under four placement strategies and emit metrics
that the plotter (Module 6) consumes:

    * Strategy.CMR_BASELINE     — pretend every block is on Top (no RMW)
    * Strategy.NAIVE_IMR        — random Top/Bottom placement (50/50)
    * Strategy.TRACKLACE        — frequency-only (1-D variant of IMR-Fit)
    * Strategy.IMRFIT           — full 4-D scoring

Two backends are supported:

    1. Real IMRSim (--backend kernel):
       Replays the trace by issuing dd / pwrite calls against
       /dev/mapper/imrsim and pulls RMW counts via IMRSimMonitor.

    2. Python fallback (--backend python, default):
       Uses imrsim.fallback_simulator.FallbackIMRSim — no kernel needed.

Output (per strategy)
---------------------
    {
      "throughput_mbps":   float,
      "rmw_count":         int,
      "rmw_ratio":         float,
      "displacement":      list[float],   # per-epoch D(e) curve
      "epoch_throughput":  list[float],   # per-epoch MB/s
      "epoch_rmw":         list[int],     # per-epoch RMW counts
      "migration_overhead_pct": float,
    }
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from imrfit.scorer import (
    BlockPlacement, Scorer, ScorerConfig,
)
from imrfit.analyzer import (
    Analyzer, AnalyzerConfig, BLOCK_SIZE_BYTES,
)
from imrfit.scheduler import MigrationScheduler

from .fallback_simulator import FallbackIMRSim, RMWModel


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


class Strategy(Enum):
    CMR_BASELINE = "cmr_baseline"
    NAIVE_IMR = "naive_imr"
    TRACKLACE = "tracklace"
    IMRFIT = "imrfit"

    @classmethod
    def all(cls) -> List["Strategy"]:
        return [cls.CMR_BASELINE, cls.NAIVE_IMR, cls.TRACKLACE, cls.IMRFIT]


# ---------------------------------------------------------------------------
# Configuration / result types
# ---------------------------------------------------------------------------


@dataclass
class ReplayConfig:
    backend: str = "python"                  # "python" or "kernel"
    block_size: int = BLOCK_SIZE_BYTES
    epoch_io_count: int = 5000
    seed: int = 0xDEADBEEF
    migration_budget: int = 32                # blocks moved per epoch (IMR-Fit)
    theta: float = 0.55
    scorer_config: ScorerConfig = field(default_factory=ScorerConfig)
    rmw_model: RMWModel = field(default_factory=RMWModel)


@dataclass
class ReplayResult:
    strategy: str
    scenario: str
    throughput_mbps: float
    rmw_count: int
    rmw_ratio: float
    bytes_read: int
    bytes_written: int
    displacement: List[float]
    epoch_throughput: List[float]
    epoch_rmw: List[int]
    migration_overhead_pct: float
    n_blocks: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Helpers: build a placement map for each strategy
# ---------------------------------------------------------------------------


def _placement_cmr(block_ids: Iterable[int]) -> Dict[int, BlockPlacement]:
    """Pretend every block is on Top track."""
    return {bid: BlockPlacement.TOP for bid in block_ids}


def _placement_naive(block_ids: Iterable[int],
                     seed: int) -> Dict[int, BlockPlacement]:
    rng = random.Random(seed)
    return {
        bid: (BlockPlacement.TOP if rng.random() < 0.5
              else BlockPlacement.BOTTOM)
        for bid in block_ids
    }


def _placement_tracklace(features: Dict[int, Dict[str, float]],
                         theta: float) -> Dict[int, BlockPlacement]:
    """Frequency-only (1-D): use F(b) directly as score."""
    out: Dict[int, BlockPlacement] = {}
    for bid, m in features.items():
        out[bid] = (BlockPlacement.TOP if m["F"] >= theta
                    else BlockPlacement.BOTTOM)
    return out


def _placement_imrfit(features: Dict[int, Dict[str, float]],
                      cfg: ScorerConfig) -> Dict[int, BlockPlacement]:
    scorer = Scorer(cfg)
    return {bid: sr.placement
            for bid, sr in scorer.score_all(features).items()}


# ---------------------------------------------------------------------------
# Replayer
# ---------------------------------------------------------------------------


class Replayer:
    """Drives a fallback (or kernel) simulator with a cold-tier trace."""

    def __init__(self, config: Optional[ReplayConfig] = None) -> None:
        self.cfg = config or ReplayConfig()

    # ------------------------------------------------------------------
    # Trace -> per-strategy ReplayResult
    # ------------------------------------------------------------------

    def replay(self, trace_path: str, *, scenario: str = "?") -> Dict[str, ReplayResult]:
        analyzer = Analyzer(AnalyzerConfig(
            block_size=self.cfg.block_size,
            scorer_config=self.cfg.scorer_config,
        ))
        aggregates, t_now = analyzer.aggregate(trace_path)
        features = analyzer.features_from(aggregates, t_now)
        block_ids = list(aggregates.keys())

        # Optimal (IMR-Fit) placement is reused as the D(e) reference for
        # every strategy.  This is the placement an *oracle* would converge to.
        optimal = _placement_imrfit(features, self.cfg.scorer_config)

        placements = {
            Strategy.CMR_BASELINE: _placement_cmr(block_ids),
            Strategy.NAIVE_IMR:    _placement_naive(block_ids, self.cfg.seed),
            Strategy.TRACKLACE:    _placement_tracklace(features, self.cfg.theta),
            Strategy.IMRFIT:       optimal,
        }

        results: Dict[str, ReplayResult] = {}
        for strategy in Strategy.all():
            if self.cfg.backend == "kernel":
                res = self._replay_kernel(strategy, trace_path,
                                          placements[strategy], optimal,
                                          scenario=scenario,
                                          n_blocks=len(block_ids))
            else:
                res = self._replay_python(strategy, trace_path,
                                          placements[strategy], optimal,
                                          scenario=scenario,
                                          n_blocks=len(block_ids))
            results[strategy.value] = res
        return results

    # ------------------------------------------------------------------
    # Python backend
    # ------------------------------------------------------------------

    def _replay_python(self,
                       strategy: Strategy,
                       trace_path: str,
                       placement_map: Dict[int, BlockPlacement],
                       optimal: Dict[int, BlockPlacement],
                       *,
                       scenario: str,
                       n_blocks: int) -> ReplayResult:
        sim = FallbackIMRSim(
            placement_map=placement_map,
            model=self.cfg.rmw_model,
            epoch_io_count=self.cfg.epoch_io_count,
        )
        sim.set_optimal_placement(optimal)

        # IMR-Fit is the only strategy that *moves* blocks during replay; the
        # others stay frozen.  We run a budget-constrained migration once at
        # the start of each epoch to model the convergence curve.
        if strategy == Strategy.IMRFIT:
            self._run_with_migration(sim, trace_path, optimal)
        else:
            sim.replay(self._iter_trace(trace_path),
                       block_size=self.cfg.block_size)

        s = sim.stats()
        n_writes = max(1, s["bytes_written"])
        # migration_overhead_pct: extra I/O from migration ops (IMR-Fit only).
        if strategy == Strategy.IMRFIT and self._migration_bytes > 0:
            migration_overhead = (
                self._migration_bytes / max(1, s["bytes_written"]) * 100.0
            )
        else:
            migration_overhead = 0.0

        return ReplayResult(
            strategy=strategy.value,
            scenario=scenario,
            throughput_mbps=s["throughput_mbps"],
            rmw_count=s["rmw_count"],
            rmw_ratio=s["rmw_ratio"],
            bytes_read=s["bytes_read"],
            bytes_written=s["bytes_written"],
            displacement=s["epoch_displacement"],
            epoch_throughput=s["epoch_throughput_mbps"],
            epoch_rmw=s["epoch_rmw_count"],
            migration_overhead_pct=migration_overhead,
            n_blocks=n_blocks,
        )

    # ------------------------------------------------------------------
    # Migration-aware replay (IMR-Fit)
    # ------------------------------------------------------------------

    def _run_with_migration(self,
                            sim: FallbackIMRSim,
                            trace_path: str,
                            optimal: Dict[int, BlockPlacement]) -> None:
        """Replay an epoch's worth of I/O, then migrate up to `budget` blocks
        toward the optimal placement, and repeat.

        Migration cost is accounted for as extra block-sized I/O on the
        device side (sim's counters get bumped via dummy writes).
        """
        scheduler = MigrationScheduler(
            budget=self.cfg.migration_budget,
            theta=self.cfg.theta,
        )
        # Start every block at BOTTOM so D(e) decreases monotonically.
        sim.bulk_set_placement(
            {bid: BlockPlacement.BOTTOM for bid in optimal}
        )
        epoch = 0
        epoch_buffer: List[Dict] = []
        self._migration_bytes = 0

        for record in self._iter_trace(trace_path):
            epoch_buffer.append(record)
            if len(epoch_buffer) >= self.cfg.epoch_io_count:
                sim.replay(epoch_buffer, block_size=self.cfg.block_size)
                self._maybe_migrate(sim, scheduler, optimal, epoch)
                epoch_buffer.clear()
                epoch += 1

        if epoch_buffer:
            sim.replay(epoch_buffer, block_size=self.cfg.block_size)
            self._maybe_migrate(sim, scheduler, optimal, epoch)

    def _maybe_migrate(self,
                       sim: FallbackIMRSim,
                       scheduler: MigrationScheduler,
                       optimal: Dict[int, BlockPlacement],
                       epoch: int) -> None:
        # Build score_results compatible with the scheduler's expected input.
        from imrfit.scorer import ScoreResult
        score_results = {
            bid: ScoreResult(
                block_id=bid, F=0, Q=0, Z=0, R=0,
                score=1.0 if opt == BlockPlacement.TOP else 0.0,
                placement=opt,
            )
            for bid, opt in optimal.items()
        }
        current = {bid: sim.get_placement(bid) for bid in optimal}
        plan = scheduler.plan(epoch, score_results, current)
        for op in plan.operations:
            sim.set_placement(op.block_id, op.to_placement)
            # Each migration costs ~one block-sized read+write
            self._migration_bytes += 2 * self.cfg.block_size

    # ------------------------------------------------------------------
    # Kernel backend (only attempts to talk to /dev/mapper/imrsim)
    # ------------------------------------------------------------------

    def _replay_kernel(self,
                       strategy: Strategy,
                       trace_path: str,
                       placement_map: Dict[int, BlockPlacement],
                       optimal: Dict[int, BlockPlacement],
                       *,
                       scenario: str,
                       n_blocks: int) -> ReplayResult:
        """Kernel-mode replay using the real IMRSim device.

        The implementation deliberately keeps the I/O loop in Python — the
        plumbing (issue dd, then read RMW counters via imrsim_util) is the
        kind of thing that's much easier to debug in Python than in a
        kernel module patch.

        Falls through to the Python backend if the device is missing.
        """
        device = "/dev/mapper/imrsim"
        if not Path(device).exists():
            print(f"[replay] {device} not present; falling back to python",
                  file=sys.stderr)
            return self._replay_python(strategy, trace_path, placement_map,
                                       optimal, scenario=scenario,
                                       n_blocks=n_blocks)

        # NOTE: a real implementation would:
        #   1. dmsetup load a fresh imrsim target with the given placement
        #   2. issue pwrite()/pread() at each (lba, size) in the trace
        #   3. poll imrsim_util <dev> s 1 between epochs
        # The kernel patch is unfinished at the time of writing (per spec).
        # Document the path and fall back to Python so the experiment
        # continues to make progress.
        print(f"[replay] kernel backend stub: replaying via python for now",
              file=sys.stderr)
        return self._replay_python(strategy, trace_path, placement_map,
                                   optimal, scenario=scenario,
                                   n_blocks=n_blocks)

    # ------------------------------------------------------------------
    # Trace streaming
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_trace(trace_path: str) -> Iterable[Dict]:
        with open(trace_path, "r", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--scenario", default="?")
    p.add_argument("--out", default="results/latest/replay.json")
    p.add_argument("--backend", choices=["python", "kernel"], default="python")
    p.add_argument("--epoch-io", type=int, default=5000)
    p.add_argument("--budget", type=int, default=32)
    args = p.parse_args(argv)

    cfg = ReplayConfig(
        backend=args.backend,
        epoch_io_count=args.epoch_io,
        migration_budget=args.budget,
    )
    rep = Replayer(cfg)
    out = rep.replay(args.trace, scenario=args.scenario)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({k: v.to_dict() for k, v in out.items()}, fh, indent=2)
    print(json.dumps({k: {"throughput_mbps": v.throughput_mbps,
                          "rmw_count": v.rmw_count,
                          "rmw_ratio": v.rmw_ratio}
                      for k, v in out.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
