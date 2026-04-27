"""fallback_simulator.py — pure-Python IMR behaviour model.

Used when the IMRSim kernel module is unavailable (kernel oops, container
without dm-mod, CI machine, etc.).

Model
-----
The drive is divided into 128 MB *blocks*.  Each block is either on a Top
track or on a Bottom track:

    * Top track    — read/write at full HDD bandwidth.  No penalty.
    * Bottom track — write requires a Read-Modify-Write of the adjacent
                     Top track first (RMW).  Penalty: one extra read +
                     one extra write of the same block size.

A read of any block costs the same regardless of placement.

Throughput is approximated using a simple bandwidth budget (MB/s) tunable
via `RMWModel.hdd_mbps`.

This is deliberately a coarse model: the goal is *relative* comparison
between strategies, not absolute throughput.  All four strategies are
charged with the same physical model, so comparisons are fair.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from imrfit.scorer import BlockPlacement


# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------


@dataclass
class RMWModel:
    """Tunable parameters of the fallback IMR behaviour model.

    Defaults reflect a mid-range consumer SMR/IMR HDD (Seagate Exos / WD Red).
    """
    hdd_mbps: float = 180.0          # average sustained sequential bandwidth
    seek_penalty_ms: float = 4.0     # added per non-sequential access
    rmw_extra_io_factor: float = 2.0 # each bottom-track write incurs 2x I/O

    def write_cost_seconds(self, size_bytes: int,
                           placement: BlockPlacement) -> float:
        """Wall-clock cost of one write.  Includes RMW penalty if BOTTOM."""
        base = size_bytes / (self.hdd_mbps * 1024 * 1024)
        if placement == BlockPlacement.BOTTOM:
            base *= self.rmw_extra_io_factor
        return base + self.seek_penalty_ms / 1000.0

    def read_cost_seconds(self, size_bytes: int) -> float:
        return size_bytes / (self.hdd_mbps * 1024 * 1024) \
            + self.seek_penalty_ms / 1000.0


# ---------------------------------------------------------------------------
# Fallback simulator
# ---------------------------------------------------------------------------


@dataclass
class _ReplayCounters:
    bytes_read: int = 0
    bytes_written: int = 0
    rmw_count: int = 0
    rmw_bytes: int = 0
    elapsed_sec: float = 0.0

    # Per-epoch slices, used for D(e) and throughput-vs-epoch plots.
    epoch_throughput_mbps: List[float] = field(default_factory=list)
    epoch_rmw_count: List[int] = field(default_factory=list)
    epoch_displacement: List[float] = field(default_factory=list)


class FallbackIMRSim:
    """Software-only IMR behaviour reproducing the kernel module's accounting.

    Args:
        placement_map: dict block_id -> BlockPlacement.  Blocks not present
                       are assumed BOTTOM (the conservative default).
        model:         physical model (RMWModel).
        epoch_io_count: how many trace records make up one "epoch" (used
                       for D(e) convergence plots; default = 5000).
    """

    def __init__(
        self,
        placement_map: Optional[Dict[int, BlockPlacement]] = None,
        *,
        model: Optional[RMWModel] = None,
        epoch_io_count: int = 5000,
    ) -> None:
        self._placement: Dict[int, BlockPlacement] = dict(placement_map or {})
        self._model = model or RMWModel()
        self._counters = _ReplayCounters()
        self._epoch_io_count = max(1, epoch_io_count)

        # Per-block scratch for write counts (used by D(e) calculation).
        self._block_writes: Dict[int, int] = defaultdict(int)
        self._optimal_placement: Dict[int, BlockPlacement] = {}

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_placement(self, block_id: int, placement: BlockPlacement) -> None:
        self._placement[block_id] = placement

    def bulk_set_placement(self, mapping: Dict[int, BlockPlacement]) -> None:
        self._placement.update(mapping)

    def set_optimal_placement(self,
                              mapping: Dict[int, BlockPlacement]) -> None:
        """Used to compute D(e) per epoch."""
        self._optimal_placement = dict(mapping)

    def get_placement(self, block_id: int) -> BlockPlacement:
        return self._placement.get(block_id, BlockPlacement.BOTTOM)

    # ------------------------------------------------------------------
    # Trace replay
    # ------------------------------------------------------------------

    def replay(self, trace: Iterable[Dict],
               *, block_size: int = 128 * 1024 * 1024) -> _ReplayCounters:
        """Replay a sequence of trace records (each = one I/O)."""
        bytes_read = 0
        bytes_written = 0
        rmw_count = 0
        rmw_bytes = 0
        elapsed = 0.0

        epoch_io = 0
        epoch_bytes = 0
        epoch_seconds = 0.0
        epoch_rmw = 0

        for rec in trace:
            size = int(rec["size"])
            op = rec["op"]
            block_id = int(rec["lba"]) // block_size
            placement = self.get_placement(block_id)

            if op == "R":
                bytes_read += size
                cost = self._model.read_cost_seconds(size)
                epoch_bytes += size
            else:  # write
                bytes_written += size
                cost = self._model.write_cost_seconds(size, placement)
                epoch_bytes += size
                self._block_writes[block_id] += 1
                if placement == BlockPlacement.BOTTOM:
                    rmw_count += 1
                    rmw_bytes += size
                    epoch_rmw += 1

            elapsed += cost
            epoch_seconds += cost
            epoch_io += 1

            if epoch_io >= self._epoch_io_count:
                self._record_epoch(
                    epoch_bytes=epoch_bytes,
                    epoch_seconds=epoch_seconds,
                    epoch_rmw=epoch_rmw,
                )
                epoch_io = 0
                epoch_bytes = 0
                epoch_seconds = 0.0
                epoch_rmw = 0

        if epoch_io > 0:
            self._record_epoch(
                epoch_bytes=epoch_bytes,
                epoch_seconds=epoch_seconds,
                epoch_rmw=epoch_rmw,
            )

        c = self._counters
        c.bytes_read += bytes_read
        c.bytes_written += bytes_written
        c.rmw_count += rmw_count
        c.rmw_bytes += rmw_bytes
        c.elapsed_sec += elapsed
        return c

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        c = self._counters
        total_bytes = c.bytes_read + c.bytes_written
        thr = (total_bytes / (1024 * 1024)) / c.elapsed_sec if c.elapsed_sec > 0 else 0.0
        rmw_ratio = c.rmw_count / max(1, c.bytes_written and 1)
        # rmw_ratio above is meaningless without a denominator; recompute
        # properly using write count instead of bytes.
        write_ops = sum(self._block_writes.values()) or 1
        rmw_ratio = c.rmw_count / write_ops
        return {
            "bytes_read": c.bytes_read,
            "bytes_written": c.bytes_written,
            "rmw_count": c.rmw_count,
            "rmw_bytes": c.rmw_bytes,
            "rmw_ratio": rmw_ratio,
            "elapsed_sec": c.elapsed_sec,
            "throughput_mbps": thr,
            "epoch_throughput_mbps": list(c.epoch_throughput_mbps),
            "epoch_rmw_count": list(c.epoch_rmw_count),
            "epoch_displacement": list(c.epoch_displacement),
            "n_epochs": len(c.epoch_throughput_mbps),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_epoch(self, *, epoch_bytes: int,
                      epoch_seconds: float,
                      epoch_rmw: int) -> None:
        thr = ((epoch_bytes / (1024 * 1024)) / epoch_seconds
               if epoch_seconds > 0 else 0.0)
        self._counters.epoch_throughput_mbps.append(thr)
        self._counters.epoch_rmw_count.append(epoch_rmw)
        self._counters.epoch_displacement.append(self._displacement())

    def _displacement(self) -> float:
        if not self._optimal_placement:
            return 0.0
        misplaced = sum(
            1 for bid, opt in self._optimal_placement.items()
            if self._placement.get(bid, BlockPlacement.BOTTOM) != opt
        )
        return misplaced / max(1, len(self._optimal_placement))
