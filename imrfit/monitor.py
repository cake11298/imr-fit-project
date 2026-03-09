"""
monitor.py - IMRSim RMW statistics monitor.

Shells out to `imrsim_util` (the IMRSim user-space tool) to read per-zone
Read-Modify-Write (RMW) counts from a live /dev/mapper/imrsim device.

When dry_run=True (or the device is unavailable), returns synthetic
statistics so the rest of the pipeline can run without a real device.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ZoneStats:
    """RMW statistics for one IMRSim zone."""
    zone_id: int
    rmw_count: int = 0
    read_count: int = 0
    write_count: int = 0
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def rmw_ratio(self) -> float:
        """Fraction of writes that triggered RMW."""
        if self.write_count == 0:
            return 0.0
        return self.rmw_count / self.write_count


@dataclass
class DeviceStats:
    """Aggregate IMRSim device statistics."""
    zones: Dict[int, ZoneStats] = field(default_factory=dict)
    total_rmw: int = 0
    total_reads: int = 0
    total_writes: int = 0
    polled_at: float = field(default_factory=time.monotonic)

    @property
    def overall_rmw_ratio(self) -> float:
        if self.total_writes == 0:
            return 0.0
        return self.total_rmw / self.total_writes


class IMRSimMonitor:
    """
    Polls IMRSim device statistics via imrsim_util.

    Args:
        device: path to the device-mapper target, e.g. /dev/mapper/imrsim
        imrsim_util: path to the imrsim_util binary
        dry_run: if True, return synthetic data without calling the real tool
        synthetic_zones: number of zones to simulate in dry-run mode
    """

    IMRSIM_UTIL_DEFAULT = "imrsim_util"

    def __init__(
        self,
        device: str = "/dev/mapper/imrsim",
        imrsim_util: str = IMRSIM_UTIL_DEFAULT,
        dry_run: bool = False,
        synthetic_zones: int = 8,
    ) -> None:
        self.device = device
        self.imrsim_util = imrsim_util
        self.dry_run = dry_run
        self.synthetic_zones = synthetic_zones
        self._history: List[DeviceStats] = []
        self._epoch_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poll(self) -> DeviceStats:
        """
        Read current RMW statistics from IMRSim.

        Returns a DeviceStats snapshot.  In dry_run mode, returns synthetic
        stats that grow monotonically to simulate a running workload.
        """
        if self.dry_run:
            stats = self._synthetic_stats()
        else:
            stats = self._real_stats()

        self._history.append(stats)
        self._epoch_counter += 1
        return stats

    def delta(self) -> Optional[DeviceStats]:
        """
        Return the *difference* between the two most recent polls.

        Useful to measure per-epoch RMW overhead.  Returns None if fewer
        than two polls have been taken.
        """
        if len(self._history) < 2:
            return None
        prev = self._history[-2]
        curr = self._history[-1]
        delta = DeviceStats(polled_at=curr.polled_at)
        delta.total_rmw = curr.total_rmw - prev.total_rmw
        delta.total_reads = curr.total_reads - prev.total_reads
        delta.total_writes = curr.total_writes - prev.total_writes
        for zid in curr.zones:
            cz = curr.zones[zid]
            pz = prev.zones.get(zid, ZoneStats(zone_id=zid))
            dz = ZoneStats(
                zone_id=zid,
                rmw_count=cz.rmw_count - pz.rmw_count,
                read_count=cz.read_count - pz.read_count,
                write_count=cz.write_count - pz.write_count,
                timestamp=cz.timestamp,
            )
            delta.zones[zid] = dz
        return delta

    def history(self) -> List[DeviceStats]:
        return list(self._history)

    def reset_history(self) -> None:
        self._history.clear()
        self._epoch_counter = 0

    # ------------------------------------------------------------------
    # Real device polling
    # ------------------------------------------------------------------

    def _real_stats(self) -> DeviceStats:
        """
        Call `imrsim_util <device> get_stats` and parse its output.

        Expected output format (one line per zone):
            zone <id>: rmw=<n> reads=<n> writes=<n>

        Adjust the regex below if the actual imrsim_util output differs.
        """
        try:
            result = subprocess.run(
                [self.imrsim_util, self.device, "get_stats"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"imrsim_util exited {result.returncode}: {result.stderr.strip()}"
                )
            return self._parse_output(result.stdout)
        except FileNotFoundError:
            raise RuntimeError(
                f"imrsim_util not found at '{self.imrsim_util}'. "
                "Install IMRSim or use dry_run=True."
            )

    @staticmethod
    def _parse_output(raw: str) -> DeviceStats:
        """Parse imrsim_util output into a DeviceStats object."""
        stats = DeviceStats(polled_at=time.monotonic())
        # Pattern: zone <id>: rmw=<n> reads=<n> writes=<n>
        pattern = re.compile(
            r"zone\s+(\d+):\s+rmw=(\d+)\s+reads=(\d+)\s+writes=(\d+)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(raw):
            zid = int(match.group(1))
            zs = ZoneStats(
                zone_id=zid,
                rmw_count=int(match.group(2)),
                read_count=int(match.group(3)),
                write_count=int(match.group(4)),
            )
            stats.zones[zid] = zs
            stats.total_rmw += zs.rmw_count
            stats.total_reads += zs.read_count
            stats.total_writes += zs.write_count
        return stats

    # ------------------------------------------------------------------
    # Dry-run / synthetic stats
    # ------------------------------------------------------------------

    def _synthetic_stats(self) -> DeviceStats:
        """
        Generate plausible synthetic RMW statistics.

        RMW counts grow at different rates per zone to simulate heterogeneous
        access patterns.  Baseline (no IMR-Fit) is set to ~30% RMW ratio.
        """
        import random
        stats = DeviceStats(polled_at=time.monotonic())
        base_writes_per_zone = 1000 * (self._epoch_counter + 1)
        for zid in range(self.synthetic_zones):
            # Zones with even IDs are "hot" (more sequential, fewer RMWs)
            rmw_ratio = 0.15 if zid % 2 == 0 else 0.35
            writes = base_writes_per_zone + random.randint(-50, 50)
            rmw = int(writes * rmw_ratio)
            reads = int(writes * 0.8)
            zs = ZoneStats(
                zone_id=zid,
                rmw_count=rmw,
                read_count=reads,
                write_count=writes,
            )
            stats.zones[zid] = zs
            stats.total_rmw += rmw
            stats.total_reads += reads
            stats.total_writes += writes
        return stats
