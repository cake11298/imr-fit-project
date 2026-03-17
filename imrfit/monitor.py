"""
monitor.py - IMRSim RMW statistics monitor.

Shells out to `imrsim_util` (the IMRSim user-space tool) to read per-zone
Read-Modify-Write (RMW) counts from a live /dev/mapper/imrsim device.

Real imrsim_util commands:
    imrsim_util <device> s 1   # get all zone stats
    imrsim_util <device> s 4   # reset all zone stats

Real imrsim_util output format (one line per zone):
    zone[0]: condition=0x1 type=0x1 ... read_count=100 write_count=50 rmw_count=10
    zone[1]: condition=0x1 type=0x2 ... read_count=200 write_count=80 rmw_count=25

When dry_run=True (or the device is unavailable), returns synthetic
statistics so the rest of the pipeline can run without a real device.
"""

from __future__ import annotations

import os
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
                     (default: ~/IMRSim/imrsim_util/imrsim_util)
        dry_run: if True, return synthetic data without calling the real tool
        synthetic_zones: number of zones to simulate in dry-run mode
    """

    IMRSIM_UTIL_DEFAULT = os.path.expanduser("~/IMRSim/imrsim_util/imrsim_util")

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
        Read current RMW statistics from IMRSim (command: s 1).

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

    def reset_stats(self) -> None:
        """
        Reset all IMRSim zone counters on the device (command: s 4).

        In dry_run mode this is a no-op (synthetic counters are already
        per-epoch and don't need explicit resetting).
        """
        if self.dry_run:
            return
        try:
            result = subprocess.run(
                [self.imrsim_util, self.device, "s", "4"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"imrsim_util reset failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
        except FileNotFoundError:
            raise RuntimeError(
                f"imrsim_util not found at '{self.imrsim_util}'. "
                "Install IMRSim or use dry_run=True."
            )

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
        Call `imrsim_util <device> s 1` and parse its output.

        Real output format (one line per zone):
            zone[0]: condition=0x1 type=0x1 ... read_count=100 write_count=50 rmw_count=10
        """
        try:
            result = subprocess.run(
                [self.imrsim_util, self.device, "s", "1"],
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
        """
        Parse imrsim_util `s 1` output into a DeviceStats object.

        Matches lines of the form:
            zone[<id>]: ... read_count=<n> write_count=<n> rmw_count=<n>
        Fields may appear in any order; unrecognised fields are ignored.
        """
        stats = DeviceStats(polled_at=time.monotonic())
        # Match the zone index from "zone[N]:" prefix
        zone_line = re.compile(r"zone\[(\d+)\]:", re.IGNORECASE)
        # Extract named counters anywhere on the same line
        read_pat  = re.compile(r"read_count=(\d+)",  re.IGNORECASE)
        write_pat = re.compile(r"write_count=(\d+)", re.IGNORECASE)
        rmw_pat   = re.compile(r"rmw_count=(\d+)",   re.IGNORECASE)

        for line in raw.splitlines():
            zm = zone_line.search(line)
            if not zm:
                continue
            zid = int(zm.group(1))

            rm = read_pat.search(line)
            wm = write_pat.search(line)
            rmwm = rmw_pat.search(line)

            zs = ZoneStats(
                zone_id=zid,
                read_count=int(rm.group(1))   if rm   else 0,
                write_count=int(wm.group(1))  if wm   else 0,
                rmw_count=int(rmwm.group(1))  if rmwm else 0,
            )
            stats.zones[zid] = zs
            stats.total_rmw    += zs.rmw_count
            stats.total_reads  += zs.read_count
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
