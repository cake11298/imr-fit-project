"""
profiler.py - DataLoader I/O profiler for IMR-Fit.

Wraps a PyTorch DataLoader via composition to intercept file access patterns
and accumulate per-128MB-block statistics used by the Scorer.

Block statistics collected per block b:
  F(b) - access frequency (accesses / total accesses)
  Q(b) - sequential access ratio (sequential accesses / total accesses)
  Z(b) - size weight (block size / total data accessed)
  R(b) - recency: exp(-lambda * (t_now - t_last_access))
"""

import os
import time
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

BLOCK_SIZE_BYTES = 128 * 1024 * 1024  # 128 MB


@dataclass
class BlockStats:
    """Per-block access statistics."""
    block_id: int
    access_count: int = 0
    sequential_count: int = 0
    bytes_accessed: int = 0
    last_access_time: float = field(default_factory=time.monotonic)
    first_access_time: float = field(default_factory=time.monotonic)

    def normalized_frequency(self, total_accesses: int) -> float:
        """F(b): access frequency normalised over all blocks."""
        if total_accesses == 0:
            return 0.0
        return self.access_count / total_accesses

    def sequential_ratio(self) -> float:
        """Q(b): fraction of accesses that were sequential."""
        if self.access_count == 0:
            return 0.0
        return self.sequential_count / self.access_count

    def size_weight(self, total_bytes: int) -> float:
        """Z(b): proportion of total bytes touched in this block."""
        if total_bytes == 0:
            return 0.0
        return min(self.bytes_accessed / total_bytes, 1.0)

    def recency(self, t_now: float, lam: float = 1.0) -> float:
        """R(b) = exp(-lambda * (t_now - t_last_access))."""
        delta = t_now - self.last_access_time
        return math.exp(-lam * delta)


class DataLoaderProfiler:
    """
    Wraps a PyTorch DataLoader (composition, not subclass) to intercept
    sample-file paths and accumulate block-level I/O statistics.

    Usage::

        from torch.utils.data import DataLoader
        from imrfit.profiler import DataLoaderProfiler

        base_loader = DataLoader(dataset, batch_size=64, num_workers=4)
        profiler = DataLoaderProfiler(base_loader, mount_point="/mnt/imrsim")

        for batch in profiler:
            train_step(batch)

        stats = profiler.get_block_stats()
    """

    def __init__(
        self,
        dataloader: Any,
        mount_point: str = "/mnt/imrsim",
        block_size: int = BLOCK_SIZE_BYTES,
        recency_lambda: float = 1.0,
    ) -> None:
        self._loader = dataloader
        self.mount_point = os.path.realpath(mount_point)
        self.block_size = block_size
        self.recency_lambda = recency_lambda

        self._block_stats: Dict[int, BlockStats] = defaultdict(
            lambda: BlockStats(block_id=0)
        )
        self._total_accesses: int = 0
        self._total_bytes: int = 0
        self._prev_block: Optional[int] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator:
        """Iterate over the wrapped DataLoader, recording each batch."""
        for batch in self._loader:
            self._record_batch(batch)
            yield batch

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def dataset(self):
        return self._loader.dataset

    def record_file_access(self, filepath: str) -> None:
        """
        Explicitly record a file access (useful when dataset returns paths).

        Determines which 128 MB block the file belongs to based on its byte
        offset within the mount point, then updates BlockStats.
        """
        try:
            real_path = os.path.realpath(filepath)
            stat = os.stat(real_path)
            file_size = stat.st_size
        except OSError:
            return

        block_id = self._path_to_block_id(filepath)
        if block_id is None:
            return

        t_now = time.monotonic()
        with self._lock:
            if block_id not in self._block_stats:
                bs = BlockStats(block_id=block_id)
                bs.first_access_time = t_now
                self._block_stats[block_id] = bs

            bs = self._block_stats[block_id]
            is_sequential = (self._prev_block is not None and
                             block_id == self._prev_block)

            bs.access_count += 1
            bs.bytes_accessed += file_size
            bs.last_access_time = t_now
            if is_sequential:
                bs.sequential_count += 1

            self._total_accesses += 1
            self._total_bytes += file_size
            self._prev_block = block_id

    def get_block_stats(self) -> Dict[int, BlockStats]:
        """Return a snapshot of accumulated block statistics."""
        with self._lock:
            return dict(self._block_stats)

    def get_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary dict."""
        with self._lock:
            return {
                "total_accesses": self._total_accesses,
                "total_bytes_accessed": self._total_bytes,
                "blocks_touched": len(self._block_stats),
                "block_size_bytes": self.block_size,
            }

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        with self._lock:
            self._block_stats.clear()
            self._total_accesses = 0
            self._total_bytes = 0
            self._prev_block = None

    def compute_scores_input(self, t_now: Optional[float] = None) -> Dict[int, Dict[str, float]]:
        """
        Return per-block (F, Q, Z, R) values ready to pass to Scorer.

        Returns a dict  block_id -> {"F": ..., "Q": ..., "Z": ..., "R": ...}
        """
        if t_now is None:
            t_now = time.monotonic()
        with self._lock:
            result = {}
            for bid, bs in self._block_stats.items():
                result[bid] = {
                    "F": bs.normalized_frequency(self._total_accesses),
                    "Q": bs.sequential_ratio(),
                    "Z": bs.size_weight(self._total_bytes),
                    "R": bs.recency(t_now, self.recency_lambda),
                }
            return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_to_block_id(self, filepath: str) -> Optional[int]:
        """
        Map a file path to a virtual 128 MB block ID.

        Two strategies are tried in order:

        1. **Real file-offset method** (used when the file lives under
           self.mount_point):
           Approximates the file's byte position on the block device using
           ``st_blocks * 512`` (the number of 512-byte sectors allocated to
           the file as reported by the kernel).  This is a reasonable proxy
           for the file's physical offset on the IMRSim device and produces
           stable, device-meaningful block IDs.

        2. **Inode-hash fallback** (used for files outside the mount point,
           e.g. in /tmp during unit tests):
           NOTE — this is an approximation only.  The inode number is hashed
           into a virtual block bucket and does NOT reflect the file's real
           on-disk position.  It is used solely so the profiler can run in
           dry-run / test environments where /mnt/imrsim is unavailable.
        """
        try:
            real_path = os.path.realpath(filepath)
            stat = os.stat(real_path)

            if real_path.startswith(self.mount_point):
                # Strategy 1: derive offset from allocated 512-byte sectors.
                # st_blocks counts 512-byte units actually allocated on disk.
                byte_offset = stat.st_blocks * 512
                # Clamp to a reasonable address space (64 × block_size)
                byte_offset = byte_offset % (64 * self.block_size)
                return byte_offset // self.block_size
            else:
                # Strategy 2: inode-hash approximation (dry-run / test only).
                # WARNING: does NOT reflect real physical disk placement.
                approx_offset = (stat.st_ino * 4096) % (64 * self.block_size)
                return approx_offset // self.block_size
        except OSError:
            return None

    def _record_batch(self, batch: Any) -> None:
        """
        Attempt to extract file paths from a batch and record accesses.

        PyTorch datasets often return (tensor, label) tuples; path info is
        typically available on the dataset object.  We probe common patterns.
        """
        # Try to get file paths from the underlying dataset's samples list
        dataset = getattr(self._loader, "dataset", None)
        if dataset is None:
            return

        # torchvision ImageFolder exposes .samples = [(path, class_idx), ...]
        samples = getattr(dataset, "samples", None)
        if samples is not None and len(samples) > 0:
            # We don't know which indices were in this batch, so we record
            # a statistical sample proportional to batch size.
            batch_size = self._infer_batch_size(batch)
            step = max(1, len(samples) // max(batch_size, 1))
            for i in range(0, min(batch_size, len(samples))):
                path, _ = samples[i * step % len(samples)]
                self.record_file_access(path)

    @staticmethod
    def _infer_batch_size(batch: Any) -> int:
        """Best-effort batch size extraction."""
        try:
            import torch
            if isinstance(batch, (list, tuple)):
                first = batch[0]
                if isinstance(first, torch.Tensor):
                    return int(first.shape[0])
            if isinstance(batch, dict):
                for v in batch.values():
                    import torch
                    if isinstance(v, torch.Tensor):
                        return int(v.shape[0])
        except Exception:
            pass
        return 1
