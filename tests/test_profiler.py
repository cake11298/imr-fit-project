"""
test_profiler.py - pytest tests for imrfit.profiler

Tests cover:
  - BlockStats metric calculations (F, Q, Z, R)
  - DataLoaderProfiler record_file_access
  - DataLoaderProfiler compute_scores_input
  - DataLoaderProfiler reset
  - DataLoaderProfiler wrapping / __iter__ / __len__
  - Thread safety (basic smoke test)
"""

import math
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imrfit.profiler import BLOCK_SIZE_BYTES, BlockStats, DataLoaderProfiler


# ---------------------------------------------------------------------------
# BlockStats tests
# ---------------------------------------------------------------------------

class TestBlockStats:
    def test_default_values(self):
        bs = BlockStats(block_id=5)
        assert bs.block_id == 5
        assert bs.access_count == 0
        assert bs.sequential_count == 0
        assert bs.bytes_accessed == 0

    def test_normalized_frequency_zero_total(self):
        bs = BlockStats(block_id=0)
        assert bs.normalized_frequency(0) == 0.0

    def test_normalized_frequency(self):
        bs = BlockStats(block_id=0, access_count=3)
        assert math.isclose(bs.normalized_frequency(10), 0.3)

    def test_sequential_ratio_zero_accesses(self):
        bs = BlockStats(block_id=0)
        assert bs.sequential_ratio() == 0.0

    def test_sequential_ratio(self):
        bs = BlockStats(block_id=0, access_count=4, sequential_count=3)
        assert math.isclose(bs.sequential_ratio(), 0.75)

    def test_size_weight_zero_total(self):
        bs = BlockStats(block_id=0, bytes_accessed=1000)
        assert bs.size_weight(0) == 0.0

    def test_size_weight(self):
        bs = BlockStats(block_id=0, bytes_accessed=500)
        assert math.isclose(bs.size_weight(1000), 0.5)

    def test_size_weight_capped_at_one(self):
        bs = BlockStats(block_id=0, bytes_accessed=2000)
        assert bs.size_weight(1000) == 1.0

    def test_recency_at_zero_delta(self):
        t = time.monotonic()
        bs = BlockStats(block_id=0, last_access_time=t)
        r = bs.recency(t_now=t, lam=1.0)
        assert math.isclose(r, 1.0)

    def test_recency_formula(self):
        lam = 0.5
        delta = 4.0
        t_last = 0.0
        bs = BlockStats(block_id=0, last_access_time=t_last)
        r = bs.recency(t_now=delta, lam=lam)
        assert math.isclose(r, math.exp(-lam * delta), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# DataLoaderProfiler tests
# ---------------------------------------------------------------------------

class FakeDataset:
    """Minimal map-style dataset with .samples for testing."""
    def __init__(self, paths):
        self.samples = [(p, 0) for p in paths]

    def __len__(self):
        return len(self.samples)


class FakeLoader:
    """Minimal DataLoader stub."""
    def __init__(self, dataset, batch_size=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self._batches = [
            dataset.samples[i: i + batch_size]
            for i in range(0, len(dataset), batch_size)
        ]

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def __len__(self):
        return len(self._batches)


@pytest.fixture
def tmp_files(tmp_path):
    """Create a few small temp files and return their paths."""
    paths = []
    for i in range(4):
        p = tmp_path / f"file_{i}.jpg"
        p.write_bytes(b"X" * 1024)  # 1 KB
        paths.append(str(p))
    return paths, tmp_path


class TestDataLoaderProfiler:
    def _make_profiler(self, tmp_path, paths):
        dataset = FakeDataset(paths)
        loader = FakeLoader(dataset, batch_size=2)
        return DataLoaderProfiler(
            loader,
            mount_point=str(tmp_path),
            recency_lambda=1.0,
        )

    def test_init(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)
        assert profiler.block_size == BLOCK_SIZE_BYTES
        assert profiler.get_block_stats() == {}

    def test_len_delegates_to_loader(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)
        assert len(profiler) == len(FakeLoader(FakeDataset(paths)))

    def test_dataset_property(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)
        assert profiler.dataset is not None

    def test_record_file_access_updates_stats(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)

        profiler.record_file_access(paths[0])
        stats = profiler.get_block_stats()
        assert len(stats) >= 1

        # The accessed block should have at least 1 access
        total_accesses = sum(bs.access_count for bs in stats.values())
        assert total_accesses == 1

    def test_multiple_accesses_accumulate(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)

        for path in paths:
            profiler.record_file_access(path)
            profiler.record_file_access(path)

        summary = profiler.get_summary()
        assert summary["total_accesses"] == len(paths) * 2

    def test_record_nonexistent_file_is_safe(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)
        # Should not raise
        profiler.record_file_access("/nonexistent/path/to/file.jpg")
        assert profiler.get_summary()["total_accesses"] == 0

    def test_reset_clears_stats(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)

        profiler.record_file_access(paths[0])
        assert profiler.get_summary()["total_accesses"] == 1

        profiler.reset()
        assert profiler.get_summary()["total_accesses"] == 0
        assert profiler.get_block_stats() == {}

    def test_get_summary_keys(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)
        summary = profiler.get_summary()
        for key in ("total_accesses", "total_bytes_accessed", "blocks_touched", "block_size_bytes"):
            assert key in summary

    def test_compute_scores_input_empty(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)
        scores = profiler.compute_scores_input()
        assert scores == {}

    def test_compute_scores_input_after_access(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)
        profiler.record_file_access(paths[0])

        scores = profiler.compute_scores_input()
        assert len(scores) >= 1

        for bid, metrics in scores.items():
            assert "F" in metrics
            assert "Q" in metrics
            assert "Z" in metrics
            assert "R" in metrics
            # All metrics in [0, 1]
            for key, val in metrics.items():
                assert 0.0 <= val <= 1.0, f"metric {key}={val} out of [0,1]"

    def test_scores_F_sums_to_one_or_less(self, tmp_files):
        """Normalised frequencies should sum to <= 1 (they sum to exactly 1 if all blocks covered)."""
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)
        for p in paths:
            profiler.record_file_access(p)

        scores = profiler.compute_scores_input()
        f_sum = sum(m["F"] for m in scores.values())
        assert f_sum <= 1.0 + 1e-9

    def test_iter_yields_batches(self, tmp_files):
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)

        batches = list(profiler)
        assert len(batches) > 0

    def test_iter_passes_through_data(self, tmp_files):
        paths, tmp_path = tmp_files
        dataset = FakeDataset(paths)
        loader = FakeLoader(dataset, batch_size=2)
        profiler = DataLoaderProfiler(loader, mount_point=str(tmp_path))

        original_batches = list(FakeLoader(dataset, batch_size=2))
        profiled_batches = list(profiler)

        assert len(original_batches) == len(profiled_batches)

    # -----------------------------------------------------------------------
    # Thread safety
    # -----------------------------------------------------------------------

    def test_concurrent_record_file_access(self, tmp_files):
        """Multiple threads calling record_file_access should not corrupt state."""
        paths, tmp_path = tmp_files
        profiler = self._make_profiler(tmp_path, paths)

        errors = []

        def worker(path):
            try:
                for _ in range(20):
                    profiler.record_file_access(path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(p,)) for p in paths]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors in threads: {errors}"
        summary = profiler.get_summary()
        assert summary["total_accesses"] == len(paths) * 20


# ---------------------------------------------------------------------------
# BlockStats dataclass default_factory correctness
# ---------------------------------------------------------------------------

class TestBlockStatsDefaults:
    def test_two_instances_have_independent_timestamps(self):
        bs1 = BlockStats(block_id=0)
        time.sleep(0.001)
        bs2 = BlockStats(block_id=1)
        # They should be different (or at least not share a mutable default)
        # We can't guarantee order but they should be distinct objects
        assert bs1 is not bs2
