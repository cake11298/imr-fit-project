"""IMR-Fit: intelligent block placement for IMR drives.

The original training-loop profiler (``DataLoaderProfiler``) is preserved
for backwards compatibility; the new RAG experiment uses
``imrfit.analyzer.Analyzer`` which consumes pre-recorded cold-tier traces.
"""

from .profiler import DataLoaderProfiler, BlockStats
from .scorer import Scorer, ScorerConfig, BlockPlacement, ScoreResult
from .scheduler import MigrationScheduler, MigrationPlan, MigrationOp
from .monitor import IMRSimMonitor, DeviceStats, ZoneStats
from .analyzer import (
    Analyzer,
    AnalyzerConfig,
    BlockFeatureRow,
    grid_search_weights,
    BLOCK_SIZE_BYTES,
    RECENCY_LAMBDA,
)

__version__ = "0.2.0"
__all__ = [
    # legacy training-loop profiler
    "DataLoaderProfiler", "BlockStats",
    # core scoring / scheduling primitives (shared by old & new pipelines)
    "Scorer", "ScorerConfig", "BlockPlacement", "ScoreResult",
    "MigrationScheduler", "MigrationPlan", "MigrationOp",
    "IMRSimMonitor", "DeviceStats", "ZoneStats",
    # new trace-based analyzer (Module 4)
    "Analyzer", "AnalyzerConfig", "BlockFeatureRow",
    "grid_search_weights",
    "BLOCK_SIZE_BYTES", "RECENCY_LAMBDA",
]
