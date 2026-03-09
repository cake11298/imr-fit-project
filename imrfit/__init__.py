"""
IMR-Fit: Intelligent block placement for Interlaced Magnetic Recording drives.

This package intercepts PyTorch DataLoader I/O, computes per-block placement
scores, and migrates data between IMR Top/Bottom tracks to minimize RMW overhead.
"""

from .profiler import DataLoaderProfiler, BlockStats
from .scorer import Scorer, BlockPlacement
from .scheduler import MigrationScheduler, MigrationPlan
from .monitor import IMRSimMonitor

__version__ = "0.1.0"
__all__ = [
    "DataLoaderProfiler",
    "BlockStats",
    "Scorer",
    "BlockPlacement",
    "MigrationScheduler",
    "MigrationPlan",
    "IMRSimMonitor",
]
