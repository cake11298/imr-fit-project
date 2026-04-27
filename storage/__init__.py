"""Storage tier simulator for the IMR-Fit experiment.

Two tiers:
    Hot  (SSD, /mnt/ssd): bounded LRU cache, ~ 15 % of corpus.
    Cold (HDD, /mnt/hdd): full corpus.  All cache misses surface here and
                          are recorded into a JSONL trace consumed by the
                          IMR-Fit analyzer (Module 4) and the IMRSim replay
                          engine (Module 5).
"""

from .lru_cache import LRUCache, CacheEntry
from .tier_simulator import (
    TieredStorageSimulator,
    TierConfig,
    IORecord,
)

__all__ = [
    "LRUCache",
    "CacheEntry",
    "TieredStorageSimulator",
    "TierConfig",
    "IORecord",
]
