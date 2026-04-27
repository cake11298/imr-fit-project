"""lru_cache.py — byte-budgeted LRU cache.

Standard `functools.lru_cache` counts entries, not bytes.  We need a byte
budget because the SSD cache size is a fraction of the on-disk corpus, and
text vs image entries differ by ~1000x.

RAM budget: bounded by `max_bytes`. Defaults to 0 (caller must size).
Concurrency: a `threading.Lock` protects the OrderedDict; this is enough for
            the experiment's ThreadPool-driven workload.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterator, Optional


@dataclass
class CacheEntry:
    """A single byte-sized cache entry."""
    key: Hashable
    value: Any
    size: int


class LRUCache:
    """Byte-budgeted LRU cache.

    Args:
        max_bytes: hard upper bound on the sum of `entry.size` over the cache.

    Methods are O(1) amortised.  The cache stores opaque values; the *caller*
    is responsible for telling us each entry's byte size on `put()`.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        self._max_bytes = int(max_bytes)
        self._store: "OrderedDict[Hashable, CacheEntry]" = OrderedDict()
        self._used_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, key: Hashable) -> Optional[Any]:
        """Return the value for `key`, or None.  Updates recency on hit."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._store.move_to_end(key)        # mark as most recent
            self._hits += 1
            return entry.value

    def contains(self, key: Hashable) -> bool:
        """Probe without touching recency."""
        with self._lock:
            return key in self._store

    def put(self, key: Hashable, value: Any, size: int) -> None:
        """Insert / replace an entry, evicting LRU items as needed."""
        if size < 0:
            raise ValueError("size must be >= 0")
        if size > self._max_bytes and self._max_bytes > 0:
            # Single object too big for the cache — silently skip.  Recording
            # the access counters still happens; we just don't hold the value.
            return
        with self._lock:
            existing = self._store.get(key)
            if existing is not None:
                self._used_bytes -= existing.size
                self._store.move_to_end(key)
                existing.value = value
                existing.size = size
                self._used_bytes += size
            else:
                self._store[key] = CacheEntry(key=key, value=value, size=size)
                self._used_bytes += size
            self._evict_if_needed_locked()

    def invalidate(self, key: Hashable) -> bool:
        """Remove `key`. Returns True if it was present."""
        with self._lock:
            entry = self._store.pop(key, None)
            if entry is None:
                return False
            self._used_bytes -= entry.size
            return True

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._used_bytes = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __iter__(self) -> Iterator[Hashable]:
        with self._lock:
            return iter(list(self._store.keys()))

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def evictions(self) -> int:
        return self._evictions

    @property
    def hit_ratio(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._store),
                "used_bytes": self._used_bytes,
                "max_bytes": self._max_bytes,
                "fill_pct": (100.0 * self._used_bytes / self._max_bytes
                             if self._max_bytes else 0.0),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_ratio": self.hit_ratio,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_if_needed_locked(self) -> None:
        while self._used_bytes > self._max_bytes and self._store:
            _, victim = self._store.popitem(last=False)   # FIFO from LRU end
            self._used_bytes -= victim.size
            self._evictions += 1
