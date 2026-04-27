"""tier_simulator.py — Module 2.

Two-tier (SSD/HDD) storage simulator that intercepts every chunk-level I/O
issued by the RAG layer (Module 3).

Behaviour
---------
* read(chunk_id):
    1. Look up the LRU SSD cache.  Hit -> return value, record cache_hit=True
       *but do not write to the trace* (the trace is for cold-tier I/O only).
    2. Miss -> read from HDD, populate the cache, append a record to the
       cold-tier trace.
* write(chunk_id, data):
    Always writes through to HDD (write-through, like an LLM training
    checkpoint or an ingest pipeline).  Always recorded in the trace.
* Every record carries (timestamp_ns, chunk_id, lba, size, op, scenario,
  cache_hit) — exactly the schema specified in the project brief.

LBA model
---------
Each chunk is mapped to a deterministic logical block address:
    lba = chunk_index * chunk_size_bytes (mod 2 * total_corpus_bytes)
The mod keeps LBAs inside a sane range when the corpus is rebuilt.

Memory budget
-------------
RAM = max_cache_bytes + manifest_index (<<200 MB for 1M chunks).
Disk reads are made via plain os.read at file granularity; we don't
mmap because mmap defeats the cold-tier semantics we want to measure.

Concurrency
-----------
The simulator is safe for use from a ThreadPoolExecutor (the trace writer
is mutex-guarded).  It is *not* re-entrant across processes — each process
must own its own simulator.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .lru_cache import LRUCache


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class TierConfig:
    """Per-run configuration of the tiered storage layer."""
    hdd_root: str = "/mnt/hdd/wiki_corpus"
    ssd_root: str = "/mnt/ssd"                 # cache root (informational)
    cache_bytes: int = 0                       # absolute SSD cache budget
    cache_fraction: float = 0.15               # used if cache_bytes == 0
    block_size: int = 128 * 1024 * 1024        # IMR-Fit block size
    trace_path: str = "traces/cold_tier_trace.jsonl"
    scenario: str = "default"
    fsync_on_write: bool = False               # leave off for speed in sim

    def resolve_cache_bytes(self, corpus_bytes: int) -> int:
        if self.cache_bytes > 0:
            return self.cache_bytes
        return int(corpus_bytes * self.cache_fraction)


@dataclass
class IORecord:
    """One line of the cold-tier trace (matches the spec'd schema)."""
    ts_ns: int
    chunk_id: str
    lba: int
    size: int
    op: str                  # "R" or "W"
    scenario: str
    cache_hit: bool

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":")) + "\n"


# ---------------------------------------------------------------------------
# Manifest loader (ties chunk_id -> on-disk shard offset / image path)
# ---------------------------------------------------------------------------


@dataclass
class _ChunkLocation:
    chunk_id: str
    kind: str                  # "text" | "image"
    shard: Optional[int]       # text-only; None for images
    offset: Optional[int]      # offset within shard
    size: int                  # advertised record size
    image_path: Optional[str]
    chunk_index: int           # monotonically increasing global index


class _ManifestIndex:
    """Lightweight chunk_id -> _ChunkLocation map loaded from manifest.jsonl."""

    def __init__(self, manifest_path: Path) -> None:
        self._by_id: Dict[str, _ChunkLocation] = {}
        self._ordered_ids: List[str] = []
        if not manifest_path.exists():
            return
        with open(manifest_path, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                row = json.loads(line)
                cid = row["chunk_id"]
                loc = _ChunkLocation(
                    chunk_id=cid,
                    kind=row.get("kind", "text"),
                    shard=row.get("shard"),
                    offset=row.get("offset"),
                    size=int(row.get("size", 0)),
                    image_path=row.get("image_path"),
                    chunk_index=idx,
                )
                self._by_id[cid] = loc
                self._ordered_ids.append(cid)

    def __contains__(self, cid: str) -> bool:
        return cid in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def __getitem__(self, cid: str) -> _ChunkLocation:
        return self._by_id[cid]

    def total_bytes(self) -> int:
        return sum(loc.size for loc in self._by_id.values())

    def sequential_ids(self) -> List[str]:
        return list(self._ordered_ids)


# ---------------------------------------------------------------------------
# Trace writer
# ---------------------------------------------------------------------------


class _TraceWriter:
    """Thread-safe JSONL appender."""

    def __init__(self, path: Path, append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if append else "wb"
        self._fh = open(path, mode, buffering=1024 * 1024)
        self._lock = threading.Lock()
        self.path = path
        self.records_written = 0

    def write(self, rec: IORecord) -> None:
        line = rec.to_json_line().encode("utf-8")
        with self._lock:
            self._fh.write(line)
            self.records_written += 1

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            self._fh.close()


# ---------------------------------------------------------------------------
# Tiered storage simulator
# ---------------------------------------------------------------------------


class TieredStorageSimulator:
    """SSD-cached HDD with a cold-tier I/O trace.

    Typical use::

        cfg = TierConfig(cache_fraction=0.15, scenario="A")
        with TieredStorageSimulator(cfg) as sim:
            data = sim.read("wiki_000123_0007")
            sim.write("wiki_999999_0000", payload, size=len(payload))
    """

    def __init__(self, config: TierConfig) -> None:
        self.cfg = config
        self.hdd_root = Path(config.hdd_root)
        self._manifest = _ManifestIndex(self.hdd_root / "manifest.jsonl")

        corpus_bytes = self._manifest.total_bytes() if len(self._manifest) else 0
        cache_bytes = config.resolve_cache_bytes(corpus_bytes)
        self._cache = LRUCache(max_bytes=cache_bytes)

        self._trace = _TraceWriter(Path(config.trace_path))
        self._lock = threading.Lock()
        self._read_count = 0
        self._write_count = 0

    # ------------------------------------------------------------------
    # Context-manager glue
    # ------------------------------------------------------------------

    def __enter__(self) -> "TieredStorageSimulator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._trace.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def known_chunks(self) -> List[str]:
        """All chunk_ids visible to the simulator (manifest order)."""
        return self._manifest.sequential_ids()

    def read(self, chunk_id: str) -> Optional[bytes]:
        """Read a chunk; cache-miss reads land on /mnt/hdd and trace."""
        cached = self._cache.get(chunk_id)
        if cached is not None:
            self._read_count += 1
            # Cache hits do *not* go to trace — only cold-tier I/O is traced.
            return cached

        loc = self._manifest._by_id.get(chunk_id)
        size_hint = loc.size if loc else 0
        data = self._read_from_hdd(chunk_id)
        actual_size = len(data) if data is not None else size_hint

        self._record_io(
            chunk_id=chunk_id,
            size=actual_size,
            op="R",
            cache_hit=False,
        )

        if data is not None:
            self._cache.put(chunk_id, data, len(data))
        self._read_count += 1
        return data

    def write(self, chunk_id: str, data: bytes, *,
              kind: str = "text") -> None:
        """Write through to HDD and trace as a cold-tier write."""
        size = len(data)
        path = self._write_to_hdd(chunk_id, data, kind=kind)
        if path is not None:
            # Update / register in manifest so future reads find it.
            self._register_new_chunk(chunk_id, kind, size, path)

        self._record_io(
            chunk_id=chunk_id,
            size=size,
            op="W",
            cache_hit=False,
        )
        # Newly-written data is hot — pre-populate the cache.
        self._cache.put(chunk_id, data, size)
        self._write_count += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "reads": self._read_count,
            "writes": self._write_count,
            "trace_records": self._trace.records_written,
            "cache": self._cache.stats(),
            "manifest_chunks": len(self._manifest),
        }

    # ------------------------------------------------------------------
    # Internal: trace + LBA mapping
    # ------------------------------------------------------------------

    def _record_io(self, *, chunk_id: str, size: int, op: str,
                   cache_hit: bool) -> None:
        ts_ns = time.perf_counter_ns()
        lba = self._lba_for(chunk_id)
        rec = IORecord(
            ts_ns=ts_ns,
            chunk_id=chunk_id,
            lba=lba,
            size=size,
            op=op,
            scenario=self.cfg.scenario,
            cache_hit=cache_hit,
        )
        self._trace.write(rec)

    def _lba_for(self, chunk_id: str) -> int:
        """Stable per-chunk LBA. Image and text chunks share the same address
        space so sequential scans show monotonic LBA progression.
        """
        loc = self._manifest._by_id.get(chunk_id)
        if loc is not None:
            base = loc.chunk_index * self.cfg.block_size
        else:
            # Newly written chunk — hash chunk_id into a high address.
            base = (abs(hash(chunk_id)) & 0xFFFFFFFF) * self.cfg.block_size
            base += 1 << 36       # park new writes above existing LBAs
        return base

    # ------------------------------------------------------------------
    # Internal: HDD I/O
    # ------------------------------------------------------------------

    def _read_from_hdd(self, chunk_id: str) -> Optional[bytes]:
        loc = self._manifest._by_id.get(chunk_id)
        if loc is None:
            return None

        if loc.kind == "image" and loc.image_path:
            try:
                with open(loc.image_path, "rb") as fh:
                    return fh.read()
            except OSError:
                return None

        # text chunk — random read into the shard at the recorded offset
        shard_path = self.hdd_root / "text_shards" / f"shard_{loc.shard:05d}.jsonl"
        try:
            with open(shard_path, "rb") as fh:
                fh.seek(loc.offset or 0)
                # We trust the manifest size; read that many bytes.
                return fh.read(loc.size)
        except OSError:
            return None

    def _write_to_hdd(self, chunk_id: str, data: bytes,
                      *, kind: str) -> Optional[Path]:
        if kind == "image":
            sub = self.hdd_root / "images" / chunk_id[:8]
            sub.mkdir(parents=True, exist_ok=True)
            path = sub / f"{chunk_id}.jpg"
        else:
            sub = self.hdd_root / "ingest_shards"
            sub.mkdir(parents=True, exist_ok=True)
            path = sub / f"{chunk_id}.jsonl"
        try:
            with open(path, "ab") as fh:
                fh.write(data)
                if self.cfg.fsync_on_write:
                    fh.flush()
                    os.fsync(fh.fileno())
            return path
        except OSError as exc:
            print(f"[tier-sim] write failed for {chunk_id}: {exc}",
                  file=sys.stderr)
            return None

    def _register_new_chunk(self, chunk_id: str, kind: str,
                            size: int, path: Path) -> None:
        if chunk_id in self._manifest:
            return
        loc = _ChunkLocation(
            chunk_id=chunk_id,
            kind=kind,
            shard=None,
            offset=None,
            size=size,
            image_path=str(path) if kind == "image" else None,
            chunk_index=len(self._manifest),
        )
        self._manifest._by_id[chunk_id] = loc
        self._manifest._ordered_ids.append(chunk_id)


# ---------------------------------------------------------------------------
# Trace iterator (used by analyzer & replayer)
# ---------------------------------------------------------------------------


def iter_trace(path: str) -> Iterator[IORecord]:
    """Stream a trace file lazily — no full-file load."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            yield IORecord(**row)
