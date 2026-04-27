"""scenarios.py — Module 3 (workload generator).

Three RAG workload scenarios that each produce an independent cold-tier
trace:

    A — Bursty Frequent
        50 hot Wikipedia topics queried in a Zipf-weighted rotation.
        Expected: high F(b), high R(b)  -> Top Track candidates.

    B — Cold Sequential Scan
        Iterate the entire corpus in manifest order, one read each.
        Expected: low F(b), high Q(b), high Z(b) -> Bottom Track.

    C — Mixed + Incremental Write
        70% random topic queries + 30% new-document ingestion.
        Expected: rich variance across all four dimensions; killer figure.

Each scenario is parameterised by total query count.  Default = 300 queries
per scenario as in the spec.
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .query_engine import RAGQueryEngine, EngineConfig
from storage.tier_simulator import TieredStorageSimulator, TierConfig


# ---------------------------------------------------------------------------
# Query lists (loaded from rag/queries/*.txt)
# ---------------------------------------------------------------------------


_QUERY_DIR = Path(__file__).parent / "queries"


def _load_query_file(name: str) -> List[str]:
    p = _QUERY_DIR / name
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


# ---------------------------------------------------------------------------
# Scenario base
# ---------------------------------------------------------------------------


@dataclass
class ScenarioReport:
    name: str
    queries_run: int = 0
    chunks_read: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    elapsed_sec: float = 0.0
    trace_path: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


class _BaseScenario:
    name: str = "base"
    label: str = "?"

    def __init__(self, n_queries: int = 300, seed: int = 0xCAFEBABE) -> None:
        self.n_queries = n_queries
        self.seed = seed

    def run(self, engine: RAGQueryEngine,
            sim: TieredStorageSimulator) -> ScenarioReport:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Scenario A — bursty frequent
# ---------------------------------------------------------------------------


class ScenarioA(_BaseScenario):
    name = "scenario_a"
    label = "A"

    HOT_QUERIES_FILE = "scenario_a_queries.txt"
    DEFAULT_TOPICS = [
        "Albert Einstein", "Quantum mechanics", "World War II",
        "Photosynthesis", "Renaissance", "DNA", "French Revolution",
        "Black hole", "Roman Empire", "Climate change",
        "Internet", "Mount Everest", "Jane Austen", "Solar system",
        "American Civil War", "Buddhism", "Linnaeus", "Mona Lisa",
        "Pyramids of Giza", "Industrial Revolution", "Beethoven",
        "Apollo 11", "Great Wall of China", "Vincent van Gogh",
        "Charles Darwin", "Eiffel Tower", "Pacific Ocean",
        "Mahatma Gandhi", "Cleopatra", "Statue of Liberty",
        "Galileo Galilei", "Mount Fuji", "Tokyo", "Nile",
        "Stonehenge", "Berlin Wall", "Marie Curie", "Sahara",
        "Marco Polo", "Niagara Falls", "Pythagoras", "Hubble Space Telescope",
        "Sigmund Freud", "Aurora", "Plato", "Vesuvius",
        "Frida Kahlo", "Bermuda Triangle", "World Wide Web", "ENIAC",
    ]

    def __init__(self, n_queries: int = 300, seed: int = 0xA1) -> None:
        super().__init__(n_queries=n_queries, seed=seed)

    def run(self, engine, sim) -> ScenarioReport:
        t0 = time.monotonic()
        rng = random.Random(self.seed)
        queries = _load_query_file(self.HOT_QUERIES_FILE) or self.DEFAULT_TOPICS
        # Zipf weights: heavy bias toward the first few topics.
        weights = [1.0 / (i + 1) for i in range(len(queries))]

        report = ScenarioReport(name=self.name)
        for _ in range(self.n_queries):
            q = rng.choices(queries, weights=weights, k=1)[0]
            res = engine.query(q)
            report.queries_run += 1
            report.chunks_read += len(res.retrieved_chunks)
            report.bytes_read += res.bytes_read
            report.cache_hits += res.cache_hits
            report.cache_misses += res.cache_misses

        report.elapsed_sec = time.monotonic() - t0
        report.trace_path = sim._trace.path.as_posix()
        return report


# ---------------------------------------------------------------------------
# Scenario B — cold sequential scan
# ---------------------------------------------------------------------------


class ScenarioB(_BaseScenario):
    name = "scenario_b"
    label = "B"

    def __init__(self, n_queries: int = 300, seed: int = 0xB2) -> None:
        super().__init__(n_queries=n_queries, seed=seed)

    def run(self, engine, sim) -> ScenarioReport:
        t0 = time.monotonic()
        report = ScenarioReport(name=self.name)
        all_ids = engine.known_chunk_ids() or sim.known_chunks()
        if not all_ids:
            print("[scenario-b] no chunks discovered — corpus empty?",
                  file=sys.stderr)
            return report

        # Sequential scan in manifest (=LBA) order.  We read in groups so the
        # "queries_run" counter stays meaningful (one query == one batch).
        batch = max(1, len(all_ids) // self.n_queries)
        for i in range(0, len(all_ids), batch):
            chunk_batch = all_ids[i:i + batch]
            pre_hits = sim._cache.hits
            pre_misses = sim._cache.misses
            bytes_in_batch = engine.read_chunks(chunk_batch)
            report.chunks_read += len(chunk_batch)
            report.bytes_read += bytes_in_batch
            report.cache_hits += sim._cache.hits - pre_hits
            report.cache_misses += sim._cache.misses - pre_misses
            report.queries_run += 1
            if report.queries_run >= self.n_queries:
                break

        report.elapsed_sec = time.monotonic() - t0
        report.trace_path = sim._trace.path.as_posix()
        return report


# ---------------------------------------------------------------------------
# Scenario C — mixed + incremental write
# ---------------------------------------------------------------------------


class ScenarioC(_BaseScenario):
    name = "scenario_c"
    label = "C"

    def __init__(self, n_queries: int = 300, seed: int = 0xC3,
                 write_ratio: float = 0.30) -> None:
        super().__init__(n_queries=n_queries, seed=seed)
        self.write_ratio = write_ratio

    def run(self, engine, sim) -> ScenarioReport:
        t0 = time.monotonic()
        rng = random.Random(self.seed)
        report = ScenarioReport(name=self.name)
        all_ids = engine.known_chunk_ids() or sim.known_chunks()

        for i in range(self.n_queries):
            if rng.random() < self.write_ratio:
                # 30% writes: simulate ingest of a new doc.
                cid = f"ingest_{int(time.time_ns())}_{i:06d}"
                size = rng.choice(
                    [2_000, 4_000, 8_000,           # text-sized
                     500_000, 1_500_000]            # image-sized
                )
                payload = os.urandom(size)
                kind = "image" if size > 100_000 else "text"
                pre = sim.stats()
                engine.ingest(cid, payload, kind=kind)
                post = sim.stats()
                report.bytes_written += size
                report.queries_run += 1
                report.cache_hits += post["cache"]["hits"] - pre["cache"]["hits"]
                report.cache_misses += post["cache"]["misses"] - pre["cache"]["misses"]
            else:
                # 70% queries: a random topic from a moderate hot set.
                if not all_ids:
                    continue
                # Pick a query word from a random chunk_id seed for variety.
                topic_seed = rng.choice(all_ids)
                q = f"summary of {topic_seed}"
                res = engine.query(q)
                report.queries_run += 1
                report.chunks_read += len(res.retrieved_chunks)
                report.bytes_read += res.bytes_read
                report.cache_hits += res.cache_hits
                report.cache_misses += res.cache_misses

        report.elapsed_sec = time.monotonic() - t0
        report.trace_path = sim._trace.path.as_posix()
        return report


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------


SCENARIOS: Dict[str, Callable[..., _BaseScenario]] = {
    "a": ScenarioA,
    "b": ScenarioB,
    "c": ScenarioC,
}


@dataclass
class ScenarioRunner:
    """Runs a list of scenarios end-to-end, each into its own trace file."""
    trace_dir: str = "traces"
    queries_per_scenario: int = 300
    engine_config: EngineConfig = field(default_factory=EngineConfig)
    tier_config: TierConfig = field(default_factory=TierConfig)

    def run(self, names: Sequence[str]) -> Dict[str, ScenarioReport]:
        Path(self.trace_dir).mkdir(parents=True, exist_ok=True)
        out: Dict[str, ScenarioReport] = {}

        for name in names:
            cls = SCENARIOS[name.lower()]
            scenario = cls(n_queries=self.queries_per_scenario)

            tcfg = TierConfig(
                hdd_root=self.tier_config.hdd_root,
                ssd_root=self.tier_config.ssd_root,
                cache_bytes=self.tier_config.cache_bytes,
                cache_fraction=self.tier_config.cache_fraction,
                block_size=self.tier_config.block_size,
                trace_path=os.path.join(
                    self.trace_dir, f"{scenario.name}.jsonl"
                ),
                scenario=scenario.label,
                fsync_on_write=self.tier_config.fsync_on_write,
            )
            with TieredStorageSimulator(tcfg) as sim:
                engine = RAGQueryEngine(sim, config=self.engine_config)
                report = scenario.run(engine, sim)
                out[scenario.name] = report
                print(
                    f"[scenarios] {scenario.name}: "
                    f"queries={report.queries_run} "
                    f"chunks_read={report.chunks_read} "
                    f"bytes_read={report.bytes_read} "
                    f"writes={report.bytes_written} "
                    f"elapsed={report.elapsed_sec:.2f}s "
                    f"trace={report.trace_path}",
                    file=sys.stderr,
                )
        return out
