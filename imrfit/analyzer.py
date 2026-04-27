"""analyzer.py — Module 4.

Parse a cold-tier I/O trace and compute, for every 128 MB block, the four
IMR-Fit feature dimensions and the composite score S(b).

Formal definitions (per spec)
-----------------------------
    F(b) = access_count(b) / max_b' access_count(b')
    Q(b) = fraction of accesses with |delta_LBA| < block_size * 0.1
    Z(b) = min(mean_io_size(b) / block_size, 1.0)
    R(b) = exp(-lambda * (t_now - t_last(b)))            (lambda = 0.1)
    S(b) = w_freq*F + w_seq*Q + w_size*Z + w_rec*R       (sum w == 1)

Outputs
-------
    placement_decisions.jsonl   one record per block (id, F, Q, Z, R, S, place)
    feature_distributions.json  per-dimension {mean, std, p50, p95, p99}
    score_variance.json         S(b) variance summary (the "killer figure"
                                 number — RAG should be > ResNet by a wide
                                 margin)

Weight sensitivity
------------------
    grid_search_weights() runs an exhaustive but constrained grid over
    {0.1, 0.2, 0.4, 0.6, 0.8} for each weight, keeping only combinations
    that sum to 1.0.  Each candidate is scored by the *mean RMW reduction*
    estimated from a simple analytic model:

        rmw_reduction(b) = max(0, S(b) - theta) * write_count(b)

    The top-k candidates are returned alongside their estimates.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from imrfit.scorer import Scorer, ScorerConfig, BlockPlacement, ScoreResult


BLOCK_SIZE_BYTES = 128 * 1024 * 1024     # 128 MB
RECENCY_LAMBDA = 0.1                     # per-spec
SEQ_DELTA_THRESHOLD = 0.1                # |dLBA| < block_size * 0.1


# ---------------------------------------------------------------------------
# Trace iteration (kept import-light to avoid a circular dep with storage/)
# ---------------------------------------------------------------------------


@dataclass
class _Access:
    ts_ns: int
    chunk_id: str
    lba: int
    size: int
    op: str
    block_id: int


def _iter_trace(trace_path: Path,
                block_size: int = BLOCK_SIZE_BYTES) -> Iterator[_Access]:
    with open(trace_path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            yield _Access(
                ts_ns=int(row["ts_ns"]),
                chunk_id=str(row["chunk_id"]),
                lba=int(row["lba"]),
                size=int(row["size"]),
                op=str(row["op"]),
                block_id=int(row["lba"]) // block_size,
            )


# ---------------------------------------------------------------------------
# Per-block accumulator
# ---------------------------------------------------------------------------


@dataclass
class _BlockAggregate:
    block_id: int
    access_count: int = 0
    write_count: int = 0
    sequential_count: int = 0
    bytes_accessed: int = 0
    last_ts_ns: int = 0
    first_ts_ns: int = 0

    def update(self, acc: _Access, prev_lba: Optional[int],
               block_size: int) -> None:
        if self.access_count == 0:
            self.first_ts_ns = acc.ts_ns
        self.access_count += 1
        if acc.op == "W":
            self.write_count += 1
        self.bytes_accessed += acc.size
        self.last_ts_ns = acc.ts_ns
        if prev_lba is not None:
            delta = abs(acc.lba - prev_lba)
            if delta < block_size * SEQ_DELTA_THRESHOLD:
                self.sequential_count += 1


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@dataclass
class AnalyzerConfig:
    block_size: int = BLOCK_SIZE_BYTES
    recency_lambda: float = RECENCY_LAMBDA
    scorer_config: ScorerConfig = field(default_factory=ScorerConfig)


@dataclass
class BlockFeatureRow:
    block_id: int
    access_count: int
    write_count: int
    F: float
    Q: float
    Z: float
    R: float
    S: float
    placement: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class Analyzer:
    """Trace -> per-block 4D features -> S(b) -> placement decisions."""

    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self.cfg = config or AnalyzerConfig()
        self._scorer = Scorer(self.cfg.scorer_config)

    # ------------------------------------------------------------------
    # Single-pass aggregation
    # ------------------------------------------------------------------

    def aggregate(self, trace_path: str) -> Tuple[Dict[int, _BlockAggregate], int]:
        """Single-pass scan over a trace.  Returns (aggregates, t_now_ns)."""
        aggregates: Dict[int, _BlockAggregate] = {}
        prev_lba: Optional[int] = None
        last_seen_ts = 0
        for acc in _iter_trace(Path(trace_path), self.cfg.block_size):
            agg = aggregates.get(acc.block_id)
            if agg is None:
                agg = _BlockAggregate(block_id=acc.block_id)
                aggregates[acc.block_id] = agg
            agg.update(acc, prev_lba, self.cfg.block_size)
            prev_lba = acc.lba
            last_seen_ts = acc.ts_ns
        return aggregates, last_seen_ts

    # ------------------------------------------------------------------
    # Feature -> score
    # ------------------------------------------------------------------

    def features_from(self,
                      aggregates: Dict[int, _BlockAggregate],
                      t_now_ns: int) -> Dict[int, Dict[str, float]]:
        """Compute (F, Q, Z, R) for every block."""
        if not aggregates:
            return {}

        max_count = max(a.access_count for a in aggregates.values())
        out: Dict[int, Dict[str, float]] = {}
        for bid, agg in aggregates.items():
            F = agg.access_count / max_count if max_count else 0.0
            Q = (agg.sequential_count / agg.access_count
                 if agg.access_count else 0.0)
            mean_size = (agg.bytes_accessed / agg.access_count
                         if agg.access_count else 0)
            Z = min(mean_size / self.cfg.block_size, 1.0)
            dt_sec = max(0.0, (t_now_ns - agg.last_ts_ns) / 1e9)
            R = math.exp(-self.cfg.recency_lambda * dt_sec)
            out[bid] = {"F": F, "Q": Q, "Z": Z, "R": R}
        return out

    # ------------------------------------------------------------------
    # End-to-end pipeline
    # ------------------------------------------------------------------

    def analyze(self, trace_path: str) -> Dict[str, Any]:
        aggregates, t_now_ns = self.aggregate(trace_path)
        features = self.features_from(aggregates, t_now_ns)
        score_results = self._scorer.score_all(features)

        rows: List[BlockFeatureRow] = []
        for bid, sr in score_results.items():
            agg = aggregates[bid]
            rows.append(BlockFeatureRow(
                block_id=bid,
                access_count=agg.access_count,
                write_count=agg.write_count,
                F=sr.F, Q=sr.Q, Z=sr.Z, R=sr.R,
                S=sr.score,
                placement=sr.placement.value,
            ))

        # Stats / distributions
        distributions = self._distributions(rows)
        s_var = self._score_variance(rows)

        return {
            "trace_path": trace_path,
            "blocks": [r.to_dict() for r in rows],
            "summary": self._scorer.score_summary(score_results),
            "distributions": distributions,
            "score_variance": s_var,
            "weights": {
                "w_freq": self.cfg.scorer_config.w_freq,
                "w_seq": self.cfg.scorer_config.w_seq,
                "w_size": self.cfg.scorer_config.w_size,
                "w_rec": self.cfg.scorer_config.w_rec,
                "theta": self.cfg.scorer_config.theta,
            },
        }

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distributions(rows: List[BlockFeatureRow]) -> Dict[str, Dict[str, float]]:
        if not rows:
            return {}
        out: Dict[str, Dict[str, float]] = {}
        for dim in ("F", "Q", "Z", "R", "S"):
            vals = [getattr(r, dim) for r in rows]
            out[dim] = {
                "mean": statistics.fmean(vals),
                "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "p50": _percentile(vals, 50),
                "p95": _percentile(vals, 95),
                "p99": _percentile(vals, 99),
                "min": min(vals),
                "max": max(vals),
            }
        return out

    @staticmethod
    def _score_variance(rows: List[BlockFeatureRow]) -> Dict[str, float]:
        if not rows:
            return {"variance": 0.0, "n": 0}
        scores = [r.S for r in rows]
        var = statistics.pvariance(scores)
        return {
            "variance": var,
            "stdev": math.sqrt(var),
            "n": len(scores),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def write_decisions(report: Dict[str, Any], out_path: str) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for row in report["blocks"]:
                fh.write(json.dumps(row) + "\n")

    @staticmethod
    def write_summary(report: Dict[str, Any], out_path: str) -> None:
        slim = {k: v for k, v in report.items() if k != "blocks"}
        slim["n_blocks"] = len(report["blocks"])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(slim, fh, indent=2)


# ---------------------------------------------------------------------------
# Weight sensitivity / grid search
# ---------------------------------------------------------------------------


def _generate_weight_grid(values: Sequence[float]) -> List[Tuple[float, float, float, float]]:
    """All (w_freq, w_seq, w_size, w_rec) combinations from `values` whose
    sum is 1.0 (within float tolerance).
    """
    out: List[Tuple[float, float, float, float]] = []
    for wf in values:
        for wq in values:
            for wz in values:
                wr = 1.0 - wf - wq - wz
                if any(math.isclose(wr, v, abs_tol=1e-6) for v in values):
                    if wr >= 0:
                        out.append((wf, wq, wz, wr))
    return out


def grid_search_weights(
    trace_path: str,
    *,
    grid: Sequence[float] = (0.1, 0.2, 0.4, 0.6, 0.8),
    theta: float = 0.55,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Sweep weight combinations and rank by predicted RMW reduction.

    Cost model (cheap analytic surrogate)
    -------------------------------------
    rmw_reduction(b) = max(0, S(b) - theta) * write_count(b)

    The intuition: only writes can trigger RMW, and only blocks whose score
    crosses the threshold benefit from migration to the Top track.  Summing
    across all blocks gives a per-config figure of merit.  This is
    deliberately a surrogate — the real RMW count comes from Module 5's
    replay engine and is what the paper plots.  The grid search just
    narrows down which weights to ship to the replayer.
    """
    candidates = _generate_weight_grid(grid)
    results: List[Dict[str, Any]] = []

    # Aggregate once — reused across every weight configuration.
    base = Analyzer(AnalyzerConfig())
    aggregates, t_now = base.aggregate(trace_path)
    features = base.features_from(aggregates, t_now)

    for wf, wq, wz, wr in candidates:
        cfg = ScorerConfig(w_freq=wf, w_seq=wq, w_size=wz, w_rec=wr,
                           theta=theta)
        scorer = Scorer(cfg)
        score_results = scorer.score_all(features)
        rmw_red = 0.0
        for bid, sr in score_results.items():
            rmw_red += max(0.0, sr.score - theta) * aggregates[bid].write_count
        results.append({
            "weights": {"w_freq": wf, "w_seq": wq,
                        "w_size": wz, "w_rec": wr},
            "rmw_reduction": rmw_red,
            "top_blocks": sum(1 for sr in score_results.values()
                              if sr.placement == BlockPlacement.TOP),
        })

    results.sort(key=lambda r: -r["rmw_reduction"])
    return results[:top_k]


# ---------------------------------------------------------------------------
# Helper: numeric percentile w/o numpy dependency
# ---------------------------------------------------------------------------


def _percentile(vals: List[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True, help="cold-tier trace JSONL")
    p.add_argument("--out-dir", default="results/latest/imrfit",
                   help="output directory")
    p.add_argument("--theta", type=float, default=0.55)
    p.add_argument("--w-freq", type=float, default=0.35)
    p.add_argument("--w-seq", type=float, default=0.30)
    p.add_argument("--w-size", type=float, default=0.15)
    p.add_argument("--w-rec", type=float, default=0.20)
    p.add_argument("--grid-search", action="store_true",
                   help="run weight grid search and emit top-10 table")
    args = p.parse_args(argv)

    cfg = AnalyzerConfig(
        scorer_config=ScorerConfig(
            w_freq=args.w_freq, w_seq=args.w_seq,
            w_size=args.w_size, w_rec=args.w_rec,
            theta=args.theta,
            recency_lambda=RECENCY_LAMBDA,
        )
    )
    analyzer = Analyzer(cfg)
    report = analyzer.analyze(args.trace)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    analyzer.write_decisions(report, str(out / "placement_decisions.jsonl"))
    analyzer.write_summary(report, str(out / "summary.json"))

    if args.grid_search:
        gs = grid_search_weights(args.trace, theta=args.theta)
        with open(out / "grid_search.json", "w", encoding="utf-8") as fh:
            json.dump(gs, fh, indent=2)

    print(json.dumps({
        "n_blocks": len(report["blocks"]),
        "summary": report["summary"],
        "score_variance": report["score_variance"],
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
