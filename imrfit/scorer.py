"""
scorer.py - Block placement scorer for IMR-Fit.

Computes S(b) for each block and decides placement:

    S(b) = w_freq*F(b) + w_seq*Q(b) + w_size*Z(b) + w_rec*R(b)

    P(b) = TOP    if S(b) >= theta
           BOTTOM otherwise

Displacement metric:
    D(e) = |misplaced blocks| / |total blocks|
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple


class BlockPlacement(Enum):
    TOP = "TOP"        # Top track — free read/write
    BOTTOM = "BOTTOM"  # Bottom track — RMW required


@dataclass
class ScorerConfig:
    """Weights and threshold for S(b) calculation."""
    w_freq: float = 0.35   # weight for access frequency F(b)
    w_seq: float = 0.30    # weight for sequential ratio Q(b)
    w_size: float = 0.15   # weight for size weight Z(b)
    w_rec: float = 0.20    # weight for recency R(b)
    theta: float = 0.55    # placement threshold
    recency_lambda: float = 1.0  # decay constant for R(b)

    def __post_init__(self) -> None:
        total = self.w_freq + self.w_seq + self.w_size + self.w_rec
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}. "
                f"(w_freq={self.w_freq}, w_seq={self.w_seq}, "
                f"w_size={self.w_size}, w_rec={self.w_rec})"
            )
        if not (0.0 <= self.theta <= 1.0):
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")


@dataclass
class ScoreResult:
    """Result of scoring a single block."""
    block_id: int
    F: float          # normalised frequency
    Q: float          # sequential ratio
    Z: float          # size weight
    R: float          # recency
    score: float      # S(b)
    placement: BlockPlacement

    def as_dict(self) -> Dict:
        return {
            "block_id": self.block_id,
            "F": self.F,
            "Q": self.Q,
            "Z": self.Z,
            "R": self.R,
            "score": self.score,
            "placement": self.placement.value,
        }


class Scorer:
    """
    Stateless scorer: computes S(b) and placement decisions.

    Usage::

        config = ScorerConfig(w_freq=0.35, w_seq=0.30, w_size=0.15, w_rec=0.20)
        scorer = Scorer(config)

        # metrics: dict[block_id -> {"F": ..., "Q": ..., "Z": ..., "R": ...}]
        results = scorer.score_all(metrics)
        displacement = scorer.displacement(results, current_placements)
    """

    def __init__(self, config: Optional[ScorerConfig] = None) -> None:
        self.config = config or ScorerConfig()

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_block(
        self,
        block_id: int,
        F: float,
        Q: float,
        Z: float,
        R: float,
    ) -> ScoreResult:
        """
        Compute S(b) for a single block given its (F, Q, Z, R) metrics.

        All inputs should be normalised to [0, 1].
        """
        cfg = self.config
        s = cfg.w_freq * F + cfg.w_seq * Q + cfg.w_size * Z + cfg.w_rec * R
        placement = BlockPlacement.TOP if s >= cfg.theta else BlockPlacement.BOTTOM
        return ScoreResult(
            block_id=block_id,
            F=F, Q=Q, Z=Z, R=R,
            score=s,
            placement=placement,
        )

    def score_all(
        self,
        metrics: Dict[int, Dict[str, float]],
    ) -> Dict[int, ScoreResult]:
        """
        Score all blocks.

        Args:
            metrics: dict mapping block_id -> {"F": f, "Q": q, "Z": z, "R": r}

        Returns:
            dict mapping block_id -> ScoreResult
        """
        results: Dict[int, ScoreResult] = {}
        for bid, m in metrics.items():
            results[bid] = self.score_block(
                block_id=bid,
                F=m.get("F", 0.0),
                Q=m.get("Q", 0.0),
                Z=m.get("Z", 0.0),
                R=m.get("R", 0.0),
            )
        return results

    # ------------------------------------------------------------------
    # Recency helper (static — also used by profiler)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_recency(t_last: float, t_now: Optional[float] = None, lam: float = 1.0) -> float:
        """R(b) = exp(-lambda * (t_now - t_last))."""
        if t_now is None:
            t_now = time.monotonic()
        delta = max(t_now - t_last, 0.0)
        return math.exp(-lam * delta)

    # ------------------------------------------------------------------
    # Displacement metric
    # ------------------------------------------------------------------

    @staticmethod
    def displacement(
        score_results: Dict[int, ScoreResult],
        current_placements: Dict[int, BlockPlacement],
    ) -> float:
        """
        D(e) = |misplaced blocks| / |total blocks|

        A block is misplaced when its current physical placement differs
        from the optimal placement derived from S(b).

        Args:
            score_results: output of score_all()
            current_placements: dict block_id -> current BlockPlacement

        Returns:
            Displacement ratio in [0, 1].  0 = perfectly placed.
        """
        total = len(score_results)
        if total == 0:
            return 0.0

        misplaced = sum(
            1
            for bid, result in score_results.items()
            if current_placements.get(bid, BlockPlacement.BOTTOM) != result.placement
        )
        return misplaced / total

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def top_blocks(
        self,
        score_results: Dict[int, ScoreResult],
        n: Optional[int] = None,
    ) -> list[ScoreResult]:
        """Return blocks assigned to TOP track, sorted by score descending."""
        top = [r for r in score_results.values() if r.placement == BlockPlacement.TOP]
        top.sort(key=lambda r: r.score, reverse=True)
        return top[:n] if n is not None else top

    def bottom_blocks(
        self,
        score_results: Dict[int, ScoreResult],
        n: Optional[int] = None,
    ) -> list[ScoreResult]:
        """Return blocks assigned to BOTTOM track, sorted by score ascending."""
        bottom = [r for r in score_results.values() if r.placement == BlockPlacement.BOTTOM]
        bottom.sort(key=lambda r: r.score)
        return bottom[:n] if n is not None else bottom

    def score_summary(self, score_results: Dict[int, ScoreResult]) -> Dict:
        """Aggregate statistics over all scored blocks."""
        if not score_results:
            return {}
        scores = [r.score for r in score_results.values()]
        n_top = sum(1 for r in score_results.values() if r.placement == BlockPlacement.TOP)
        return {
            "total_blocks": len(scores),
            "top_blocks": n_top,
            "bottom_blocks": len(scores) - n_top,
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "theta": self.config.theta,
        }
