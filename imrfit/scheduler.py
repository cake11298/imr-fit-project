"""
scheduler.py - Migration scheduler for IMR-Fit.

Given a set of ScoreResults and a migration budget (max blocks to move per
epoch), the scheduler produces an ordered MigrationPlan that maximises the
improvement in displacement D(e).

Strategy:
  1. Identify misplaced blocks (current placement != optimal).
  2. Prioritise by |S(b) - theta|: blocks far from the threshold gain/lose
     most from correct placement.
  3. Emit at most `budget` migration operations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .scorer import BlockPlacement, ScoreResult


@dataclass
class MigrationOp:
    """A single block migration operation."""
    block_id: int
    from_placement: BlockPlacement
    to_placement: BlockPlacement
    score: float
    priority: float  # |S(b) - theta| — higher = more urgent

    def __repr__(self) -> str:
        return (
            f"MigrationOp(block={self.block_id}, "
            f"{self.from_placement.value} -> {self.to_placement.value}, "
            f"score={self.score:.4f}, priority={self.priority:.4f})"
        )


@dataclass
class MigrationPlan:
    """Ordered list of migration operations for one scheduling round."""
    epoch: int
    operations: List[MigrationOp] = field(default_factory=list)
    budget: int = 0
    created_at: float = field(default_factory=time.monotonic)

    @property
    def to_top(self) -> List[MigrationOp]:
        return [op for op in self.operations if op.to_placement == BlockPlacement.TOP]

    @property
    def to_bottom(self) -> List[MigrationOp]:
        return [op for op in self.operations if op.to_placement == BlockPlacement.BOTTOM]

    def summary(self) -> Dict:
        return {
            "epoch": self.epoch,
            "budget": self.budget,
            "total_ops": len(self.operations),
            "promote_to_top": len(self.to_top),
            "demote_to_bottom": len(self.to_bottom),
        }


class MigrationScheduler:
    """
    Schedules block migrations between IMR Top and Bottom tracks.

    Usage::

        scheduler = MigrationScheduler(budget=10, theta=0.55)
        plan = scheduler.plan(
            epoch=3,
            score_results=scorer.score_all(metrics),
            current_placements=placement_map,
        )
        # Apply the plan (in practice: call imrsim_util or dmsetup)
        updated = scheduler.apply_plan(plan, current_placements)
    """

    def __init__(
        self,
        budget: int = 10,
        theta: float = 0.55,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be >= 0")
        self.budget = budget
        self.theta = theta
        self._history: List[MigrationPlan] = []

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(
        self,
        epoch: int,
        score_results: Dict[int, ScoreResult],
        current_placements: Dict[int, BlockPlacement],
    ) -> MigrationPlan:
        """
        Produce a MigrationPlan for one epoch.

        Only misplaced blocks are considered.  Operations are sorted by
        priority (distance from theta) descending so the most impactful
        migrations happen first.

        Args:
            epoch: current epoch number (for bookkeeping)
            score_results: output of Scorer.score_all()
            current_placements: dict block_id -> current BlockPlacement
                                 (defaults to BOTTOM for unknown blocks)

        Returns:
            MigrationPlan with at most `self.budget` operations.
        """
        candidates: List[MigrationOp] = []

        for bid, result in score_results.items():
            current = current_placements.get(bid, BlockPlacement.BOTTOM)
            optimal = result.placement

            if current != optimal:
                priority = abs(result.score - self.theta)
                candidates.append(MigrationOp(
                    block_id=bid,
                    from_placement=current,
                    to_placement=optimal,
                    score=result.score,
                    priority=priority,
                ))

        # Sort by priority descending; secondary sort by block_id for stability
        candidates.sort(key=lambda op: (-op.priority, op.block_id))

        plan = MigrationPlan(
            epoch=epoch,
            operations=candidates[: self.budget],
            budget=self.budget,
        )
        self._history.append(plan)
        return plan

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    @staticmethod
    def apply_plan(
        plan: MigrationPlan,
        current_placements: Dict[int, BlockPlacement],
    ) -> Dict[int, BlockPlacement]:
        """
        Apply a MigrationPlan to a placement map (in-memory simulation).

        Returns a *new* dict with updated placements (non-destructive).

        In a real system, each operation would invoke dmsetup / imrsim_util.
        """
        updated = dict(current_placements)
        for op in plan.operations:
            updated[op.block_id] = op.to_placement
        return updated

    # ------------------------------------------------------------------
    # History / reporting
    # ------------------------------------------------------------------

    def history_summary(self) -> List[Dict]:
        """Return summary dicts for all past migration plans."""
        return [plan.summary() for plan in self._history]

    def total_migrations(self) -> int:
        """Total number of migration operations issued across all epochs."""
        return sum(len(p.operations) for p in self._history)
