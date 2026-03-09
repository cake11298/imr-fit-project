"""
test_scorer.py - pytest tests for imrfit.scorer

Tests cover:
  - ScorerConfig weight validation
  - score_block correctness
  - score_all multi-block scoring
  - TOP/BOTTOM placement decisions
  - displacement metric D(e)
  - Recency calculation R(b)
  - summary statistics
"""

import math
import time

import pytest

from imrfit.scorer import (
    BlockPlacement,
    ScoreResult,
    Scorer,
    ScorerConfig,
)


# ---------------------------------------------------------------------------
# ScorerConfig tests
# ---------------------------------------------------------------------------

class TestScorerConfig:
    def test_default_weights_sum_to_one(self):
        cfg = ScorerConfig()
        total = cfg.w_freq + cfg.w_seq + cfg.w_size + cfg.w_rec
        assert math.isclose(total, 1.0, rel_tol=1e-6)

    def test_custom_weights_valid(self):
        cfg = ScorerConfig(w_freq=0.25, w_seq=0.25, w_size=0.25, w_rec=0.25)
        total = cfg.w_freq + cfg.w_seq + cfg.w_size + cfg.w_rec
        assert math.isclose(total, 1.0)

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ScorerConfig(w_freq=0.5, w_seq=0.5, w_size=0.5, w_rec=0.5)

    def test_invalid_theta_low(self):
        with pytest.raises(ValueError, match="theta"):
            ScorerConfig(theta=-0.1)

    def test_invalid_theta_high(self):
        with pytest.raises(ValueError, match="theta"):
            ScorerConfig(theta=1.1)

    def test_theta_boundaries(self):
        # theta=0.0 and theta=1.0 should be accepted
        ScorerConfig(theta=0.0)
        ScorerConfig(theta=1.0)


# ---------------------------------------------------------------------------
# Scorer.score_block tests
# ---------------------------------------------------------------------------

class TestScoreBlock:
    def setup_method(self):
        self.scorer = Scorer()  # default config

    def test_score_formula(self):
        cfg = self.scorer.config
        F, Q, Z, R = 0.8, 0.6, 0.4, 0.9
        expected = cfg.w_freq * F + cfg.w_seq * Q + cfg.w_size * Z + cfg.w_rec * R
        result = self.scorer.score_block(0, F, Q, Z, R)
        assert math.isclose(result.score, expected, rel_tol=1e-9)

    def test_score_in_range(self):
        # With all metrics in [0,1] and weights summing to 1, score in [0,1]
        result = self.scorer.score_block(0, 1.0, 1.0, 1.0, 1.0)
        assert 0.0 <= result.score <= 1.0
        result2 = self.scorer.score_block(0, 0.0, 0.0, 0.0, 0.0)
        assert math.isclose(result2.score, 0.0)

    def test_placement_top_above_theta(self):
        # score = 1.0 > default theta 0.55 -> TOP
        result = self.scorer.score_block(0, 1.0, 1.0, 1.0, 1.0)
        assert result.placement == BlockPlacement.TOP

    def test_placement_bottom_below_theta(self):
        # score = 0.0 < theta -> BOTTOM
        result = self.scorer.score_block(0, 0.0, 0.0, 0.0, 0.0)
        assert result.placement == BlockPlacement.BOTTOM

    def test_placement_at_theta_is_top(self):
        # S(b) == theta -> TOP (>= theta condition)
        scorer = Scorer(ScorerConfig(theta=0.5))
        # w_freq=0.35, w_seq=0.30, w_size=0.15, w_rec=0.20
        # Need S = 0.5. Let F=Q=Z=R=0.5 => S = 0.5
        result = scorer.score_block(0, 0.5, 0.5, 0.5, 0.5)
        assert math.isclose(result.score, 0.5)
        assert result.placement == BlockPlacement.TOP

    def test_result_fields(self):
        result = self.scorer.score_block(42, 0.3, 0.4, 0.2, 0.5)
        assert result.block_id == 42
        assert result.F == 0.3
        assert result.Q == 0.4
        assert result.Z == 0.2
        assert result.R == 0.5

    def test_as_dict(self):
        result = self.scorer.score_block(1, 0.5, 0.5, 0.5, 0.5)
        d = result.as_dict()
        assert "block_id" in d
        assert "score" in d
        assert "placement" in d
        assert d["placement"] in ("TOP", "BOTTOM")


# ---------------------------------------------------------------------------
# Scorer.score_all tests
# ---------------------------------------------------------------------------

class TestScoreAll:
    def setup_method(self):
        self.scorer = Scorer()

    def test_empty_metrics(self):
        results = self.scorer.score_all({})
        assert results == {}

    def test_multiple_blocks(self):
        metrics = {
            0: {"F": 0.9, "Q": 0.8, "Z": 0.7, "R": 0.9},
            1: {"F": 0.1, "Q": 0.1, "Z": 0.1, "R": 0.1},
            2: {"F": 0.5, "Q": 0.5, "Z": 0.5, "R": 0.5},
        }
        results = self.scorer.score_all(metrics)
        assert len(results) == 3
        assert results[0].placement == BlockPlacement.TOP   # high score
        assert results[1].placement == BlockPlacement.BOTTOM  # low score

    def test_missing_metric_defaults_to_zero(self):
        metrics = {0: {"F": 0.5}}  # Q, Z, R missing
        results = self.scorer.score_all(metrics)
        # score = 0.35 * 0.5 = 0.175 < 0.55 -> BOTTOM
        assert results[0].placement == BlockPlacement.BOTTOM
        assert math.isclose(results[0].score, 0.35 * 0.5, rel_tol=1e-9)

    def test_block_ids_preserved(self):
        metrics = {10: {"F": 1, "Q": 1, "Z": 1, "R": 1},
                   20: {"F": 0, "Q": 0, "Z": 0, "R": 0}}
        results = self.scorer.score_all(metrics)
        assert 10 in results
        assert 20 in results
        assert results[10].block_id == 10
        assert results[20].block_id == 20


# ---------------------------------------------------------------------------
# Displacement metric tests
# ---------------------------------------------------------------------------

class TestDisplacement:
    def setup_method(self):
        self.scorer = Scorer()

    def _make_results(self, placements_map):
        """Helper: create ScoreResults from {block_id: BlockPlacement}."""
        results = {}
        for bid, pl in placements_map.items():
            score = 0.8 if pl == BlockPlacement.TOP else 0.2
            results[bid] = ScoreResult(
                block_id=bid, F=0, Q=0, Z=0, R=0,
                score=score, placement=pl,
            )
        return results

    def test_perfect_placement_zero_displacement(self):
        optimal = {0: BlockPlacement.TOP, 1: BlockPlacement.BOTTOM}
        score_results = self._make_results(optimal)
        # current == optimal
        d = Scorer.displacement(score_results, optimal)
        assert math.isclose(d, 0.0)

    def test_all_misplaced_full_displacement(self):
        optimal = {0: BlockPlacement.TOP, 1: BlockPlacement.TOP}
        score_results = self._make_results(optimal)
        current = {0: BlockPlacement.BOTTOM, 1: BlockPlacement.BOTTOM}
        d = Scorer.displacement(score_results, current)
        assert math.isclose(d, 1.0)

    def test_half_misplaced(self):
        optimal = {
            0: BlockPlacement.TOP,
            1: BlockPlacement.TOP,
            2: BlockPlacement.BOTTOM,
            3: BlockPlacement.BOTTOM,
        }
        score_results = self._make_results(optimal)
        current = {
            0: BlockPlacement.TOP,     # correct
            1: BlockPlacement.BOTTOM,  # wrong
            2: BlockPlacement.BOTTOM,  # correct
            3: BlockPlacement.TOP,     # wrong
        }
        d = Scorer.displacement(score_results, current)
        assert math.isclose(d, 0.5)

    def test_empty_results_zero_displacement(self):
        d = Scorer.displacement({}, {})
        assert d == 0.0

    def test_unknown_block_defaults_to_bottom(self):
        # block 0 optimal is TOP, current_placements has no entry -> defaults BOTTOM -> misplaced
        optimal = {0: BlockPlacement.TOP}
        score_results = self._make_results(optimal)
        d = Scorer.displacement(score_results, {})
        assert math.isclose(d, 1.0)


# ---------------------------------------------------------------------------
# Recency helper tests
# ---------------------------------------------------------------------------

class TestRecency:
    def test_recency_at_zero_delta(self):
        t = time.monotonic()
        r = Scorer.compute_recency(t_last=t, t_now=t, lam=1.0)
        assert math.isclose(r, 1.0)

    def test_recency_decreases_over_time(self):
        t_now = 100.0
        r1 = Scorer.compute_recency(t_last=99.0, t_now=t_now, lam=1.0)
        r2 = Scorer.compute_recency(t_last=95.0, t_now=t_now, lam=1.0)
        assert r1 > r2

    def test_recency_formula(self):
        lam = 2.0
        delta = 3.0
        expected = math.exp(-lam * delta)
        r = Scorer.compute_recency(t_last=0.0, t_now=delta, lam=lam)
        assert math.isclose(r, expected, rel_tol=1e-9)

    def test_recency_negative_delta_clipped(self):
        # t_now < t_last should not give >1 (delta clamped to 0)
        r = Scorer.compute_recency(t_last=100.0, t_now=90.0, lam=1.0)
        assert math.isclose(r, 1.0)

    def test_recency_lambda_zero(self):
        # lam=0 => R = exp(0) = 1 always
        r = Scorer.compute_recency(t_last=0.0, t_now=1000.0, lam=0.0)
        assert math.isclose(r, 1.0)


# ---------------------------------------------------------------------------
# Scorer utilities: top_blocks, bottom_blocks, score_summary
# ---------------------------------------------------------------------------

class TestScorerUtilities:
    def setup_method(self):
        self.scorer = Scorer()
        metrics = {
            0: {"F": 1.0, "Q": 1.0, "Z": 1.0, "R": 1.0},
            1: {"F": 0.0, "Q": 0.0, "Z": 0.0, "R": 0.0},
            2: {"F": 0.5, "Q": 0.6, "Z": 0.5, "R": 0.6},
        }
        self.results = self.scorer.score_all(metrics)

    def test_top_blocks_sorted_descending(self):
        top = self.scorer.top_blocks(self.results)
        scores = [r.score for r in top]
        assert scores == sorted(scores, reverse=True)

    def test_bottom_blocks_sorted_ascending(self):
        bottom = self.scorer.bottom_blocks(self.results)
        scores = [r.score for r in bottom]
        assert scores == sorted(scores)

    def test_top_n_limit(self):
        top = self.scorer.top_blocks(self.results, n=1)
        assert len(top) <= 1

    def test_score_summary_keys(self):
        summary = self.scorer.score_summary(self.results)
        for key in ("total_blocks", "top_blocks", "bottom_blocks",
                    "mean_score", "min_score", "max_score", "theta"):
            assert key in summary

    def test_score_summary_totals(self):
        summary = self.scorer.score_summary(self.results)
        assert summary["top_blocks"] + summary["bottom_blocks"] == summary["total_blocks"]

    def test_score_summary_empty(self):
        assert self.scorer.score_summary({}) == {}
