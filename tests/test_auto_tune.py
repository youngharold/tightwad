"""Tests for adaptive draft-token tuning."""

import pytest

from tightwad.proxy import (
    AdaptiveDraftTokens,
    _AUTO_HIGH_THRESHOLD,
    _AUTO_LOW_THRESHOLD,
    _AUTO_MAX,
    _AUTO_MIN,
    _AUTO_STEP_DOWN,
    _AUTO_STEP_UP,
    _AUTO_WINDOW,
)


class TestAdaptiveDraftTokens:
    def test_initial_value(self):
        a = AdaptiveDraftTokens(initial=32)
        assert a.current == 32

    def test_clamped_to_range(self):
        a = AdaptiveDraftTokens(initial=1000)
        assert a.current == _AUTO_MAX

        a = AdaptiveDraftTokens(initial=1)
        assert a.current == _AUTO_MIN

    def test_no_adjustment_before_min_window(self):
        a = AdaptiveDraftTokens(initial=16)
        # Record fewer rounds than half the window
        for _ in range(_AUTO_WINDOW // 2 - 1):
            a.record_round(drafted=16, accepted=16)
        assert a.current == 16
        assert a.adjustments == 0

    def test_increases_on_high_acceptance(self):
        a = AdaptiveDraftTokens(initial=16)
        # Feed high acceptance rounds
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=15)  # 93.75%
        assert a.current > 16
        assert a.adjustments >= 1

    def test_decreases_on_low_acceptance(self):
        a = AdaptiveDraftTokens(initial=32)
        # Feed low acceptance rounds
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=32, accepted=5)  # 15.6%
        assert a.current < 32
        assert a.adjustments >= 1

    def test_stays_stable_in_middle(self):
        a = AdaptiveDraftTokens(initial=16)
        # Feed moderate acceptance (between thresholds)
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=10)  # 62.5%
        assert a.current == 16
        assert a.adjustments == 0

    def test_does_not_exceed_max(self):
        a = AdaptiveDraftTokens(initial=_AUTO_MAX)
        for _ in range(_AUTO_WINDOW * 3):
            a.record_round(drafted=64, accepted=64)  # 100%
        assert a.current == _AUTO_MAX

    def test_does_not_go_below_min(self):
        a = AdaptiveDraftTokens(initial=_AUTO_MIN)
        for _ in range(_AUTO_WINDOW * 3):
            a.record_round(drafted=4, accepted=0)  # 0%
        assert a.current == _AUTO_MIN

    def test_rolling_window_forgets_old_data(self):
        a = AdaptiveDraftTokens(initial=16)
        # First: fill window with high acceptance -> increases
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=16)
        increased = a.current
        assert increased > 16

        # Now: fill window with low acceptance -> should decrease
        for _ in range(_AUTO_WINDOW * 2):
            a.record_round(drafted=increased, accepted=1)
        assert a.current < increased

    def test_rolling_acceptance_rate(self):
        a = AdaptiveDraftTokens(initial=16)
        a.record_round(drafted=10, accepted=8)
        a.record_round(drafted=10, accepted=6)
        assert abs(a.rolling_acceptance - 0.7) < 0.01

    def test_rolling_acceptance_empty(self):
        a = AdaptiveDraftTokens(initial=16)
        assert a.rolling_acceptance == 0.0

    def test_zero_drafted_ignored(self):
        a = AdaptiveDraftTokens(initial=16)
        a.record_round(drafted=0, accepted=0)
        assert a.rolling_acceptance == 0.0
        assert len(a._window) == 0

    def test_multiple_adjustments(self):
        a = AdaptiveDraftTokens(initial=16)
        # Many high-acceptance rounds
        for _ in range(_AUTO_WINDOW * 5):
            a.record_round(drafted=a.current, accepted=a.current)
        # Should have made multiple upward adjustments
        assert a.adjustments > 1
        assert a.current > 16


class TestCostAwareTuning:
    """Tests for cost-aware adaptive draft length tuning.

    The key insight: when verify dominates (draft is cheap), increase
    draft length aggressively.  When draft dominates, keep drafts short.
    """

    def test_verify_dominates_increases_aggressively(self):
        """When draft=5ms and verify=500ms, step up more than base _AUTO_STEP_UP."""
        a = AdaptiveDraftTokens(initial=16)
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=15,  # 93.75% > 80%
                           draft_ms=5.0, verify_ms=500.0)
        # ratio=0.01 -> aggression=min(3, 1/sqrt(0.01))=min(3,10)=3.0
        # step=max(1, int(4*3.0))=12 per adjustment (vs 4 baseline)
        assert a.current > 16 + _AUTO_STEP_UP  # more than one base step
        assert a.adjustments >= 1

    def test_draft_dominates_increases_conservatively(self):
        """When draft=200ms and verify=100ms, step up less than base."""
        a = AdaptiveDraftTokens(initial=16)
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=15,  # 93.75%
                           draft_ms=200.0, verify_ms=100.0)
        # ratio=2.0 -> aggression=1/sqrt(2)~0.71 -> step=max(1, int(4*0.71))=2
        # Should increase but by less than the no-timing case
        assert a.current > 16
        # With the same acceptance and window, no-timing would step by 4
        b = AdaptiveDraftTokens(initial=16)
        for _ in range(_AUTO_WINDOW):
            b.record_round(drafted=16, accepted=15)  # no timing
        assert a.current <= b.current

    def test_verify_dominates_decreases_conservatively(self):
        """When verify is expensive but acceptance is low, still decrease
        but less aggressively since verify cost dominates anyway."""
        a = AdaptiveDraftTokens(initial=32)
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=32, accepted=5,  # 15.6% < 40%
                           draft_ms=5.0, verify_ms=500.0)
        # ratio=0.01 -> cut_aggression=sqrt(0.01)=0.1 -> step=max(1, int(4*0.1))=1
        assert a.current < 32
        # Should decrease less than the no-timing case
        b = AdaptiveDraftTokens(initial=32)
        for _ in range(_AUTO_WINDOW):
            b.record_round(drafted=32, accepted=5)
        assert a.current >= b.current  # cost-aware cuts less when draft is cheap

    def test_draft_dominates_decreases_aggressively(self):
        """When draft is expensive and acceptance is low, cut hard."""
        a = AdaptiveDraftTokens(initial=32)
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=32, accepted=5,  # 15.6%
                           draft_ms=200.0, verify_ms=100.0)
        # ratio=2.0 -> cut_aggression=sqrt(2)~1.41 -> step=max(1, int(4*1.41))=5
        assert a.current < 32
        # Should decrease more than the no-timing case
        b = AdaptiveDraftTokens(initial=32)
        for _ in range(_AUTO_WINDOW):
            b.record_round(drafted=32, accepted=5)
        assert a.current <= b.current  # cost-aware cuts more when draft is expensive

    def test_balanced_timing_matches_baseline(self):
        """When draft_ms == verify_ms, ratio=1.0, aggression=1.0 -> same as baseline."""
        a = AdaptiveDraftTokens(initial=16)
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=15,
                           draft_ms=100.0, verify_ms=100.0)
        b = AdaptiveDraftTokens(initial=16)
        for _ in range(_AUTO_WINDOW):
            b.record_round(drafted=16, accepted=15)  # no timing
        assert a.current == b.current

    def test_no_timing_falls_back_to_threshold(self):
        """Without timing data, uses the original threshold-based stepping."""
        a = AdaptiveDraftTokens(initial=16)
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=15)  # 93.75%, no timing
        # Multiple adjustments happen once window is half-full; verify no timing used
        assert a.current > 16
        assert a.adjustments >= 1
        assert a.draft_verify_ratio is None

    def test_draft_verify_ratio_property(self):
        a = AdaptiveDraftTokens(initial=16)
        assert a.draft_verify_ratio is None

        a.record_round(drafted=10, accepted=8, draft_ms=10.0, verify_ms=100.0)
        a.record_round(drafted=10, accepted=8, draft_ms=20.0, verify_ms=200.0)
        # total_d=30, total_v=300 -> ratio=0.1
        assert abs(a.draft_verify_ratio - 0.1) < 0.01

    def test_timing_window_rolls(self):
        """Timing window respects _AUTO_WINDOW size."""
        a = AdaptiveDraftTokens(initial=16)
        # Fill with fast drafts
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=15,
                           draft_ms=5.0, verify_ms=500.0)
        assert abs(a.draft_verify_ratio - 0.01) < 0.001

        # Now fill with slow drafts — old data should roll off
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=15,
                           draft_ms=500.0, verify_ms=100.0)
        assert a.draft_verify_ratio > 1.0  # draft now dominates

    def test_record_timing_standalone(self):
        """record_timing can be called directly."""
        a = AdaptiveDraftTokens(initial=16)
        a.record_timing(10.0, 100.0)
        a.record_timing(20.0, 200.0)
        assert len(a._timing_window) == 2
        assert abs(a.draft_verify_ratio - 0.1) < 0.01

    def test_extreme_ratio_capped(self):
        """Aggression factor is capped at 3.0 even with very small ratio."""
        a = AdaptiveDraftTokens(initial=16)
        # Fill exactly half the window so only ONE adjustment fires on the last round
        for _ in range(_AUTO_WINDOW // 2):
            a.record_round(drafted=16, accepted=16,  # 100%
                           draft_ms=0.1, verify_ms=10000.0)
        # ratio=0.00001 -> aggression capped at 3.0 -> step=int(4*3)=12
        assert a.current == 16 + 12  # 28 after first adjustment
        assert a.adjustments == 1

    def test_cost_aware_reaches_max_faster_when_verify_dominates(self):
        """With verify dominating and high acceptance, should reach _AUTO_MAX
        in fewer rounds than baseline."""
        a = AdaptiveDraftTokens(initial=16)
        b = AdaptiveDraftTokens(initial=16)
        rounds_a = 0
        rounds_b = 0
        for i in range(_AUTO_WINDOW * 20):
            if a.current < _AUTO_MAX:
                a.record_round(drafted=a.current, accepted=a.current,
                               draft_ms=5.0, verify_ms=500.0)
                rounds_a = i + 1
            if b.current < _AUTO_MAX:
                b.record_round(drafted=b.current, accepted=b.current)
                rounds_b = i + 1
        assert a.current == _AUTO_MAX
        assert b.current == _AUTO_MAX
        assert rounds_a < rounds_b  # cost-aware gets there faster

    def test_middle_acceptance_no_change_with_timing(self):
        """Moderate acceptance (between thresholds) causes no change,
        even with timing data."""
        a = AdaptiveDraftTokens(initial=16)
        for _ in range(_AUTO_WINDOW):
            a.record_round(drafted=16, accepted=10,  # 62.5%
                           draft_ms=5.0, verify_ms=500.0)
        assert a.current == 16
        assert a.adjustments == 0

    def test_zero_verify_ms_excluded_from_ratio(self):
        """Rounds with verify_ms=0 (consensus skip) should not corrupt ratio."""
        a = AdaptiveDraftTokens(initial=16)
        # Some rounds with real timing
        a.record_round(drafted=10, accepted=8, draft_ms=10.0, verify_ms=100.0)
        a.record_round(drafted=10, accepted=8, draft_ms=20.0, verify_ms=200.0)
        # Consensus round — no verify
        a.record_round(drafted=10, accepted=10, draft_ms=5.0, verify_ms=0.0)
        # ratio should only consider the two real rounds: 30/300 = 0.1
        assert abs(a.draft_verify_ratio - 0.1) < 0.01
