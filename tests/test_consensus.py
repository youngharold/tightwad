"""Tests for multi-drafter consensus verification logic."""

import pytest

from tightwad.speculation import (
    ConsensusMode,
    ConsensusResult,
    DraftToken,
    verify_consensus,
)


def _draft(token_id: int, logprob: float = -0.1, text: str = "") -> DraftToken:
    return DraftToken(token_id=token_id, logprob=logprob, text=text or f"t{token_id}")


class TestStrictConsensus:
    def test_strict_all_agree(self):
        """3 drafters agree on 5 tokens -> all accepted, no target needed."""
        tokens = [_draft(i) for i in range(5)]
        outputs = [list(tokens), list(tokens), list(tokens)]

        result = verify_consensus(outputs, ConsensusMode.STRICT)

        assert result.accepted_count == 5
        assert result.needs_target_verification is False
        assert result.disagreed_at is None
        assert len(result.agreement_rates) == 5
        assert all(rate == 1.0 for rate in result.agreement_rates)

    def test_strict_one_disagrees(self):
        """Disagreement at position 3 -> accepts 0-2, needs target."""
        base = [_draft(i) for i in range(5)]
        different = [_draft(i) for i in range(5)]
        different[3] = _draft(99)

        outputs = [list(base), list(base), list(different)]

        result = verify_consensus(outputs, ConsensusMode.STRICT)

        assert result.accepted_count == 3
        assert result.needs_target_verification is True
        assert result.disagreed_at == 3
        # Positions 0-2 had full agreement
        assert all(rate == 1.0 for rate in result.agreement_rates[:3])


class TestMajorityConsensus:
    def test_majority_two_of_three(self):
        """2/3 agree at each position -> majority tokens accepted."""
        a = [_draft(10), _draft(20), _draft(30), _draft(40)]
        b = [_draft(10), _draft(20), _draft(30), _draft(40)]
        c = [_draft(99), _draft(98), _draft(97), _draft(96)]

        outputs = [a, b, c]
        result = verify_consensus(outputs, ConsensusMode.MAJORITY)

        assert result.accepted_count == 4
        assert result.needs_target_verification is False
        assert result.disagreed_at is None
        # 2/3 = ~0.667 at each position
        for rate in result.agreement_rates:
            assert abs(rate - 2 / 3) < 1e-9

    def test_majority_all_disagree(self):
        """All different -> needs target at position 0."""
        a = [_draft(1)]
        b = [_draft(2)]
        c = [_draft(3)]

        result = verify_consensus([a, b, c], ConsensusMode.MAJORITY)

        assert result.accepted_count == 0
        assert result.needs_target_verification is True
        assert result.disagreed_at == 0
        # Each token got 1/3 — no majority
        assert len(result.agreement_rates) == 1
        assert abs(result.agreement_rates[0] - 1 / 3) < 1e-9


class TestAnyDisagreeConsensus:
    def test_any_disagree_all_agree(self):
        """Unanimous -> accepted without target."""
        tokens = [_draft(10), _draft(20), _draft(30)]
        outputs = [list(tokens), list(tokens)]

        result = verify_consensus(outputs, ConsensusMode.ANY_DISAGREE)

        assert result.accepted_count == 3
        assert result.needs_target_verification is False
        assert result.disagreed_at is None

    def test_any_disagree_partial(self):
        """Agree for 4 tokens then disagree -> 4 accepted + needs target."""
        base = [_draft(i) for i in range(6)]
        variant = [_draft(i) for i in range(6)]
        variant[4] = _draft(999)

        outputs = [list(base), list(variant)]

        result = verify_consensus(outputs, ConsensusMode.ANY_DISAGREE)

        assert result.accepted_count == 4
        assert result.needs_target_verification is True
        assert result.disagreed_at == 4


class TestEdgeCases:
    def test_empty_inputs(self):
        """Empty drafter list returns empty result."""
        result = verify_consensus([], ConsensusMode.STRICT)

        assert result.accepted_count == 0
        assert result.needs_target_verification is False
        assert result.disagreed_at is None
        assert result.agreement_rates == []

    def test_single_drafter(self):
        """Single drafter in strict mode accepts all its tokens."""
        tokens = [_draft(1), _draft(2), _draft(3)]
        outputs = [tokens]

        result = verify_consensus(outputs, ConsensusMode.STRICT)

        assert result.accepted_count == 3
        assert result.needs_target_verification is False
        assert result.disagreed_at is None
        # 1/1 = 1.0 at every position
        assert all(rate == 1.0 for rate in result.agreement_rates)

    def test_different_length_outputs(self):
        """Drafters return different counts -> uses shortest."""
        short = [_draft(1), _draft(2)]
        long = [_draft(1), _draft(2), _draft(3), _draft(4)]

        outputs = [short, long]

        result = verify_consensus(outputs, ConsensusMode.STRICT)

        assert result.accepted_count == 2
        assert result.needs_target_verification is False
        assert len(result.agreement_rates) == 2

    def test_agreement_rates(self):
        """Verify per-position rates are correct."""
        # 4 drafters, position 0: all agree (4/4), position 1: 3/4, position 2: 2/4
        d1 = [_draft(10), _draft(20), _draft(30)]
        d2 = [_draft(10), _draft(20), _draft(31)]
        d3 = [_draft(10), _draft(21), _draft(30)]
        d4 = [_draft(10), _draft(20), _draft(32)]

        # Use MAJORITY so we can see rates for all positions before stopping
        result = verify_consensus([d1, d2, d3, d4], ConsensusMode.MAJORITY)

        # Position 0: 4/4 = 1.0 (all agree on 10)
        assert result.agreement_rates[0] == 1.0
        # Position 1: 3/4 = 0.75 (d1, d2, d4 agree on 20, d3 has 21)
        assert result.agreement_rates[1] == 0.75
        # Position 2: 2/4 = 0.5 (d1 and d3 agree on 30, d2=31, d4=32)
        # 2/4 = 0.5, NOT > 50% so majority fails here
        assert result.accepted_count == 2
        assert result.disagreed_at == 2
        assert result.needs_target_verification is True


class TestConsensusResult:
    def test_mean_agreement_rate(self):
        result = ConsensusResult(
            accepted_tokens=[],
            accepted_count=0,
            needs_target_verification=False,
            disagreed_at=None,
            agreement_rates=[1.0, 0.75, 0.5],
        )
        assert abs(result.mean_agreement_rate - 0.75) < 1e-9

    def test_mean_agreement_rate_empty(self):
        result = ConsensusResult(
            accepted_tokens=[],
            accepted_count=0,
            needs_target_verification=False,
            disagreed_at=None,
            agreement_rates=[],
        )
        assert result.mean_agreement_rate == 0.0
