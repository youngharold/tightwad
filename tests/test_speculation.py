"""Tests for speculative decoding verification logic."""

import random

import pytest

from tightwad.speculation import (
    DraftToken,
    TargetLogprob,
    VerificationResult,
    verify_draft_tokens,
    verify_greedy,
)


def _draft(token_id: int, logprob: float = -0.1, text: str = "") -> DraftToken:
    return DraftToken(token_id=token_id, logprob=logprob, text=text or f"t{token_id}")


def _target(token_id: int, logprob: float = -0.1) -> TargetLogprob:
    return TargetLogprob(token_id=token_id, logprob=logprob)


class TestGreedyVerification:
    def test_all_accepted_with_bonus(self):
        draft = [_draft(1), _draft(2), _draft(3)]
        target = [_target(1), _target(2), _target(3), _target(4)]  # bonus at pos 3

        result = verify_draft_tokens(draft, target, temperature=0.0)

        assert result.accepted_count == 3
        assert result.rejected_at is None
        assert result.bonus_token is not None
        assert result.bonus_token.token_id == 4
        assert result.total_tokens == 4  # 3 accepted + 1 bonus

    def test_all_accepted_no_bonus(self):
        draft = [_draft(1), _draft(2), _draft(3)]
        target = [_target(1), _target(2), _target(3)]

        result = verify_draft_tokens(draft, target, temperature=0.0)

        assert result.accepted_count == 3
        assert result.bonus_token is None
        assert result.total_tokens == 3

    def test_first_token_rejected(self):
        draft = [_draft(1), _draft(2), _draft(3)]
        target = [_target(99), _target(2), _target(3)]

        result = verify_draft_tokens(draft, target, temperature=0.0)

        assert result.accepted_count == 0
        assert result.rejected_at == 0
        assert result.resample_token is not None
        assert result.resample_token.token_id == 99
        assert result.total_tokens == 1  # just the resample token

    def test_partial_acceptance(self):
        # Accept 3 of 8, reject at position 3
        draft = [_draft(i) for i in range(8)]
        target_ids = [0, 1, 2, 99, 4, 5, 6, 7]
        target = [_target(tid) for tid in target_ids]

        result = verify_draft_tokens(draft, target, temperature=0.0)

        assert result.accepted_count == 3
        assert result.rejected_at == 3
        assert result.resample_token.token_id == 99

    def test_empty_draft(self):
        result = verify_draft_tokens([], [_target(42)], temperature=0.0)

        assert result.accepted_count == 0
        assert result.rejected_at is None
        assert result.bonus_token is not None
        assert result.bonus_token.token_id == 42

    def test_empty_draft_no_target(self):
        result = verify_draft_tokens([], [], temperature=0.0)

        assert result.accepted_count == 0
        assert result.bonus_token is None


class TestStochasticVerification:
    def test_matching_tokens_always_accepted(self):
        """When draft and target agree, acceptance prob = 1.0."""
        random.seed(42)
        draft = [_draft(1, logprob=-0.5), _draft(2, logprob=-0.5)]
        target = [_target(1, logprob=-0.5), _target(2, logprob=-0.5)]

        # Same tokens, same probs → always accept
        result = verify_draft_tokens(draft, target, temperature=1.0)
        assert result.accepted_count == 2

    def test_low_prob_draft_likely_rejected(self):
        """If draft assigns high prob but target assigns low, likely reject."""
        random.seed(0)
        # Draft thinks token 1 is very likely (logprob ~ 0, P ~ 1)
        # Target thinks token 1 is very unlikely — we use draft_token_logprob
        draft = [_draft(1, logprob=-0.01)]
        target = [TargetLogprob(
            token_id=99,  # target prefers 99
            logprob=-0.01,
            draft_token_logprob=-10.0,  # target assigns ~0.00005 to token 1
        )]

        result = verify_draft_tokens(draft, target, temperature=1.0)
        # Very likely to reject since P_target/P_draft ≈ 0.00005
        assert result.rejected_at == 0

    def test_bonus_token_on_full_accept(self):
        random.seed(42)
        draft = [_draft(1, logprob=-1.0)]
        target = [
            TargetLogprob(token_id=1, logprob=-1.0),
            TargetLogprob(token_id=5, logprob=-0.5),  # bonus
        ]

        result = verify_draft_tokens(draft, target, temperature=1.0)
        assert result.accepted_count == 1
        assert result.bonus_token is not None
        assert result.bonus_token.token_id == 5


class TestVerificationResult:
    def test_total_tokens_all_accepted_bonus(self):
        r = VerificationResult(
            accepted_tokens=[_draft(1), _draft(2)],
            bonus_token=_draft(3),
            accepted_count=2,
        )
        assert r.total_tokens == 3

    def test_total_tokens_rejected_with_resample(self):
        r = VerificationResult(
            accepted_tokens=[_draft(1)],
            bonus_token=None,
            accepted_count=1,
            rejected_at=1,
            resample_token=_draft(99),
        )
        assert r.total_tokens == 2  # 1 accepted + 1 resample

    def test_total_tokens_empty(self):
        r = VerificationResult(
            accepted_tokens=[],
            bonus_token=None,
            accepted_count=0,
        )
        assert r.total_tokens == 0
