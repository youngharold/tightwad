"""Core speculative decoding verification logic (pure, no I/O)."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class DraftToken:
    token_id: int
    logprob: float  # log probability from draft model
    text: str = ""


@dataclass
class TargetLogprob:
    token_id: int  # target's top token at this position
    logprob: float  # log probability of that top token
    # logprob of the draft token according to the target model
    draft_token_logprob: float | None = None


@dataclass
class VerificationResult:
    accepted_tokens: list[DraftToken]
    bonus_token: DraftToken | None  # extra token from target if all accepted
    accepted_count: int = 0
    rejected_at: int | None = None  # index of first rejection, None if all accepted
    resample_token: DraftToken | None = None  # target's token at rejection point

    @property
    def total_tokens(self) -> int:
        n = self.accepted_count
        if self.bonus_token is not None:
            n += 1
        if self.resample_token is not None:
            n += 1
        return n


def verify_greedy(
    draft_tokens: list[DraftToken],
    target_logprobs: list[TargetLogprob],
) -> VerificationResult:
    """Verify draft tokens in greedy (temperature=0) mode.

    Accept if target's argmax matches draft token, reject otherwise.
    """
    accepted: list[DraftToken] = []

    for i, (draft, target) in enumerate(zip(draft_tokens, target_logprobs)):
        if draft.token_id == target.token_id:
            accepted.append(draft)
        else:
            # Reject: use target's token at this position
            resample = DraftToken(
                token_id=target.token_id,
                logprob=target.logprob,
            )
            return VerificationResult(
                accepted_tokens=accepted,
                bonus_token=None,
                accepted_count=len(accepted),
                rejected_at=i,
                resample_token=resample,
            )

    # All accepted — bonus token comes from target at position after last draft
    bonus = None
    if len(target_logprobs) > len(draft_tokens):
        tp = target_logprobs[len(draft_tokens)]
        bonus = DraftToken(token_id=tp.token_id, logprob=tp.logprob)

    return VerificationResult(
        accepted_tokens=accepted,
        bonus_token=bonus,
        accepted_count=len(accepted),
        rejected_at=None,
    )


def verify_stochastic(
    draft_tokens: list[DraftToken],
    target_logprobs: list[TargetLogprob],
) -> VerificationResult:
    """Verify draft tokens with standard rejection sampling.

    Accept token i with probability min(1, P_target(token_i) / P_draft(token_i)).
    """
    accepted: list[DraftToken] = []

    for i, (draft, target) in enumerate(zip(draft_tokens, target_logprobs)):
        # Get P_target for the draft token
        if target.draft_token_logprob is not None:
            p_target = math.exp(target.draft_token_logprob)
        elif draft.token_id == target.token_id:
            p_target = math.exp(target.logprob)
        else:
            # Draft token not in target's top — treat as very low probability
            p_target = 0.0

        p_draft = math.exp(draft.logprob)
        if p_draft <= 0:
            # Avoid division by zero; accept if target also agrees
            accept = draft.token_id == target.token_id
        else:
            accept_prob = min(1.0, p_target / p_draft)
            accept = random.random() < accept_prob

        if accept:
            accepted.append(draft)
        else:
            resample = DraftToken(
                token_id=target.token_id,
                logprob=target.logprob,
            )
            return VerificationResult(
                accepted_tokens=accepted,
                bonus_token=None,
                accepted_count=len(accepted),
                rejected_at=i,
                resample_token=resample,
            )

    # All accepted
    bonus = None
    if len(target_logprobs) > len(draft_tokens):
        tp = target_logprobs[len(draft_tokens)]
        bonus = DraftToken(token_id=tp.token_id, logprob=tp.logprob)

    return VerificationResult(
        accepted_tokens=accepted,
        bonus_token=bonus,
        accepted_count=len(accepted),
        rejected_at=None,
    )


def verify_draft_tokens(
    draft_tokens: list[DraftToken],
    target_logprobs: list[TargetLogprob],
    temperature: float = 0.0,
) -> VerificationResult:
    """Verify draft tokens against target model logprobs.

    At temperature=0, uses greedy comparison (accept iff argmax matches).
    At temperature>0, uses standard rejection sampling.
    """
    if not draft_tokens:
        # Empty draft — nothing to verify
        bonus = None
        if target_logprobs:
            tp = target_logprobs[0]
            bonus = DraftToken(token_id=tp.token_id, logprob=tp.logprob)
        return VerificationResult(
            accepted_tokens=[],
            bonus_token=bonus,
            accepted_count=0,
            rejected_at=None,
        )

    if temperature == 0.0:
        return verify_greedy(draft_tokens, target_logprobs)
    else:
        return verify_stochastic(draft_tokens, target_logprobs)


# ---------------------------------------------------------------------------
# Multi-drafter consensus verification
# ---------------------------------------------------------------------------


class ConsensusMode(Enum):
    STRICT = "strict"        # Unanimous agreement required
    MAJORITY = "majority"    # >50% of drafters agree
    ANY_DISAGREE = "any_disagree"  # Accept unanimous, verify on any disagreement


@dataclass
class ConsensusResult:
    accepted_tokens: list[DraftToken]
    accepted_count: int
    needs_target_verification: bool
    disagreed_at: int | None
    agreement_rates: list[float]  # per-position agreement rate

    @property
    def mean_agreement_rate(self) -> float:
        if not self.agreement_rates:
            return 0.0
        return sum(self.agreement_rates) / len(self.agreement_rates)


def verify_consensus(
    drafter_outputs: list[list[DraftToken]],
    mode: ConsensusMode,
) -> ConsensusResult:
    """Compare multiple drafter outputs and determine consensus tokens.

    Pure function — no I/O.  Iterates token positions up to the shortest
    drafter output and applies the consensus policy from *mode* to decide
    which tokens can be accepted without target verification.

    Parameters
    ----------
    drafter_outputs:
        One list of ``DraftToken`` per drafter.
    mode:
        The consensus policy to apply at each position.

    Returns
    -------
    ConsensusResult
        Accepted tokens, per-position agreement rates, and whether the
        target model still needs to verify the remaining (or all) tokens.
    """
    if not drafter_outputs:
        return ConsensusResult(
            accepted_tokens=[],
            accepted_count=0,
            needs_target_verification=False,
            disagreed_at=None,
            agreement_rates=[],
        )

    n_drafters = len(drafter_outputs)
    min_len = min(len(out) for out in drafter_outputs)

    accepted: list[DraftToken] = []
    agreement_rates: list[float] = []
    disagreed_at: int | None = None

    for pos in range(min_len):
        # Count occurrences of each token_id at this position
        ids = [drafter_outputs[d][pos].token_id for d in range(n_drafters)]
        counter = Counter(ids)
        most_common_id, most_common_count = counter.most_common(1)[0]
        rate = most_common_count / n_drafters
        agreement_rates.append(rate)

        if mode is ConsensusMode.STRICT:
            if most_common_count == n_drafters:
                # All agree — pick token from first drafter at this position
                accepted.append(drafter_outputs[0][pos])
            else:
                disagreed_at = pos
                break

        elif mode is ConsensusMode.MAJORITY:
            if most_common_count > n_drafters / 2:
                # Majority agrees — pick token from a drafter that produced
                # the majority token_id.
                for d in range(n_drafters):
                    if drafter_outputs[d][pos].token_id == most_common_id:
                        accepted.append(drafter_outputs[d][pos])
                        break
            else:
                disagreed_at = pos
                break

        elif mode is ConsensusMode.ANY_DISAGREE:
            if most_common_count == n_drafters:
                accepted.append(drafter_outputs[0][pos])
            else:
                disagreed_at = pos
                break

    needs_target = disagreed_at is not None
    return ConsensusResult(
        accepted_tokens=accepted,
        accepted_count=len(accepted),
        needs_target_verification=needs_target,
        disagreed_at=disagreed_at,
        agreement_rates=agreement_rates,
    )
