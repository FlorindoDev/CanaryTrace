"""Score the expected continuation from full-vocabulary model logits.

Each row predicts its corresponding target token given the prompt and the
preceding target tokens. For a causal model with logits over prompt + target,
take T rows starting at index P - 1, where P and T are their token lengths.
"""

from collections.abc import Iterator, Sequence
from math import exp, fsum, isfinite, isnan, log


def _target_rows(
    logits: Sequence[Sequence[float]], target_token_ids: Sequence[int]
) -> Iterator[tuple[Sequence[float], float, float]]:
    if len(logits) == 0 or len(logits) != len(target_token_ids):
        raise ValueError("Provide one logit row per target token and at least one target token.")

    vocabulary_size = len(logits[0])
    for scores, token_id in zip(logits, target_token_ids):
        if len(scores) != vocabulary_size:
            raise ValueError("Each logit row must cover the same vocabulary.")
        if type(token_id) is not int or not 0 <= token_id < vocabulary_size:
            raise ValueError("Each target token ID must index the vocabulary.")

        maximum = max(scores)
        if not isfinite(maximum) or any(isnan(score) for score in scores):
            raise ValueError("Each row must have a finite maximum and no NaN logits.")
        yield scores, scores[token_id], maximum


def likelihood(logits: Sequence[Sequence[float]], target_token_ids: Sequence[int]) -> float:
    """Return the product of the conditional probabilities of the target tokens."""
    log_probability = 0.0
    for scores, target_score, maximum in _target_rows(logits, target_token_ids):
        normalizer = fsum(exp(score - maximum) for score in scores)
        log_probability += (target_score - maximum) - log(normalizer)
    return exp(log_probability)


def token_rank(logits: Sequence[Sequence[float]], target_token_ids: Sequence[int]) -> float:
    """Return the mean rank, counting from 1 and sharing ranks for equal scores."""
    ranks = (
        1 + sum(score > target_score for score in scores)
        for scores, target_score, _ in _target_rows(logits, target_token_ids)
    )
    return sum(ranks) / len(target_token_ids)
