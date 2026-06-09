"""Finite-N Best-of-N laws and samplers."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

import numpy as np


def _normalize_probabilities(n: int, probs: Iterable[float] | None) -> np.ndarray:
    if probs is None:
        return np.full(n, 1.0 / n, dtype=float)
    p = np.asarray(list(probs), dtype=float)
    if p.shape != (n,):
        raise ValueError("probs must have the same length as scores.")
    total = float(np.sum(p))
    if total <= 0:
        raise ValueError("probs must have positive mass.")
    return p / total


def exact_best_of_n_expectation(
    scores: Iterable[float],
    utilities: Iterable[float],
    n: int,
    probs: Iterable[float] | None = None,
) -> dict[str, float]:
    """Tie-aware expectation for iid Best-of-N with replacement.

    Samples are drawn iid from a finite distribution. The chosen candidate has
    maximal score; ties at the maximal score are broken uniformly. Since all
    candidates in the winning score level are exchangeable under uniform
    tie-breaking, the conditional utility is the probability-weighted mean
    utility inside that score level.
    """

    if n < 1:
        raise ValueError("n must be at least 1.")
    s = np.asarray(list(scores), dtype=float)
    r = np.asarray(list(utilities), dtype=float)
    if s.shape != r.shape or s.ndim != 1 or len(s) == 0:
        raise ValueError("scores and utilities must be non-empty vectors of equal length.")
    p = _normalize_probabilities(len(s), probs)

    groups: dict[float, dict[str, float]] = defaultdict(lambda: {"prob": 0.0, "utility": 0.0})
    for score, utility, prob in zip(s, r, p):
        groups[float(score)]["prob"] += float(prob)
        groups[float(score)]["utility"] += float(prob) * float(utility)

    expected_utility = 0.0
    expected_score = 0.0
    cdf_prev = 0.0
    for score in sorted(groups):
        group_prob = groups[score]["prob"]
        cdf = cdf_prev + group_prob
        win_prob = cdf**n - cdf_prev**n
        group_utility = groups[score]["utility"] / group_prob
        expected_utility += win_prob * group_utility
        expected_score += win_prob * score
        cdf_prev = cdf

    return {
        "expected_utility": float(expected_utility),
        "expected_score": float(expected_score),
        "n": float(n),
    }


def exact_finite_pool_best_of_n_without_replacement(
    scores: Iterable[float],
    utilities: Iterable[float],
    n: int,
) -> dict[str, float]:
    """Exact tie-aware finite-pool utility for uniform samples without replacement."""

    s = np.asarray(list(scores), dtype=float)
    r = np.asarray(list(utilities), dtype=float)
    if s.shape != r.shape or s.ndim != 1 or len(s) == 0:
        raise ValueError("scores and utilities must be non-empty vectors of equal length.")
    if not 1 <= n <= len(s):
        raise ValueError("n must be between 1 and the finite pool size.")
    if len(s) > 26:
        raise ValueError("Exact finite-pool enumeration is intended for small test pools.")

    total_utility = 0.0
    total_score = 0.0
    count = 0
    for idxs in combinations(range(len(s)), n):
        sample_scores = s[list(idxs)]
        max_score = float(np.max(sample_scores))
        winners = [idx for idx in idxs if float(s[idx]) == max_score]
        total_utility += float(np.mean(r[winners]))
        total_score += max_score
        count += 1
    return {
        "expected_utility": total_utility / count,
        "expected_score": total_score / count,
        "n": float(n),
    }


def best_of_n_monte_carlo(
    scores: Iterable[float],
    utilities: Iterable[float],
    n: int,
    n_trials: int = 20_000,
    seed: int = 0,
    probs: Iterable[float] | None = None,
) -> dict[str, float]:
    """Monte Carlo validation sampler for iid Best-of-N with uniform tie-breaking."""

    s = np.asarray(list(scores), dtype=float)
    r = np.asarray(list(utilities), dtype=float)
    p = _normalize_probabilities(len(s), probs)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(s), size=(n_trials, n), replace=True, p=p)
    sample_scores = s[idxs]
    max_scores = np.max(sample_scores, axis=1, keepdims=True)
    tie_mask = sample_scores == max_scores
    tie_random = rng.random(size=tie_mask.shape)
    tie_random[~tie_mask] = -1.0
    chosen_positions = np.argmax(tie_random, axis=1)
    selected = idxs[np.arange(n_trials), chosen_positions]
    selected_scores = s[selected]
    selected_utilities = r[selected]
    return {
        "expected_utility": float(np.mean(selected_utilities)),
        "expected_score": float(np.mean(selected_scores)),
        "stderr_utility": float(np.std(selected_utilities) / np.sqrt(n_trials)),
        "n": float(n),
    }
