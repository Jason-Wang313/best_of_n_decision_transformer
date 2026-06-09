import math

import numpy as np

from decision_transformer_best_of_n.best_of_n import (
    best_of_n_monte_carlo,
    exact_best_of_n_expectation,
    exact_finite_pool_best_of_n_without_replacement,
)


def test_exact_law_handles_ties():
    scores = [1.0, 2.0, 2.0]
    utilities = [0.0, 10.0, 20.0]

    n1 = exact_best_of_n_expectation(scores, utilities, 1)
    n2 = exact_best_of_n_expectation(scores, utilities, 2)

    assert math.isclose(n1["expected_utility"], 10.0)
    assert math.isclose(n2["expected_utility"], (8.0 / 9.0) * 15.0)
    assert math.isclose(n2["expected_score"], (1.0 / 9.0) * 1.0 + (8.0 / 9.0) * 2.0)


def test_exact_law_matches_monte_carlo():
    scores = [0.0, 1.0, 1.0, 2.0]
    utilities = [1.0, 2.0, 4.0, 8.0]
    exact = exact_best_of_n_expectation(scores, utilities, 4)
    mc = best_of_n_monte_carlo(scores, utilities, 4, n_trials=40_000, seed=7)

    assert abs(exact["expected_utility"] - mc["expected_utility"]) < 0.08


def test_finite_pool_without_replacement():
    scores = [1.0, 2.0, 2.0]
    utilities = [0.0, 10.0, 20.0]

    result = exact_finite_pool_best_of_n_without_replacement(scores, utilities, 2)

    # Pairs: (0,1)->10, (0,2)->20, (1,2)->15.
    assert math.isclose(result["expected_utility"], 15.0)


def test_weighted_exact_law():
    result = exact_best_of_n_expectation(
        scores=[0.0, 1.0],
        utilities=[0.0, 10.0],
        n=2,
        probs=[0.75, 0.25],
    )

    assert np.isclose(result["expected_utility"], (1.0 - 0.75**2) * 10.0)

