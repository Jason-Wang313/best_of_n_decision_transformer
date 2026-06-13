"""Controlled return-support audits for return-conditioned sequence policies."""

from .tail_selection import (
    top_score_monte_carlo,
    exact_top_score_expectation,
    exact_finite_pool_top_score_without_replacement,
)
from .config import ExperimentConfig
from .data import generate_offline_dataset
from .model import TinyDecisionTransformer

__all__ = [
    "ExperimentConfig",
    "TinyDecisionTransformer",
    "top_score_monte_carlo",
    "exact_top_score_expectation",
    "exact_finite_pool_top_score_without_replacement",
    "generate_offline_dataset",
]
