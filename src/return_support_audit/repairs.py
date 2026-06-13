"""Support-aware repair diagnostics for Decision-Transformer prompt audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .data import Rollout, TrajectoryBatch

GateDecision = Literal[
    "allow_candidate_sweep",
    "lower_target_return",
    "collect_pilot_labels",
    "block_candidate_sweep",
]

GATE_DECISIONS: tuple[GateDecision, ...] = (
    "allow_candidate_sweep",
    "lower_target_return",
    "collect_pilot_labels",
    "block_candidate_sweep",
)


@dataclass(frozen=True)
class Candidate:
    rollout: Rollout
    behavior_logp: float

    @property
    def proxy(self) -> float:
        return self.rollout.proxy_return

    @property
    def real(self) -> float:
        return self.rollout.real_return

    @property
    def risk(self) -> float:
        excess = np.maximum(np.abs(self.rollout.actions) - 0.72, 0.0)
        return float(np.mean(excess**2))


@dataclass(frozen=True)
class SupportEstimator:
    q50: float
    q80: float
    q90: float
    q95: float
    q99: float
    max_return: float
    action_mean: float
    action_std: float
    action_p01: float
    action_p99: float
    trajectory_logp_q05: float
    trajectory_logp_q10: float

    @classmethod
    def fit(cls, batch: TrajectoryBatch) -> "SupportEstimator":
        flat_actions = batch.actions.reshape(-1)
        action_mean = float(np.mean(flat_actions))
        action_std = float(np.std(flat_actions) + 1e-6)
        logps = []
        for actions in batch.actions:
            z = (actions - action_mean) / action_std
            logps.append(float(np.mean(-0.5 * z**2 - np.log(action_std) - 0.5 * np.log(2 * np.pi))))
        return cls(
            q50=float(np.quantile(batch.proxy_returns, 0.50)),
            q80=float(np.quantile(batch.proxy_returns, 0.80)),
            q90=float(np.quantile(batch.proxy_returns, 0.90)),
            q95=float(np.quantile(batch.proxy_returns, 0.95)),
            q99=float(np.quantile(batch.proxy_returns, 0.99)),
            max_return=float(np.max(batch.proxy_returns)),
            action_mean=action_mean,
            action_std=action_std,
            action_p01=float(np.quantile(flat_actions, 0.01)),
            action_p99=float(np.quantile(flat_actions, 0.99)),
            trajectory_logp_q05=float(np.quantile(logps, 0.05)),
            trajectory_logp_q10=float(np.quantile(logps, 0.10)),
        )

    def support_gap(self, target_return: float, quantile: float = 0.95) -> float:
        boundary = self.boundary(quantile)
        return float(max(target_return - boundary, 0.0))

    def boundary(self, quantile: float = 0.95) -> float:
        if quantile <= 0.50:
            return self.q50
        if quantile <= 0.80:
            return self.q80
        if quantile <= 0.90:
            return self.q90
        if quantile <= 0.95:
            return self.q95
        if quantile <= 0.99:
            return self.q99
        return self.max_return

    def calibrate_target(self, target_return: float, quantile: float = 0.95) -> float:
        return float(min(target_return, self.boundary(quantile)))

    def action_log_likelihood(self, actions: Iterable[float]) -> float:
        actions_arr = np.asarray(list(actions), dtype=float)
        z = (actions_arr - self.action_mean) / self.action_std
        base = -0.5 * z**2 - np.log(self.action_std) - 0.5 * np.log(2 * np.pi)
        tail_penalty = 1.8 * np.maximum(actions_arr - self.action_p99, 0.0) ** 2
        return float(np.mean(base - tail_penalty))

    def action_log_likelihood_batch(self, actions: np.ndarray) -> np.ndarray:
        z = (actions - self.action_mean) / self.action_std
        base = -0.5 * z**2 - np.log(self.action_std) - 0.5 * np.log(2 * np.pi)
        tail_penalty = 1.8 * np.maximum(actions - self.action_p99, 0.0) ** 2
        return np.mean(base - tail_penalty, axis=1)

    def describe_target(self, target_return: float) -> dict[str, float]:
        return {
            "target_return": float(target_return),
            "support_gap_q95": self.support_gap(target_return, 0.95),
            "support_gap_q99": self.support_gap(target_return, 0.99),
            "q50": self.q50,
            "q90": self.q90,
            "q95": self.q95,
            "q99": self.q99,
            "max_return": self.max_return,
        }


def make_candidate(rollout: Rollout, support: SupportEstimator) -> Candidate:
    return Candidate(rollout=rollout, behavior_logp=support.action_log_likelihood(rollout.actions))


def make_candidates(rollouts: Iterable[Rollout], support: SupportEstimator) -> list[Candidate]:
    values = list(rollouts)
    if not values:
        return []
    action_matrix = np.vstack([rollout.actions for rollout in values])
    logps = support.action_log_likelihood_batch(action_matrix)
    return [
        Candidate(rollout=rollout, behavior_logp=float(logp))
        for rollout, logp in zip(values, logps)
    ]


def select_naive(candidates: Iterable[Candidate]) -> Candidate:
    return max(candidates, key=lambda c: (c.proxy, c.behavior_logp))


def select_behavior_likelihood_constrained(
    candidates: Iterable[Candidate],
    support: SupportEstimator,
    threshold: float | None = None,
) -> Candidate:
    values = list(candidates)
    if not values:
        raise ValueError("Need at least one candidate.")
    cutoff = support.trajectory_logp_q05 if threshold is None else threshold
    feasible = [candidate for candidate in values if candidate.behavior_logp >= cutoff]
    if feasible:
        return max(feasible, key=lambda c: (c.proxy, c.behavior_logp))
    return max(values, key=lambda c: (c.proxy - 0.55 * (cutoff - c.behavior_logp), c.behavior_logp))


def fit_pilot_real_utility_calibrator(pilot_candidates: Iterable[Candidate]) -> np.ndarray:
    values = list(pilot_candidates)
    if len(values) < 3:
        return np.array([0.0, 1.0, -1.0], dtype=float)
    x = np.array([[1.0, c.proxy, c.risk] for c in values], dtype=float)
    y = np.array([c.real for c in values], dtype=float)
    reg = 1e-4 * np.eye(x.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + reg, x.T @ y)
    if beta[2] > 0.0:
        beta[2] = -abs(beta[2])
    return beta


def select_pilot_calibrated(
    candidates: Iterable[Candidate],
    pilot_candidates: Iterable[Candidate],
) -> Candidate:
    values = list(candidates)
    beta = fit_pilot_real_utility_calibrator(pilot_candidates)

    def predicted_real(candidate: Candidate) -> float:
        return float(beta @ np.array([1.0, candidate.proxy, candidate.risk], dtype=float))

    return max(values, key=lambda c: (predicted_real(c), c.proxy))


def select_oracle_real(candidates: Iterable[Candidate]) -> Candidate:
    return max(candidates, key=lambda c: (c.real, c.proxy))


def deployment_gate(
    target_return: float,
    support: SupportEstimator,
    n: int,
    pilot_delta_real: float | None = None,
    behavior_logp: float | None = None,
) -> GateDecision:
    """Return exactly one large candidate-count deployment decision."""

    support_gap = support.support_gap(target_return, 0.95)
    likelihood_bad = (
        behavior_logp is not None and behavior_logp < support.trajectory_logp_q05 - 0.20
    )

    if support_gap <= 0.05 and not likelihood_bad:
        return "allow_candidate_sweep"
    if pilot_delta_real is not None and (pilot_delta_real < -0.20 or likelihood_bad):
        return "block_candidate_sweep"
    if support_gap <= 0.85:
        return "lower_target_return"
    if pilot_delta_real is None:
        return "collect_pilot_labels"
    if pilot_delta_real >= -0.05 and not likelihood_bad:
        return "lower_target_return"
    return "block_candidate_sweep"
