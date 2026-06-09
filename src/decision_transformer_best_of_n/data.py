"""Synthetic offline trajectories with limited high-return support.

The environment deliberately separates a proxy/internal return from real
utility. Offline behavior mostly lives in the safe action region. A
return-conditioned model can extrapolate toward larger actions for out-of-
support target returns; this can raise proxy score while real utility falls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrajectoryBatch:
    states: np.ndarray
    actions: np.ndarray
    proxy_rewards: np.ndarray
    real_rewards: np.ndarray
    returns_to_go: np.ndarray
    proxy_returns: np.ndarray
    real_returns: np.ndarray
    requested_returns: np.ndarray
    safe_action: float

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])


@dataclass(frozen=True)
class Rollout:
    states: np.ndarray
    actions: np.ndarray
    proxy_rewards: np.ndarray
    real_rewards: np.ndarray
    target_return: float

    @property
    def proxy_return(self) -> float:
        return float(np.sum(self.proxy_rewards))

    @property
    def real_return(self) -> float:
        return float(np.sum(self.real_rewards))

    @property
    def risk(self) -> float:
        excess = np.maximum(np.abs(self.actions) - 0.72, 0.0)
        return float(np.mean(excess**2))


def proxy_reward(action: float | np.ndarray) -> float | np.ndarray:
    """Internal score reward: progress is attractive and risk is underpriced."""

    return np.asarray(action) - 0.025 * np.asarray(action) ** 2


def real_reward(action: float | np.ndarray, safe_action: float = 0.72) -> float | np.ndarray:
    """True utility: high actions carry a convex off-support penalty."""

    action_arr = np.asarray(action)
    risk = np.maximum(np.abs(action_arr) - safe_action, 0.0)
    return proxy_reward(action_arr) - 10.0 * risk**2 - 0.18 * np.maximum(action_arr - 1.20, 0.0)


def _sample_requested_returns(rng: np.random.Generator, n: int) -> np.ndarray:
    body = 1.4 + 5.4 * rng.beta(2.5, 3.8, size=n)
    tail_mask = rng.random(n) < 0.08
    body[tail_mask] = rng.uniform(6.0, 7.25, size=int(np.sum(tail_mask)))
    return body


def generate_offline_dataset(
    n_trajectories: int = 600,
    horizon: int = 12,
    seed: int = 313,
    safe_action: float = 0.72,
) -> TrajectoryBatch:
    """Generate behavior trajectories with scarce high-return support."""

    rng = np.random.default_rng(seed)
    requested = _sample_requested_returns(rng, n_trajectories)
    states = np.zeros((n_trajectories, horizon + 1), dtype=float)
    actions = np.zeros((n_trajectories, horizon), dtype=float)
    proxy_rewards = np.zeros_like(actions)
    real_rewards = np.zeros_like(actions)

    for i, target in enumerate(requested):
        remaining_proxy = float(target)
        previous_action = 0.0
        for t in range(horizon):
            remaining_steps = horizon - t
            target_rate = remaining_proxy / remaining_steps
            behavior_noise = rng.normal(0.0, 0.075)
            smooth = 0.12 * previous_action
            action = 0.88 * target_rate + smooth + behavior_noise
            action = float(np.clip(action, -0.15, 0.92))
            # The behavior data contains a few near-edge examples but very few
            # truly unsafe actions, creating the intended support boundary.
            if rng.random() < 0.015:
                action = float(np.clip(action + rng.uniform(0.15, 0.28), -0.15, 0.98))

            actions[i, t] = action
            proxy_rewards[i, t] = proxy_reward(action)
            real_rewards[i, t] = real_reward(action, safe_action=safe_action)
            states[i, t + 1] = states[i, t] + action + rng.normal(0.0, 0.015)
            remaining_proxy -= proxy_rewards[i, t]
            previous_action = action

    returns_to_go = np.flip(np.cumsum(np.flip(proxy_rewards, axis=1), axis=1), axis=1)
    return TrajectoryBatch(
        states=states,
        actions=actions,
        proxy_rewards=proxy_rewards,
        real_rewards=real_rewards,
        returns_to_go=returns_to_go,
        proxy_returns=np.sum(proxy_rewards, axis=1),
        real_returns=np.sum(real_rewards, axis=1),
        requested_returns=requested,
        safe_action=safe_action,
    )


def rollout_from_actions(
    actions: Iterable[float],
    target_return: float,
    seed: int = 0,
    safe_action: float = 0.72,
) -> Rollout:
    """Roll out known actions through the simple 1D dynamics."""

    rng = np.random.default_rng(seed)
    actions_arr = np.asarray(list(actions), dtype=float)
    states = np.zeros(len(actions_arr) + 1, dtype=float)
    for t, action in enumerate(actions_arr):
        states[t + 1] = states[t] + action + rng.normal(0.0, 0.015)
    return Rollout(
        states=states,
        actions=actions_arr,
        proxy_rewards=proxy_reward(actions_arr),
        real_rewards=real_reward(actions_arr, safe_action=safe_action),
        target_return=float(target_return),
    )
