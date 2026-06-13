"""A tiny Decision Transformer-style sequence policy.

This is intentionally not a benchmark-scale Transformer. It is a learned
sequence policy that conditions each action on state, previous action, time,
and desired return-to-go, which is the DT-style ingredient needed for the
controlled return-support audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Rollout, TrajectoryBatch, proxy_reward, real_reward


@dataclass
class TinyDecisionTransformer:
    horizon: int
    safe_action: float = 0.72
    action_clip: float = 1.65
    ridge: float = 1e-3
    coef_: np.ndarray | None = None
    residual_std_: float = 0.08
    feature_mean_: np.ndarray | None = None
    feature_std_: np.ndarray | None = None

    def _raw_features(
        self,
        state: float,
        previous_action: float,
        return_to_go: float,
        t: int,
    ) -> np.ndarray:
        remaining = max(self.horizon - t, 1)
        time = t / max(self.horizon - 1, 1)
        rate = return_to_go / remaining
        return np.array(
            [
                1.0,
                state,
                previous_action,
                return_to_go,
                rate,
                time,
                state * 0.05 * return_to_go,
                rate * rate,
                previous_action * rate,
            ],
            dtype=float,
        )

    def _features(self, raw: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_std_ is None:
            return raw
        scaled = raw.copy()
        scaled[1:] = (scaled[1:] - self.feature_mean_[1:]) / self.feature_std_[1:]
        return scaled

    def _raw_features_batch(
        self,
        states: np.ndarray,
        previous_actions: np.ndarray,
        returns_to_go: np.ndarray,
        t: int,
    ) -> np.ndarray:
        remaining = max(self.horizon - t, 1)
        time = t / max(self.horizon - 1, 1)
        rates = returns_to_go / remaining
        return np.column_stack(
            [
                np.ones_like(states),
                states,
                previous_actions,
                returns_to_go,
                rates,
                np.full_like(states, time),
                states * 0.05 * returns_to_go,
                rates * rates,
                previous_actions * rates,
            ]
        )

    def _features_batch(self, raw: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_std_ is None:
            return raw
        scaled = raw.copy()
        scaled[:, 1:] = (scaled[:, 1:] - self.feature_mean_[1:]) / self.feature_std_[1:]
        return scaled

    def fit(self, batch: TrajectoryBatch) -> "TinyDecisionTransformer":
        xs: list[np.ndarray] = []
        ys: list[float] = []
        for i in range(batch.actions.shape[0]):
            previous_action = 0.0
            for t in range(batch.horizon):
                xs.append(
                    self._raw_features(
                        state=float(batch.states[i, t]),
                        previous_action=previous_action,
                        return_to_go=float(batch.returns_to_go[i, t]),
                        t=t,
                    )
                )
                ys.append(float(batch.actions[i, t]))
                previous_action = float(batch.actions[i, t])

        x = np.vstack(xs)
        y = np.asarray(ys)
        self.feature_mean_ = np.mean(x, axis=0)
        self.feature_std_ = np.std(x, axis=0) + 1e-8
        self.feature_mean_[0] = 0.0
        self.feature_std_[0] = 1.0
        x_scaled = np.vstack([self._features(row) for row in x])

        reg = self.ridge * np.eye(x_scaled.shape[1])
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.solve(x_scaled.T @ x_scaled + reg, x_scaled.T @ y)
        residuals = y - x_scaled @ self.coef_
        self.residual_std_ = float(max(np.std(residuals), 0.035))
        return self

    def predict_action_mean(
        self,
        state: float,
        previous_action: float,
        return_to_go: float,
        t: int,
    ) -> float:
        if self.coef_ is None:
            raise RuntimeError("TinyDecisionTransformer.fit must be called before prediction.")
        raw = self._raw_features(state, previous_action, return_to_go, t)
        return float(self._features(raw) @ self.coef_)

    def predict_action_mean_batch(
        self,
        states: np.ndarray,
        previous_actions: np.ndarray,
        returns_to_go: np.ndarray,
        t: int,
    ) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("TinyDecisionTransformer.fit must be called before prediction.")
        raw = self._raw_features_batch(states, previous_actions, returns_to_go, t)
        return self._features_batch(raw) @ self.coef_

    def rollout(
        self,
        target_return: float,
        rng: np.random.Generator,
        noise_scale: float = 0.32,
    ) -> Rollout:
        states = np.zeros(self.horizon + 1, dtype=float)
        actions = np.zeros(self.horizon, dtype=float)
        proxy_rewards = np.zeros(self.horizon, dtype=float)
        real_rewards = np.zeros(self.horizon, dtype=float)
        previous_action = 0.0
        remaining_return = float(target_return)

        for t in range(self.horizon):
            mean = self.predict_action_mean(
                state=float(states[t]),
                previous_action=previous_action,
                return_to_go=remaining_return,
                t=t,
            )
            exploration = rng.normal(0.0, noise_scale * self.residual_std_ + 0.045)
            action = float(np.clip(mean + exploration, -0.35, self.action_clip))
            actions[t] = action
            proxy_rewards[t] = proxy_reward(action)
            real_rewards[t] = real_reward(action, safe_action=self.safe_action)
            states[t + 1] = states[t] + action + rng.normal(0.0, 0.015)
            remaining_return -= proxy_rewards[t]
            previous_action = action

        return Rollout(
            states=states,
            actions=actions,
            proxy_rewards=proxy_rewards,
            real_rewards=real_rewards,
            target_return=float(target_return),
        )

    def rollout_batch(
        self,
        target_return: float,
        n: int,
        rng: np.random.Generator,
        noise_scale: float = 0.32,
    ) -> list[Rollout]:
        states = np.zeros((n, self.horizon + 1), dtype=float)
        actions = np.zeros((n, self.horizon), dtype=float)
        proxy_rewards = np.zeros((n, self.horizon), dtype=float)
        real_rewards = np.zeros((n, self.horizon), dtype=float)
        previous_actions = np.zeros(n, dtype=float)
        remaining_returns = np.full(n, float(target_return), dtype=float)
        sigma = noise_scale * self.residual_std_ + 0.045

        for t in range(self.horizon):
            means = self.predict_action_mean_batch(
                states=states[:, t],
                previous_actions=previous_actions,
                returns_to_go=remaining_returns,
                t=t,
            )
            action = np.clip(means + rng.normal(0.0, sigma, size=n), -0.35, self.action_clip)
            actions[:, t] = action
            proxy_rewards[:, t] = proxy_reward(action)
            real_rewards[:, t] = real_reward(action, safe_action=self.safe_action)
            states[:, t + 1] = states[:, t] + action + rng.normal(0.0, 0.015, size=n)
            remaining_returns -= proxy_rewards[:, t]
            previous_actions = action

        return [
            Rollout(
                states=states[i].copy(),
                actions=actions[i].copy(),
                proxy_rewards=proxy_rewards[i].copy(),
                real_rewards=real_rewards[i].copy(),
                target_return=float(target_return),
            )
            for i in range(n)
        ]
