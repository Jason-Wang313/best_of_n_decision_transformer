"""CPU-light CartPole-v1 benchmark for return-support selection audits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from math import cos, pi, sin
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np


GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02
X_THRESHOLD = 2.4
THETA_THRESHOLD_RADIANS = 12.0 * 2.0 * pi / 360.0
MAX_STEPS = 500
LOOKAHEAD = 22
MAX_N = 64


POLICY_KINDS = (
    "expert",
    "angle",
    "smooth",
    "short_exploit_right",
    "short_exploit_left",
    "random",
)
POLICY_PROBS = np.asarray((0.18, 0.22, 0.22, 0.18, 0.16, 0.04), dtype=float)


@dataclass(frozen=True)
class CartPoleCandidate:
    policy_kind: str
    episode_return: int
    proxy_score: float
    behavior_logp: float
    early_switch_rate: float
    late_action_imbalance: float
    max_abs_x: float
    max_abs_theta: float


def _step(state: tuple[float, float, float, float], action: int) -> tuple[tuple[float, float, float, float], bool]:
    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if int(action) == 1 else -FORCE_MAG
    costheta = cos(theta)
    sintheta = sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta**2 / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * thetaacc
    terminated = x < -X_THRESHOLD or x > X_THRESHOLD or theta < -THETA_THRESHOLD_RADIANS or theta > THETA_THRESHOLD_RADIANS
    return (x, x_dot, theta, theta_dot), bool(terminated)


def _expert_action(state: tuple[float, float, float, float]) -> int:
    x, x_dot, theta, theta_dot = state
    return int(theta + 0.25 * theta_dot + 0.01 * x + 0.02 * x_dot > 0.0)


def _policy_action(
    state: tuple[float, float, float, float],
    kind: str,
    rng: np.random.Generator,
    step_idx: int,
) -> int:
    x, _x_dot, theta, theta_dot = state
    if kind == "expert":
        action = _expert_action(state)
        return 1 - action if rng.random() < 0.02 else action
    if kind == "smooth":
        action = int(theta + 0.30 * theta_dot > 0.0)
        return 1 - action if rng.random() < 0.015 else action
    if kind == "angle":
        action = int(theta + 0.45 * theta_dot > 0.0)
        return 1 - action if rng.random() < 0.04 else action
    if kind == "short_exploit_right":
        return _expert_action(state) if step_idx < LOOKAHEAD else (1 if rng.random() < 0.95 else _expert_action(state))
    if kind == "short_exploit_left":
        return _expert_action(state) if step_idx < LOOKAHEAD else (0 if rng.random() < 0.95 else _expert_action(state))
    if kind == "random":
        return int(rng.random() < 0.5)
    raise ValueError(f"unknown policy kind {kind}")


def _rollout(initial_state: np.ndarray, policy_seed: int, kind: str) -> CartPoleCandidate:
    state = tuple(float(x) for x in initial_state)
    rng = np.random.default_rng(policy_seed)
    stability_values: list[float] = []
    actions: list[int] = []
    xs: list[float] = []
    thetas: list[float] = []

    for step_idx in range(MAX_STEPS):
        x, _x_dot, theta, theta_dot = state
        xs.append(x)
        thetas.append(theta)
        stability_values.append(1.0 - 8.0 * abs(theta) - 0.35 * abs(theta_dot) - 0.01 * abs(x))
        action = _policy_action(state, kind, rng, step_idx)
        actions.append(action)
        state, terminated = _step(state, action)
        if terminated:
            break

    episode_return = len(stability_values)
    early_actions = actions[: min(LOOKAHEAD, len(actions))]
    early_switch = float(np.mean(np.diff(early_actions) != 0)) if len(early_actions) > 2 else 0.0
    late_actions = actions[LOOKAHEAD:]
    late_imbalance = float(abs(np.mean(late_actions) - 0.5) * 2.0) if late_actions else 0.0
    early = stability_values[: min(LOOKAHEAD, len(stability_values))]

    # This is the intentionally attackable learned-return proxy: it extrapolates
    # from short-horizon pole stability and over-rewards later one-sided action
    # pressure, while the real CartPole-v1 return still comes from full rollout.
    proxy_score = float(np.mean(early) + 0.04 * np.percentile(early, 90) - 0.01 * early_switch + 0.075 * late_imbalance)
    behavior_logp = float(-late_imbalance - 0.20 * early_switch)
    return CartPoleCandidate(
        policy_kind=kind,
        episode_return=int(episode_return),
        proxy_score=proxy_score,
        behavior_logp=behavior_logp,
        early_switch_rate=early_switch,
        late_action_imbalance=late_imbalance,
        max_abs_x=float(max(abs(x) for x in xs)),
        max_abs_theta=float(max(abs(theta) for theta in thetas)),
    )


def _candidate_pool(trial: int, seed: int, n: int = MAX_N) -> list[CartPoleCandidate]:
    rng = np.random.default_rng(seed + trial)
    initial_state = np.random.default_rng(seed * 17 + trial).uniform(-0.05, 0.05, size=4)
    pool = []
    for idx in range(int(n)):
        kind = str(rng.choice(POLICY_KINDS, p=POLICY_PROBS))
        pool.append(_rollout(initial_state, seed * 1000 + trial * 100 + idx, kind))
    return pool


def _fixed_rollout(trial: int, seed: int, kind: str) -> CartPoleCandidate:
    initial_state = np.random.default_rng(seed * 17 + trial).uniform(-0.05, 0.05, size=4)
    return _rollout(initial_state, seed * 2000 + trial, kind)


def _select(selector: str, pool: list[CartPoleCandidate], trial: int, seed: int) -> CartPoleCandidate:
    if selector == "raw_proxy":
        return max(pool, key=lambda item: item.proxy_score)
    if selector == "support_filter":
        feasible = [item for item in pool if item.behavior_logp > -0.45]
        return max(feasible or pool, key=lambda item: item.proxy_score)
    if selector == "conservative_q":
        return max(pool, key=lambda item: item.proxy_score + 0.50 * item.behavior_logp)
    if selector == "oracle_real":
        return max(pool, key=lambda item: item.episode_return)
    if selector == "behavior_clone":
        return _fixed_rollout(trial, seed, "expert")
    if selector == "random_policy":
        return _fixed_rollout(trial, seed, "random")
    raise ValueError(selector)


def _row(trial: int, n: int, selector: str, candidate: CartPoleCandidate) -> dict[str, Any]:
    return {
        "trial": int(trial),
        "N": int(n),
        "selector": selector,
        "policy_kind": candidate.policy_kind,
        "episode_return": int(candidate.episode_return),
        "proxy_score": candidate.proxy_score,
        "behavior_logp": candidate.behavior_logp,
        "early_switch_rate": candidate.early_switch_rate,
        "late_action_imbalance": candidate.late_action_imbalance,
        "max_abs_x": candidate.max_abs_x,
        "max_abs_theta": candidate.max_abs_theta,
    }


def run_cartpole_benchmark(*, trials: int = 32, seed: int = 2026) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_values = (1, 4, 16, 64)
    selectors = ("raw_proxy", "support_filter", "conservative_q", "behavior_clone", "random_policy", "oracle_real")
    for trial in range(int(trials)):
        pool = _candidate_pool(trial, seed, MAX_N)
        for n in n_values:
            subpool = pool[:n]
            for selector in selectors:
                rows.append(_row(trial, n, selector, _select(selector, subpool, trial, seed)))
    meta = {
        "benchmark": "CartPole-v1",
        "dynamics": "Gymnasium CartPole-v1 constants",
        "max_steps": MAX_STEPS,
        "trials": int(trials),
        "seed": int(seed),
        "candidate_counts": list(n_values),
        "selectors": list(selectors),
    }
    return rows, meta


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    selectors = sorted({str(row["selector"]) for row in rows})
    n_values = sorted({int(row["N"]) for row in rows})
    for selector in selectors:
        for n in n_values:
            group = [row for row in rows if str(row["selector"]) == selector and int(row["N"]) == n]
            if not group:
                continue
            item: dict[str, Any] = {"selector": selector, "N": n, "count": len(group)}
            for key in (
                "episode_return",
                "proxy_score",
                "behavior_logp",
                "early_switch_rate",
                "late_action_imbalance",
                "max_abs_x",
                "max_abs_theta",
            ):
                item[f"{key}_mean"] = float(np.mean([float(row[key]) for row in group]))
            item["exploit_policy_rate"] = float(
                np.mean([str(row["policy_kind"]).startswith("short_exploit") for row in group])
            )
            out.append(item)
    return out


def claim_gates(summary: list[dict[str, Any]]) -> dict[str, Any]:
    def row(selector: str, n: int) -> dict[str, Any]:
        matches = [item for item in summary if item["selector"] == selector and int(item["N"]) == n]
        if not matches:
            raise KeyError((selector, n))
        return matches[0]

    raw_1 = row("raw_proxy", 1)
    raw_64 = row("raw_proxy", 64)
    support_64 = row("support_filter", 64)
    conservative_64 = row("conservative_q", 64)
    bc_64 = row("behavior_clone", 64)
    oracle_64 = row("oracle_real", 64)

    checks = {
        "cartpole_raw_proxy_extremizes": _claim(
            raw_64["proxy_score_mean"] - raw_1["proxy_score_mean"],
            0.04,
            ">",
            "Increasing candidate count raises the raw short-horizon proxy score on CartPole-v1.",
        ),
        "cartpole_raw_return_collapses": _claim(
            raw_64["episode_return_mean"] - raw_1["episode_return_mean"],
            -120.0,
            "<",
            "The same raw selection lowers true CartPole-v1 episode return.",
        ),
        "cartpole_support_filter_repairs": _claim(
            support_64["episode_return_mean"] - raw_64["episode_return_mean"],
            150.0,
            ">",
            "A support-likelihood filter recovers true return over raw proxy selection.",
        ),
        "cartpole_conservative_q_repairs": _claim(
            conservative_64["episode_return_mean"] - raw_64["episode_return_mean"],
            150.0,
            ">",
            "A conservative support-penalized selector recovers true return over raw proxy selection.",
        ),
        "cartpole_behavior_clone_baseline_beats_raw": _claim(
            bc_64["episode_return_mean"] - raw_64["episode_return_mean"],
            150.0,
            ">",
            "A behavior-cloning baseline beats raw high-N proxy selection.",
        ),
        "cartpole_oracle_gap_visible": _claim(
            oracle_64["episode_return_mean"] - raw_64["episode_return_mean"],
            150.0,
            ">",
            "The sampled pool contains high-return candidates that raw proxy selection ignores.",
        ),
        "cartpole_raw_exploit_rate_high": _claim(
            raw_64["exploit_policy_rate"],
            0.50,
            ">",
            "Raw high-N proxy selection is dominated by short-horizon exploit policies.",
        ),
    }
    return {
        "all_passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "summary": (
            f"raw return change {raw_64['episode_return_mean'] - raw_1['episode_return_mean']:.1f}, "
            f"support repair {support_64['episode_return_mean'] - raw_64['episode_return_mean']:.1f}, "
            f"conservative repair {conservative_64['episode_return_mean'] - raw_64['episode_return_mean']:.1f}."
        ),
    }


def _claim(value: float, threshold: float, op: str, description: str) -> dict[str, Any]:
    passed = value > threshold if op == ">" else value < threshold
    return {
        "passed": bool(passed),
        "observed": float(value),
        "threshold": float(threshold),
        "op": op,
        "description": description,
    }


def write_outputs(rows: list[dict[str, Any]], meta: dict[str, Any], output_dir: Path, figure_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    claims = claim_gates(summary)
    _write_csv(output_dir / "cartpole_trials.csv", rows)
    _write_csv(output_dir / "cartpole_summary.csv", summary)
    _write_json(output_dir / "claims.json", claims)
    manifest = {
        **meta,
        "trials": str(output_dir / "cartpole_trials.csv"),
        "summary": str(output_dir / "cartpole_summary.csv"),
        "claims": str(output_dir / "claims.json"),
        "figure": str(figure_path),
        "all_passed": claims["all_passed"],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def make_figure(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharex=True)
    order = ("raw_proxy", "support_filter", "conservative_q", "behavior_clone", "oracle_real")
    colors = {
        "raw_proxy": "#d95f02",
        "support_filter": "#1b9e77",
        "conservative_q": "#7570b3",
        "behavior_clone": "#66a61e",
        "oracle_real": "#4daf4a",
    }
    for selector in order:
        rows = sorted([item for item in summary if item["selector"] == selector], key=lambda item: int(item["N"]))
        n_values = [int(item["N"]) for item in rows]
        label = selector.replace("_", " ")
        axes[0].plot(n_values, [item["proxy_score_mean"] for item in rows], marker="o", color=colors[selector], label=label)
        axes[1].plot(n_values, [item["episode_return_mean"] for item in rows], marker="o", color=colors[selector])
        axes[2].plot(n_values, [item["exploit_policy_rate"] for item in rows], marker="o", color=colors[selector])
    axes[0].set_ylabel("proxy score")
    axes[1].set_ylabel("episode return")
    axes[2].set_ylabel("exploit-policy rate")
    axes[0].set_title("short-horizon proxy")
    axes[1].set_title("true CartPole return")
    axes[2].set_title("selected exploit rate")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks((1, 4, 16, 64))
        ax.set_xticklabels(("1", "4", "16", "64"))
        ax.set_xlabel("candidate count N")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("CartPole-v1 return-support benchmark")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
