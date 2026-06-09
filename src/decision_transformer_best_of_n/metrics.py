"""Metrics for return-conditioning fantasy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Microscope:
    target_return: float
    prompt_satisfaction_gap: float
    support_gap: float
    real_utility_gap: float
    selected_proxy_return: float
    selected_real_utility: float

    def to_dict(self) -> dict[str, float]:
        return {
            "target_return": self.target_return,
            "prompt_satisfaction_gap": self.prompt_satisfaction_gap,
            "support_gap": self.support_gap,
            "real_utility_gap": self.real_utility_gap,
            "selected_proxy_return": self.selected_proxy_return,
            "selected_real_utility": self.selected_real_utility,
        }


def return_fantasy_microscope(
    target_return: float,
    selected_proxy_return: float,
    selected_real_utility: float,
    support_boundary: float,
    baseline_real_utility: float,
) -> Microscope:
    """Compute the three gaps used by the Return Fantasy Microscope."""

    return Microscope(
        target_return=float(target_return),
        prompt_satisfaction_gap=float(abs(target_return - selected_proxy_return)),
        support_gap=float(max(target_return - support_boundary, 0.0)),
        real_utility_gap=float(selected_real_utility - baseline_real_utility),
        selected_proxy_return=float(selected_proxy_return),
        selected_real_utility=float(selected_real_utility),
    )


def phase_label(
    n_values: Iterable[int],
    proxy_values: Iterable[float],
    real_values: Iterable[float],
    min_proxy_gain: float = 0.15,
    utility_tol: float = 0.10,
) -> str:
    """Classify the tail phase as helps, saturates, or hurts."""

    n_arr = np.asarray(list(n_values), dtype=float)
    proxy = np.asarray(list(proxy_values), dtype=float)
    real = np.asarray(list(real_values), dtype=float)
    if len(n_arr) < 2 or proxy.shape != real.shape or proxy.shape != n_arr.shape:
        raise ValueError("n_values, proxy_values, and real_values must align and contain >=2 points.")

    proxy_gain = float(proxy[-1] - proxy[0])
    real_gain = float(real[-1] - real[0])
    if proxy_gain < min_proxy_gain and abs(real_gain) <= utility_tol:
        return "saturates"
    if real_gain > utility_tol:
        return "helps"
    if real_gain < -utility_tol:
        return "hurts"
    return "saturates"


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    """Aggregate repeated selection rows by strategy, target, and N."""

    buckets: dict[tuple[str, str, int], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["strategy"]), str(row["target_label"]), int(row["n"]))
        buckets.setdefault(key, []).append(row)

    out: list[dict[str, float | int | str]] = []
    numeric_fields = [
        "target_return",
        "selected_proxy",
        "selected_real",
        "selected_risk",
        "behavior_logp",
        "prompt_satisfaction_gap",
        "support_gap",
    ]
    for (strategy, target_label, n), values in sorted(buckets.items()):
        agg: dict[str, float | int | str] = {
            "strategy": strategy,
            "target_label": target_label,
            "n": n,
            "replicates": len(values),
        }
        for field in numeric_fields:
            arr = np.asarray([float(v[field]) for v in values], dtype=float)
            agg[f"{field}_mean"] = float(np.mean(arr))
            agg[f"{field}_std"] = float(np.std(arr))
        out.append(agg)

    # Add real utility gap against the N=1 value within strategy/target.
    baseline: dict[tuple[str, str], float] = {}
    for row in out:
        if int(row["n"]) == 1:
            baseline[(str(row["strategy"]), str(row["target_label"]))] = float(row["selected_real_mean"])
    for row in out:
        base = baseline.get((str(row["strategy"]), str(row["target_label"])), float(row["selected_real_mean"]))
        row["real_utility_gap_vs_n1"] = float(row["selected_real_mean"]) - base
    return out

