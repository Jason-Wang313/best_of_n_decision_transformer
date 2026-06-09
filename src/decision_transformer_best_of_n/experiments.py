"""Experiment pipeline for the controlled Best-of-N DT study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit import run_claim_audit
from .best_of_n import best_of_n_monte_carlo, exact_best_of_n_expectation
from .config import ExperimentConfig
from .data import generate_offline_dataset
from .figures import make_all_figures
from .metrics import aggregate_rows, phase_label, return_fantasy_microscope
from .model import TinyDecisionTransformer
from .repairs import (
    Candidate,
    SupportEstimator,
    deployment_gate,
    make_candidates,
    select_behavior_likelihood_constrained,
    select_naive,
    select_oracle_real,
    select_pilot_calibrated,
)


STRATEGIES = (
    "naive",
    "support_calibrated",
    "likelihood_constrained",
    "conservative_gate",
    "pilot_calibrated",
    "oracle_real",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_set(
    model: TinyDecisionTransformer,
    support: SupportEstimator,
    target_return: float,
    n: int,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> list[Candidate]:
    return make_candidates(
        model.rollout_batch(
            target_return=target_return,
            n=n,
            rng=rng,
            noise_scale=config.model_noise_scale,
        ),
        support=support,
    )


def _select(
    strategy: str,
    model: TinyDecisionTransformer,
    support: SupportEstimator,
    original_target: float,
    n: int,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> tuple[Candidate, float, str]:
    effective_target = float(original_target)
    gate = deployment_gate(original_target, support, n)

    if strategy == "support_calibrated":
        effective_target = support.calibrate_target(original_target, 0.95)
    elif strategy == "conservative_gate":
        if gate in {"lower_target_return", "collect_pilot_labels", "block_high_n"}:
            effective_target = support.calibrate_target(original_target, 0.90)

    candidates = _candidate_set(model, support, effective_target, n, rng, config)

    if strategy == "naive" or strategy == "support_calibrated":
        selected = select_naive(candidates)
    elif strategy == "likelihood_constrained" or strategy == "conservative_gate":
        selected = select_behavior_likelihood_constrained(candidates, support)
    elif strategy == "pilot_calibrated":
        pilot = _candidate_set(
            model,
            support,
            original_target,
            max(config.n_pilot, 3),
            rng,
            config,
        )
        selected = select_pilot_calibrated(candidates, pilot)
        pilot_naive = select_naive(pilot)
        pilot_safe = select_pilot_calibrated(pilot, pilot)
        gate = deployment_gate(
            original_target,
            support,
            n,
            pilot_delta_real=pilot_safe.real - pilot_naive.real,
            behavior_logp=selected.behavior_logp,
        )
    elif strategy == "oracle_real":
        selected = select_oracle_real(candidates)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return selected, effective_target, gate


def _row_from_selection(
    strategy: str,
    target_label: str,
    target_return: float,
    n: int,
    replicate: int,
    selected: Candidate,
    support: SupportEstimator,
    effective_target: float,
    gate: str,
    baseline_real: float,
) -> dict[str, Any]:
    microscope = return_fantasy_microscope(
        target_return=target_return,
        selected_proxy_return=selected.proxy,
        selected_real_utility=selected.real,
        support_boundary=support.q95,
        baseline_real_utility=baseline_real,
    )
    return {
        "strategy": strategy,
        "target_label": target_label,
        "target_return": float(target_return),
        "effective_target": float(effective_target),
        "n": int(n),
        "replicate": int(replicate),
        "selected_proxy": selected.proxy,
        "selected_real": selected.real,
        "selected_risk": selected.risk,
        "behavior_logp": selected.behavior_logp,
        "prompt_satisfaction_gap": microscope.prompt_satisfaction_gap,
        "support_gap": microscope.support_gap,
        "real_utility_gap": microscope.real_utility_gap,
        "gate_decision": gate,
    }


def evaluate_grid(
    model: TinyDecisionTransformer,
    support: SupportEstimator,
    targets: dict[str, float],
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(config.seed + 1000)
    rows: list[dict[str, Any]] = []
    baseline_real: dict[tuple[str, str], float] = {}

    for strategy in STRATEGIES:
        for target_label, target_return in targets.items():
            for n in config.n_values:
                interim: list[dict[str, Any]] = []
                for replicate in range(config.n_replicates):
                    selected, effective_target, gate = _select(
                        strategy, model, support, target_return, n, rng, config
                    )
                    base = baseline_real.get((strategy, target_label), selected.real)
                    row = _row_from_selection(
                        strategy=strategy,
                        target_label=target_label,
                        target_return=target_return,
                        n=n,
                        replicate=replicate,
                        selected=selected,
                        support=support,
                        effective_target=effective_target,
                        gate=gate,
                        baseline_real=base,
                    )
                    interim.append(row)
                if n == 1:
                    baseline_real[(strategy, target_label)] = float(
                        np.mean([r["selected_real"] for r in interim])
                    )
                    for row in interim:
                        row["real_utility_gap"] = float(
                            row["selected_real"] - baseline_real[(strategy, target_label)]
                        )
                rows.extend(interim)

    aggregated = aggregate_rows(rows)
    return rows, aggregated


def evaluate_target_sweep(
    model: TinyDecisionTransformer,
    support: SupportEstimator,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.seed + 2000)
    n_reps = max(6 if config.mode == "smoke" else 18, config.n_replicates // 3)
    target_count = 7 if config.mode == "smoke" else 11
    targets = np.linspace(support.q50, support.q95 + 3.2, target_count)
    n_values = (1, max(config.n_values))
    rows: list[dict[str, Any]] = []
    baseline_by_target: dict[float, float] = {}

    for target in targets:
        for n in n_values:
            selected_proxy = []
            selected_real = []
            prompt_gap = []
            for _ in range(n_reps):
                candidates = _candidate_set(model, support, float(target), n, rng, config)
                selected = select_naive(candidates)
                selected_proxy.append(selected.proxy)
                selected_real.append(selected.real)
                prompt_gap.append(abs(float(target) - selected.proxy))
            mean_real = float(np.mean(selected_real))
            if n == 1:
                baseline_by_target[float(target)] = mean_real
            rows.append(
                {
                    "target_return": float(target),
                    "n": int(n),
                    "selected_proxy_mean": float(np.mean(selected_proxy)),
                    "selected_real_mean": mean_real,
                    "prompt_satisfaction_gap_mean": float(np.mean(prompt_gap)),
                    "support_gap": support.support_gap(float(target), 0.95),
                    "real_utility_gap_vs_n1": mean_real - baseline_by_target.get(float(target), mean_real),
                }
            )
    return rows


def evaluate_anti_aligned_control(
    model: TinyDecisionTransformer,
    support: SupportEstimator,
    target_return: float,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.seed + 3000)
    n_reps = max(8 if config.mode == "smoke" else 25, config.n_replicates // 2)
    rows: list[dict[str, Any]] = []
    for n in config.n_values:
        selected_real = []
        selected_proxy = []
        for _ in range(n_reps):
            candidates = _candidate_set(model, support, target_return, n, rng, config)
            selected = max(candidates, key=lambda c: (-c.real, c.proxy))
            selected_real.append(selected.real)
            selected_proxy.append(selected.proxy)
        rows.append(
            {
                "target_label": "in_support",
                "target_return": float(target_return),
                "n": int(n),
                "selected_real_mean": float(np.mean(selected_real)),
                "selected_proxy_mean": float(np.mean(selected_proxy)),
                "score_description": "anti_aligned_score_equals_negative_real_utility",
            }
        )
    return rows


def evaluate_exact_law_validation(
    model: TinyDecisionTransformer,
    support: SupportEstimator,
    target_return: float,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.seed + 4000)
    pool_size = 160 if config.mode == "smoke" else 420
    candidates = _candidate_set(model, support, target_return, pool_size, rng, config)
    # Rounding creates real ties, so this figure validates the tie-aware law.
    scores = np.array([round(candidate.proxy, 1) for candidate in candidates], dtype=float)
    utilities = np.array([candidate.real for candidate in candidates], dtype=float)
    trials = 2_000 if config.mode == "smoke" else 10_000
    rows = []
    for n in config.n_values:
        exact = exact_best_of_n_expectation(scores, utilities, n)
        mc = best_of_n_monte_carlo(scores, utilities, n, n_trials=trials, seed=config.seed + n)
        rows.append(
            {
                "n": int(n),
                "exact_expected_real": exact["expected_utility"],
                "mc_expected_real": mc["expected_utility"],
                "abs_error_real": abs(exact["expected_utility"] - mc["expected_utility"]),
                "exact_expected_score": exact["expected_score"],
                "mc_expected_score": mc["expected_score"],
                "stderr_utility": mc["stderr_utility"],
            }
        )
    return rows


def _phase_rows(
    aggregated: list[dict[str, Any]],
    targets: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_label in targets:
        subset = [
            row
            for row in aggregated
            if row["strategy"] == "naive" and row["target_label"] == target_label
        ]
        subset = sorted(subset, key=lambda row: int(row["n"]))
        rows.append(
            {
                "target_label": target_label,
                "target_return": float(targets[target_label]),
                "phase": phase_label(
                    [int(row["n"]) for row in subset],
                    [float(row["selected_proxy_mean"]) for row in subset],
                    [float(row["selected_real_mean"]) for row in subset],
                ),
            }
        )
    return rows


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    root = config.output_dir
    results_dir = root / "results"
    figures_dir = root / "figures"
    docs_dir = root / "docs"
    for directory in (results_dir, figures_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    batch = generate_offline_dataset(
        n_trajectories=config.n_trajectories,
        horizon=config.horizon,
        seed=config.seed,
        safe_action=config.safe_action,
    )
    support = SupportEstimator.fit(batch)
    model = TinyDecisionTransformer(
        horizon=config.horizon,
        safe_action=config.safe_action,
        action_clip=config.action_clip,
        ridge=config.ridge,
    ).fit(batch)

    targets = {
        "in_support": support.q80,
        "edge_support": support.q95,
        "out_of_support": support.q95 + 3.0,
    }

    raw_rows, aggregated = evaluate_grid(model, support, targets, config)
    target_sweep = evaluate_target_sweep(model, support, config)
    anti_aligned = evaluate_anti_aligned_control(model, support, targets["in_support"], config)
    exact_rows = evaluate_exact_law_validation(model, support, targets["out_of_support"], config)
    phase_rows = _phase_rows(aggregated, targets)

    gate_examples = {
        label: deployment_gate(target, support, max(config.n_values))
        for label, target in targets.items()
    }
    summary = {
        "config": config.to_jsonable(),
        "targets": targets,
        "support": support.describe_target(targets["out_of_support"]),
        "gate_examples": gate_examples,
        "phase_rows": phase_rows,
        "mode": config.mode,
    }

    write_json(results_dir / "config.json", config.to_jsonable())
    write_json(results_dir / "summary.json", summary)
    write_csv(results_dir / "selection_raw.csv", raw_rows)
    write_csv(results_dir / "selection_summary.csv", aggregated)
    write_csv(results_dir / "target_return_sweep.csv", target_sweep)
    write_csv(results_dir / "anti_aligned_control.csv", anti_aligned)
    write_csv(results_dir / "exact_law_validation.csv", exact_rows)
    write_csv(results_dir / "tail_phase_diagram.csv", phase_rows)
    write_json(results_dir / "support_diagnostics.json", support.describe_target(targets["out_of_support"]))

    make_all_figures(
        aggregated=aggregated,
        target_sweep=target_sweep,
        exact_rows=exact_rows,
        support=support,
        config=config,
        output_dir=figures_dir,
    )

    audit = run_claim_audit(root)
    summary["audit_verdict"] = audit["verdict"]
    write_json(results_dir / "summary.json", summary)
    return summary


def main(mode: str = "full") -> dict[str, Any]:
    root = Path.cwd()
    config = ExperimentConfig.smoke(root) if mode == "smoke" else ExperimentConfig.full(root)
    return run_experiment(config)
