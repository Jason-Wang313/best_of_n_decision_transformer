"""Expanded v3 stress suite for return-support Decision Transformer audits."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from return_support_audit.config import ExperimentConfig
from return_support_audit.data import generate_offline_dataset
from return_support_audit.model import TinyDecisionTransformer
from return_support_audit.repairs import (
    Candidate,
    SupportEstimator,
    make_candidates,
    select_behavior_likelihood_constrained,
    select_naive,
    select_oracle_real,
    select_pilot_calibrated,
)


METRIC_KEYS = (
    "selected_proxy",
    "selected_real",
    "selected_risk",
    "behavior_logp",
    "prompt_satisfaction_gap",
    "support_gap_q95",
    "unsafe_action_rate",
    "mean_action",
    "max_action",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _fit_world(config: ExperimentConfig) -> tuple[TinyDecisionTransformer, SupportEstimator]:
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
    return model, support


def _candidate_set(
    model: TinyDecisionTransformer,
    support: SupportEstimator,
    target_return: float,
    n: int,
    rng: np.random.Generator,
    config: ExperimentConfig,
    noise_scale: float | None = None,
) -> list[Candidate]:
    return make_candidates(
        model.rollout_batch(
            target_return=float(target_return),
            n=int(n),
            rng=rng,
            noise_scale=config.model_noise_scale if noise_scale is None else float(noise_scale),
        ),
        support=support,
    )


def _select(
    selector: str,
    candidates: list[Candidate],
    support: SupportEstimator,
    rng: np.random.Generator,
    model: TinyDecisionTransformer,
    target_return: float,
    n: int,
    config: ExperimentConfig,
    pilot_size: int | None = None,
) -> Candidate:
    if selector == "naive":
        return select_naive(candidates)
    if selector == "random":
        return candidates[int(rng.integers(0, len(candidates)))]
    if selector == "behavior_q05":
        return select_behavior_likelihood_constrained(candidates, support, threshold=support.trajectory_logp_q05)
    if selector == "behavior_q10":
        return select_behavior_likelihood_constrained(candidates, support, threshold=support.trajectory_logp_q10)
    if selector == "pilot_calibrated":
        pilot = _candidate_set(
            model,
            support,
            target_return,
            max(int(pilot_size or config.n_pilot), 3),
            rng,
            config,
        )
        return select_pilot_calibrated(candidates, pilot)
    if selector == "oracle_real":
        return select_oracle_real(candidates)
    raise ValueError(f"unknown selector: {selector}")


def _row(
    *,
    family: str,
    setting: str,
    setting_value: str | float | int,
    selector: str,
    target_label: str,
    target_return: float,
    effective_target: float,
    n: int,
    replicate: int,
    selected: Candidate,
    support: SupportEstimator,
    noise_scale: float,
    pilot_size: int | None = None,
) -> dict[str, Any]:
    actions = selected.rollout.actions
    unsafe = np.mean(np.abs(actions) > selected.rollout.safe_action) if hasattr(selected.rollout, "safe_action") else np.mean(np.abs(actions) > 0.72)
    return {
        "family": family,
        "setting": setting,
        "setting_value": setting_value,
        "selector": selector,
        "target_label": target_label,
        "target_return": float(target_return),
        "target_delta_q95": float(target_return - support.q95),
        "effective_target": float(effective_target),
        "n": int(n),
        "replicate": int(replicate),
        "pilot_size": "" if pilot_size is None else int(pilot_size),
        "noise_scale": float(noise_scale),
        "selected_proxy": selected.proxy,
        "selected_real": selected.real,
        "selected_risk": selected.risk,
        "behavior_logp": selected.behavior_logp,
        "prompt_satisfaction_gap": abs(float(target_return) - selected.proxy),
        "support_gap_q95": support.support_gap(float(target_return), 0.95),
        "unsafe_action_rate": float(unsafe),
        "mean_action": float(np.mean(actions)),
        "max_action": float(np.max(actions)),
    }


def _aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = (
        "family",
        "setting",
        "setting_value",
        "selector",
        "target_label",
        "target_return",
        "target_delta_q95",
        "effective_target",
        "n",
        "pilot_size",
        "noise_scale",
    )
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    out: list[dict[str, Any]] = []
    for group_key, values in groups.items():
        item = {key: group_key[idx] for idx, key in enumerate(keys)}
        item["count"] = len(values)
        for metric in METRIC_KEYS:
            vals = np.asarray([float(row[metric]) for row in values], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(vals))
            item[f"{metric}_std"] = float(np.std(vals))
        out.append(item)
    return sorted(out, key=lambda r: (r["family"], str(r["setting_value"]), r["selector"], int(r["n"])))


def _run_selection_grid(
    *,
    model: TinyDecisionTransformer,
    support: SupportEstimator,
    config: ExperimentConfig,
    reps: int,
    family: str,
    setting: str,
    setting_value: str | float | int,
    selector: str,
    target_label: str,
    target_return: float,
    n_values: Iterable[int],
    seed: int,
    effective_quantile: float | None = None,
    noise_scale: float | None = None,
    pilot_size: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    effective_target = (
        support.calibrate_target(float(target_return), effective_quantile)
        if effective_quantile is not None
        else float(target_return)
    )
    selector_name = selector if effective_quantile is None else f"{selector}_q{int(effective_quantile * 100)}"
    for n in n_values:
        for replicate in range(reps):
            candidates = _candidate_set(
                model,
                support,
                effective_target,
                int(n),
                rng,
                config,
                noise_scale=noise_scale,
            )
            selected = _select(
                selector,
                candidates,
                support,
                rng,
                model,
                target_return=float(target_return),
                n=int(n),
                config=config,
                pilot_size=pilot_size,
            )
            rows.append(
                _row(
                    family=family,
                    setting=setting,
                    setting_value=setting_value,
                    selector=selector_name,
                    target_label=target_label,
                    target_return=float(target_return),
                    effective_target=effective_target,
                    n=int(n),
                    replicate=replicate,
                    selected=selected,
                    support=support,
                    noise_scale=config.model_noise_scale if noise_scale is None else float(noise_scale),
                    pilot_size=pilot_size,
                )
            )
    return rows


def _make_figures(summary: list[dict[str, Any]], output_dir: Path) -> list[str]:
    figures_dir = output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    tail = [r for r in summary if r["family"] == "candidate_count_256" and r["selector"] == "naive"]
    tail = sorted(tail, key=lambda r: int(r["n"]))
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 8.0), sharex=True)
    n = [int(r["n"]) for r in tail]
    axes[0].plot(n, [r["selected_proxy_mean"] for r in tail], marker="o", color="#1b9e77")
    axes[0].set_ylabel("proxy S")
    axes[1].plot(n, [r["selected_real_mean"] for r in tail], marker="s", color="#d95f02")
    axes[1].set_ylabel("real R")
    axes[2].plot(n, [r["unsafe_action_rate_mean"] for r in tail], marker="^", color="#7570b3")
    axes[2].set_ylabel("unsafe action rate")
    axes[2].set_xlabel("candidate count N")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Out-of-support DT prompt tail to N=256")
    path = figures_dir / "figure6_candidate_count_256.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    sweep = [r for r in summary if r["family"] == "target_overshoot" and r["selector"] == "naive"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for n_value, marker, color in [(1, "o", "#4c566a"), (256, "s", "#d95f02")]:
        rows = sorted([r for r in sweep if int(r["n"]) == n_value], key=lambda r: float(r["target_delta_q95"]))
        ax.plot(
            [r["target_delta_q95"] for r in rows],
            [r["selected_real_mean"] for r in rows],
            marker=marker,
            color=color,
            label=f"N={n_value}",
        )
    ax.axvline(0.0, color="#7570b3", linestyle="--", linewidth=1.1)
    ax.set_xlabel("target return minus offline q95")
    ax.set_ylabel("selected real utility")
    ax.set_title("Target overshoot sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = figures_dir / "figure7_target_overshoot.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    repair = [
        r
        for r in summary
        if r["family"] == "repair_ladder_256" and int(r["n"]) == 256
    ]
    order = ["random", "naive", "naive_q90", "naive_q95", "behavior_q10", "behavior_q05", "pilot_calibrated", "oracle_real"]
    repair_map = {r["selector"]: r for r in repair}
    labels = [name.replace("_", "\n") for name in order if name in repair_map]
    values = [repair_map[name]["selected_real_mean"] for name in order if name in repair_map]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(labels, values, color=["#7f7f7f", "#d95f02", "#e6ab02", "#1b9e77", "#7570b3", "#66a61e", "#e7298a", "#4daf4a"][: len(values)])
    ax.axhline(repair_map["naive"]["selected_real_mean"], color="#4c566a", linestyle=":", linewidth=1.2)
    ax.set_ylabel("selected real utility")
    ax.set_title("Return-support repair ladder at N=256")
    ax.grid(True, axis="y", alpha=0.25)
    path = figures_dir / "figure8_repair_ladder_256.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    pilot = sorted(
        [r for r in summary if r["family"] == "pilot_size_ablation"],
        key=lambda r: int(r["pilot_size"]),
    )
    axes[0].plot([int(r["pilot_size"]) for r in pilot], [r["selected_real_mean"] for r in pilot], marker="o", color="#e7298a")
    axes[0].set_xlabel("pilot labels")
    axes[0].set_ylabel("selected real utility")
    axes[0].set_title("Pilot calibration size")
    noise = sorted(
        [r for r in summary if r["family"] == "noise_sweep" and int(r["n"]) == 256],
        key=lambda r: float(r["noise_scale"]),
    )
    axes[1].plot([r["noise_scale"] for r in noise], [r["selected_real_mean"] for r in noise], marker="s", color="#d95f02")
    ax2 = axes[1].twinx()
    ax2.plot([r["noise_scale"] for r in noise], [r["unsafe_action_rate_mean"] for r in noise], marker="^", color="#7570b3")
    axes[1].set_xlabel("rollout noise scale")
    axes[1].set_ylabel("real utility")
    ax2.set_ylabel("unsafe action rate")
    axes[1].set_title("Noise stress at N=256")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    path = figures_dir / "figure9_pilot_noise_stress.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))
    return paths


def _by(summary: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    matches = [
        row for row in summary
        if all(str(row.get(key)) == str(value) for key, value in kwargs.items())
    ]
    if len(matches) != 1:
        raise KeyError(kwargs)
    return matches[0]


def _claims(summary: list[dict[str, Any]], quick: bool = False) -> dict[str, Any]:
    tail_1 = _by(summary, family="candidate_count_256", selector="naive", n=1)
    tail_256 = _by(summary, family="candidate_count_256", selector="naive", n=256)
    raw_256 = _by(summary, family="repair_ladder_256", selector="naive", n=256)
    pilot_256 = _by(summary, family="repair_ladder_256", selector="pilot_calibrated", n=256)
    behavior_256 = _by(summary, family="repair_ladder_256", selector="behavior_q05", n=256)
    oracle_256 = _by(summary, family="repair_ladder_256", selector="oracle_real", n=256)
    overshoot_hi_1 = max(
        [r for r in summary if r["family"] == "target_overshoot" and int(r["n"]) == 1],
        key=lambda r: float(r["target_delta_q95"]),
    )
    overshoot_hi_256 = max(
        [r for r in summary if r["family"] == "target_overshoot" and int(r["n"]) == 256],
        key=lambda r: float(r["target_delta_q95"]),
    )
    pilot_rows = [r for r in summary if r["family"] == "pilot_size_ablation"]
    noise_rows = [r for r in summary if r["family"] == "noise_sweep" and int(r["n"]) == 256]

    checks = {
        "candidate_count_proxy_rises_to_256": {
            "observed": tail_256["selected_proxy_mean"] - tail_1["selected_proxy_mean"],
            "threshold": 0.10,
            "op": ">",
        },
        "candidate_count_real_drops_to_256": {
            "observed": tail_256["selected_real_mean"] - tail_1["selected_real_mean"],
            "threshold": -0.25,
            "op": "<",
        },
        "candidate_count_selected_risk_rises": {
            "observed": tail_256["selected_risk_mean"] - tail_1["selected_risk_mean"],
            "threshold": 0.003 if quick else 0.005,
            "op": ">",
        },
        "pilot_repair_recovers_raw_tail": {
            "observed": pilot_256["selected_real_mean"] - raw_256["selected_real_mean"],
            "threshold": 0.45,
            "op": ">",
        },
        "behavior_filter_recovers_raw_tail": {
            "observed": behavior_256["selected_real_mean"] - raw_256["selected_real_mean"],
            "threshold": 0.45,
            "op": ">",
        },
        "oracle_gap_shows_recoverable_candidates": {
            "observed": oracle_256["selected_real_mean"] - raw_256["selected_real_mean"],
            "threshold": 0.60,
            "op": ">",
        },
        "target_overshoot_high_n_worsens_high_gap": {
            "observed": overshoot_hi_256["selected_real_mean"] - overshoot_hi_1["selected_real_mean"],
            "threshold": -0.20,
            "op": "<",
        },
        "pilot_size_changes_calibrated_real": {
            "observed": max(r["selected_real_mean"] for r in pilot_rows) - min(r["selected_real_mean"] for r in pilot_rows),
            "threshold": 0.03,
            "op": ">",
        },
        "noise_scale_changes_unsafe_rate": {
            "observed": max(r["unsafe_action_rate_mean"] for r in noise_rows) - min(r["unsafe_action_rate_mean"] for r in noise_rows),
            "threshold": 0.02,
            "op": ">",
        },
    }

    for payload in checks.values():
        observed = float(payload["observed"])
        threshold = float(payload["threshold"])
        op = payload["op"]
        payload["passed"] = (
            observed > threshold if op == ">" else
            observed < threshold if op == "<" else
            observed >= threshold
        )
    return {
        "mode": "quick-smoke" if quick else "full-v3",
        "checks": checks,
        "all_passed": all(payload["passed"] for payload in checks.values()),
    }


def run(quick: bool = False, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path.cwd()
    artifact_dir = Path(output_dir) if output_dir is not None else root / "results" / "expansion"
    config = ExperimentConfig.smoke(root) if quick else ExperimentConfig.full(root)
    reps = 6 if quick else 42
    config = replace(config, n_values=(1, 2, 4, 8, 16, 32, 64, 128, 256))
    model, support = _fit_world(config)
    out_target = support.q95 + 3.0
    rows: list[dict[str, Any]] = []

    rows.extend(
        _run_selection_grid(
            model=model,
            support=support,
            config=config,
            reps=reps,
            family="candidate_count_256",
            setting="candidate_count",
            setting_value="Nmax256",
            selector="naive",
            target_label="out_of_support",
            target_return=out_target,
            n_values=config.n_values,
            seed=config.seed + 10,
        )
    )

    overshoot_deltas = (0.0, 1.5, 3.0, 4.5) if quick else (0.0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5)
    for delta in overshoot_deltas:
        target = support.q95 + delta
        rows.extend(
            _run_selection_grid(
                model=model,
                support=support,
                config=config,
                reps=reps,
                family="target_overshoot",
                setting="target_delta_q95",
                setting_value=delta,
                selector="naive",
                target_label=f"q95_plus_{delta:.2f}",
                target_return=target,
                n_values=(1, 256),
                seed=config.seed + 100 + int(delta * 100),
            )
        )

    repair_specs = [
        ("random", None),
        ("naive", None),
        ("naive", 0.90),
        ("naive", 0.95),
        ("behavior_q10", None),
        ("behavior_q05", None),
        ("pilot_calibrated", None),
        ("oracle_real", None),
    ]
    for selector, quantile in repair_specs:
        rows.extend(
            _run_selection_grid(
                model=model,
                support=support,
                config=config,
                reps=reps,
                family="repair_ladder_256",
                setting="repair",
                setting_value="N256",
                selector=selector,
                target_label="out_of_support",
                target_return=out_target,
                n_values=(64, 128, 256),
                seed=config.seed + 500 + len(rows),
                effective_quantile=quantile,
            )
        )

    pilot_sizes = (4, 16) if quick else (4, 8, 16, 32, 64)
    for pilot_size in pilot_sizes:
        rows.extend(
            _run_selection_grid(
                model=model,
                support=support,
                config=config,
                reps=reps,
                family="pilot_size_ablation",
                setting="pilot_size",
                setting_value=pilot_size,
                selector="pilot_calibrated",
                target_label="out_of_support",
                target_return=out_target,
                n_values=(256,),
                seed=config.seed + 900 + int(pilot_size),
                pilot_size=pilot_size,
            )
        )

    noise_values = (0.16, 0.48) if quick else (0.16, 0.32, 0.48, 0.64)
    for noise in noise_values:
        rows.extend(
            _run_selection_grid(
                model=model,
                support=support,
                config=config,
                reps=reps,
                family="noise_sweep",
                setting="noise_scale",
                setting_value=noise,
                selector="naive",
                target_label="out_of_support",
                target_return=out_target,
                n_values=(1, 64, 256),
                seed=config.seed + 1200 + int(noise * 100),
                noise_scale=noise,
            )
        )

    summary = _aggregate(rows)
    figures_dir = root / "figures" if output_dir is None else artifact_dir / "figures"
    figures = _make_figures(summary, figures_dir)
    claims = _claims(summary, quick=quick)

    summary_path = artifact_dir / "expanded_summary.csv"
    trials_path = artifact_dir / "expanded_trials.csv"
    claims_path = artifact_dir / "claims.json"
    manifest_path = artifact_dir / "manifest.json"
    _write_csv(summary_path, summary)
    _write_csv(trials_path, rows)
    _write_json(claims_path, claims)
    manifest = {
        "quick": quick,
        "summary": summary_path,
        "trials": trials_path,
        "claims": claims_path,
        "figures": figures,
        "support": support.describe_target(out_target),
        "all_passed": claims["all_passed"],
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest": manifest_path}


if __name__ == "__main__":
    payload = run(quick="--quick" in sys.argv)
    claims = json.loads(Path(payload["claims"]).read_text(encoding="utf-8"))
    tail = claims["checks"]["candidate_count_real_drops_to_256"]["observed"]
    repair = claims["checks"]["pilot_repair_recovers_raw_tail"]["observed"]
    print(f"expanded decision-transformer suite: real tail change {tail:.3f}, pilot repair {repair:.3f}")
    print(f"Manifest: {payload['manifest']}")
