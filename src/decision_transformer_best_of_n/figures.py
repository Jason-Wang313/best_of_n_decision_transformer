"""Matplotlib figure generation for the paper skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .repairs import SupportEstimator


PALETTE = {
    "proxy": "#1b9e77",
    "real": "#d95f02",
    "support": "#7570b3",
    "pilot": "#e7298a",
    "oracle": "#66a61e",
    "neutral": "#4c566a",
}


def _save(fig: plt.Figure, output_dir: Path, stem: str, formats: tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _subset(
    rows: list[dict[str, Any]],
    strategy: str | None = None,
    target_label: str | None = None,
    n: int | None = None,
) -> list[dict[str, Any]]:
    out = rows
    if strategy is not None:
        out = [row for row in out if row["strategy"] == strategy]
    if target_label is not None:
        out = [row for row in out if row["target_label"] == target_label]
    if n is not None:
        out = [row for row in out if int(row["n"]) == n]
    return sorted(out, key=lambda row: int(row.get("n", 0)))


def plot_return_conditioning_fantasy(
    aggregated: list[dict[str, Any]],
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    rows = _subset(aggregated, strategy="naive", target_label="out_of_support")
    n = np.asarray([int(row["n"]) for row in rows])
    proxy = np.asarray([float(row["selected_proxy_mean"]) for row in rows])
    real = np.asarray([float(row["selected_real_mean"]) for row in rows])
    target = float(rows[0]["target_return_mean"])

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.2), sharex=True)
    axes[0].plot(n, proxy, marker="o", color=PALETTE["proxy"], label="Selected proxy return S")
    axes[0].axhline(target, color=PALETTE["support"], linestyle="--", linewidth=1.2, label="Target return")
    axes[0].set_ylabel("Proxy return")
    axes[0].legend(frameon=False)
    axes[0].set_title("Return-conditioning fantasy under out-of-support prompts")

    axes[1].plot(n, real, marker="s", color=PALETTE["real"], label="Real utility R")
    axes[1].axhline(real[0], color=PALETTE["neutral"], linestyle=":", linewidth=1.2, label="N=1 baseline")
    axes[1].set_xlabel("Best-of-N candidates")
    axes[1].set_ylabel("Real utility")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(n)
    axes[1].set_xticklabels([str(v) for v in n])
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, alpha=0.22)
    _save(fig, output_dir, "return_conditioning_fantasy", config.figure_formats)


def plot_repair_comparison(
    aggregated: list[dict[str, Any]],
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    n_max = max(config.n_values)
    rows = _subset(aggregated, target_label="out_of_support", n=n_max)
    order = [
        "naive",
        "support_calibrated",
        "likelihood_constrained",
        "conservative_gate",
        "pilot_calibrated",
        "oracle_real",
    ]
    rows_by_strategy = {row["strategy"]: row for row in rows}
    labels = [s.replace("_", "\n") for s in order if s in rows_by_strategy]
    values = [float(rows_by_strategy[s]["selected_real_mean"]) for s in order if s in rows_by_strategy]
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e6ab02", "#e7298a", "#66a61e"][: len(values)]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(labels, values, color=colors)
    ax.axhline(values[0], color=PALETTE["neutral"], linestyle=":", linewidth=1.2)
    ax.set_ylabel("Selected real utility")
    ax.set_title(f"Repair ladder at N={n_max} for out-of-support target")
    ax.grid(True, axis="y", alpha=0.22)
    _save(fig, output_dir, "repair_comparison", config.figure_formats)


def plot_target_return_sweep(
    target_sweep: list[dict[str, Any]],
    support: SupportEstimator,
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    n_max = max(config.n_values)
    rows_n1 = sorted([row for row in target_sweep if int(row["n"]) == 1], key=lambda row: row["target_return"])
    rows_nmax = sorted(
        [row for row in target_sweep if int(row["n"]) == n_max],
        key=lambda row: row["target_return"],
    )
    targets = np.asarray([float(row["target_return"]) for row in rows_n1])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True)
    axes[0].plot(
        targets,
        [float(row["selected_proxy_mean"]) for row in rows_n1],
        marker="o",
        color=PALETTE["proxy"],
        label="N=1",
    )
    axes[0].plot(
        targets,
        [float(row["selected_proxy_mean"]) for row in rows_nmax],
        marker="s",
        color=PALETTE["support"],
        label=f"N={n_max}",
    )
    axes[0].set_ylabel("Selected proxy return")
    axes[0].set_title("Prompt satisfaction")

    axes[1].plot(
        targets,
        [float(row["selected_real_mean"]) for row in rows_n1],
        marker="o",
        color=PALETTE["real"],
        label="N=1",
    )
    axes[1].plot(
        targets,
        [float(row["selected_real_mean"]) for row in rows_nmax],
        marker="s",
        color=PALETTE["pilot"],
        label=f"N={n_max}",
    )
    axes[1].set_ylabel("Selected real utility")
    axes[1].set_title("Real utility")

    for ax in axes:
        ax.axvline(support.q95, color=PALETTE["neutral"], linestyle="--", linewidth=1.2)
        ax.set_xlabel("Target return")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False)
    _save(fig, output_dir, "target_return_sweep", config.figure_formats)


def plot_support_diagnostics(
    target_sweep: list[dict[str, Any]],
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    n_max = max(config.n_values)
    rows = sorted([row for row in target_sweep if int(row["n"]) == n_max], key=lambda row: row["target_return"])
    support_gap = np.asarray([float(row["support_gap"]) for row in rows])
    prompt_gap = np.asarray([float(row["prompt_satisfaction_gap_mean"]) for row in rows])
    real_gap = np.asarray([float(row["real_utility_gap_vs_n1"]) for row in rows])
    targets = np.asarray([float(row["target_return"]) for row in rows])

    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    sc = ax.scatter(
        support_gap,
        real_gap,
        c=targets,
        s=58,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.7,
    )
    ax.plot(support_gap, real_gap, color="#5e6a71", alpha=0.35)
    for x, y, gap in zip(support_gap, real_gap, prompt_gap):
        if x > 0:
            ax.annotate(f"gap {gap:.1f}", (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.0, color=PALETTE["neutral"], linestyle=":", linewidth=1.2)
    ax.set_xlabel("Support gap beyond offline q95")
    ax.set_ylabel(f"Real utility gain of N={n_max} vs N=1")
    ax.set_title("Return Fantasy Microscope diagnostics")
    ax.grid(True, alpha=0.22)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Target return")
    _save(fig, output_dir, "support_diagnostics", config.figure_formats)


def plot_exact_law_validation(
    exact_rows: list[dict[str, Any]],
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    rows = sorted(exact_rows, key=lambda row: int(row["n"]))
    n = np.asarray([int(row["n"]) for row in rows])
    exact = np.asarray([float(row["exact_expected_real"]) for row in rows])
    mc = np.asarray([float(row["mc_expected_real"]) for row in rows])

    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    ax.plot(n, exact, marker="o", color=PALETTE["support"], label="Exact tie-aware law")
    ax.plot(n, mc, marker="s", color=PALETTE["real"], linestyle="--", label="Monte Carlo")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.set_xlabel("Best-of-N candidates")
    ax.set_ylabel("Expected selected real utility")
    ax.set_title("Finite-N law validation")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    _save(fig, output_dir, "exact_law_validation", config.figure_formats)


def make_all_figures(
    aggregated: list[dict[str, Any]],
    target_sweep: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
    support: SupportEstimator,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    plot_return_conditioning_fantasy(aggregated, output_dir, config)
    plot_repair_comparison(aggregated, output_dir, config)
    plot_target_return_sweep(target_sweep, support, output_dir, config)
    plot_support_diagnostics(target_sweep, output_dir, config)
    plot_exact_law_validation(exact_rows, output_dir, config)

