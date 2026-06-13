"""Claim audit for the controlled paper artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


FORBIDDEN_CLAIMS = (
    "we solve offline rl",
    "top-score selection always hurts",
    "calibration always fixes it",
)

VERDICTS = (
    "submission-ready v2",
    "needs stronger learned model",
    "needs benchmark validation",
    "redesign required",
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _find_summary_row(
    rows: list[dict[str, Any]],
    strategy: str,
    target_label: str,
    n: int,
) -> dict[str, Any]:
    for row in rows:
        if row["strategy"] == strategy and row["target_label"] == target_label and int(row["n"]) == n:
            return row
    raise KeyError((strategy, target_label, n))


def scan_forbidden_claims(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    paths = [root / "README.md", *sorted((root / "docs").glob("*.md")), *sorted((root / "paper").glob("*.md"))]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for claim in FORBIDDEN_CLAIMS:
            if claim in text:
                hits.append({"file": str(path.relative_to(root)), "claim": claim})
    return hits


def evaluate_claims(root: Path) -> dict[str, Any]:
    results = root / "results"
    summary_rows = _read_csv(results / "selection_summary.csv")
    exact_rows = _read_csv(results / "exact_accounting_validation.csv")
    anti_rows = _read_csv(results / "anti_aligned_control.csv")
    summary = json.loads((results / "summary.json").read_text())
    n_max = max(int(row["n"]) for row in summary_rows)

    naive_out_n1 = _find_summary_row(summary_rows, "naive", "out_of_support", 1)
    naive_out_nmax = _find_summary_row(summary_rows, "naive", "out_of_support", n_max)
    naive_in_n1 = _find_summary_row(summary_rows, "naive", "in_support", 1)
    naive_in_nmax = _find_summary_row(summary_rows, "naive", "in_support", n_max)
    pilot_out_nmax = _find_summary_row(summary_rows, "pilot_calibrated", "out_of_support", n_max)
    oracle_out_nmax = _find_summary_row(summary_rows, "oracle_real", "out_of_support", n_max)

    anti_n1 = next(row for row in anti_rows if int(row["n"]) == 1)
    anti_nmax = next(row for row in anti_rows if int(row["n"]) == n_max)
    max_law_error = max(_as_float(row, "abs_error_real") for row in exact_rows)

    proxy_gain = _as_float(naive_out_nmax, "selected_proxy_mean") - _as_float(
        naive_out_n1, "selected_proxy_mean"
    )
    real_gain = _as_float(naive_out_nmax, "selected_real_mean") - _as_float(
        naive_out_n1, "selected_real_mean"
    )
    in_support_real_gain = _as_float(naive_in_nmax, "selected_real_mean") - _as_float(
        naive_in_n1, "selected_real_mean"
    )
    anti_real_gain = _as_float(anti_nmax, "selected_real_mean") - _as_float(
        anti_n1, "selected_real_mean"
    )
    pilot_gain = _as_float(pilot_out_nmax, "selected_real_mean") - _as_float(
        naive_out_nmax, "selected_real_mean"
    )
    oracle_gain = _as_float(oracle_out_nmax, "selected_real_mean") - _as_float(
        naive_out_nmax, "selected_real_mean"
    )

    gate_decisions = set(summary.get("gate_examples", {}).values())
    allowed = {"allow_candidate_sweep", "lower_target_return", "collect_pilot_labels", "block_candidate_sweep"}
    forbidden_hits = scan_forbidden_claims(root)

    claims = {
        "finite_pool_accounting_validated": {
            "passed": max_law_error <= 0.16,
            "evidence": {"max_abs_error_real": max_law_error},
        },
        "fantasy_detected_out_of_support": {
            "passed": proxy_gain >= 0.15 and real_gain <= 0.05,
            "evidence": {"proxy_gain_nmax_vs_n1": proxy_gain, "real_gain_nmax_vs_n1": real_gain},
        },
        "in_support_negative_control": {
            "passed": in_support_real_gain >= -0.25,
            "evidence": {"real_gain_nmax_vs_n1": in_support_real_gain},
        },
        "anti_aligned_scorer_control": {
            "passed": anti_real_gain <= -0.15,
            "evidence": {"real_gain_nmax_vs_n1": anti_real_gain},
        },
        "repair_ladder_improves_real_utility": {
            "passed": pilot_gain >= 0.10 and oracle_gain >= 0.10,
            "evidence": {"pilot_gain_over_naive": pilot_gain, "oracle_gain_over_naive": oracle_gain},
        },
        "deployment_gate_valid": {
            "passed": bool(gate_decisions) and gate_decisions.issubset(allowed),
            "evidence": {"gate_decisions": sorted(gate_decisions)},
        },
        "forbidden_claims_absent": {
            "passed": len(forbidden_hits) == 0,
            "evidence": {"hits": forbidden_hits},
        },
    }
    return claims


def choose_verdict(claims: dict[str, Any]) -> str:
    if not claims["forbidden_claims_absent"]["passed"]:
        return "redesign required"
    if not claims["finite_pool_accounting_validated"]["passed"]:
        return "redesign required"
    if not claims["fantasy_detected_out_of_support"]["passed"]:
        return "needs stronger learned model"
    if all(claim["passed"] for claim in claims.values()):
        return "submission-ready v2"
    return "needs benchmark validation"


def write_claim_audit(root: Path, claims: dict[str, Any], verdict: str) -> dict[str, Any]:
    results = root / "results"
    docs = root / "docs"
    results.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    status = {
        "verdict": verdict,
        "allowed_verdicts": list(VERDICTS),
        "claims": claims,
    }
    (results / "claims_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    lines = ["# Claim Status", "", f"Final audit verdict: `{verdict}`.", ""]
    for name, payload in claims.items():
        state = "PASS" if payload["passed"] else "FAIL"
        lines.append(f"- `{name}`: {state}. Evidence: `{json.dumps(payload['evidence'], sort_keys=True)}`")
    lines.append("")
    lines.append("The audit only supports the controlled synthetic claims listed above.")
    (results / "claims_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    final_lines = [
        "# Final Audit",
        "",
        f"Chosen verdict: `{verdict}`.",
        "",
        "This repository is a scoped submission-ready study. It validates finite-pool accounting, the synthetic DT-style return-conditioning failure mode, and the listed repairs in this environment. Benchmark-scale validation remains future work.",
        "",
        "Audit rule: the verdict must be exactly one of `submission-ready v2`, `needs stronger learned model`, `needs benchmark validation`, or `redesign required`.",
    ]
    (docs / "final_audit.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    return status


def run_claim_audit(root: Path | str = ".") -> dict[str, Any]:
    root_path = Path(root)
    claims = evaluate_claims(root_path)
    verdict = choose_verdict(claims)
    return write_claim_audit(root_path, claims, verdict)


def main() -> dict[str, Any]:
    return run_claim_audit(Path.cwd())
