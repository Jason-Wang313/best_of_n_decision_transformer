# Return-Support Decision Transformer Audits

This repository supports the controlled study **When High Returns Lie: Return-Support Audits for Decision-Transformer Prompts**.

The project studies a specific failure mode for return-conditioned sequence policies: when target returns move beyond offline support, larger candidate pools can improve the internal/proxy score `S` or prompt satisfaction while real utility `R` stalls or drops. The repo contains a small learned DT-style model, finite-pool accounting, synthetic offline data with limited high-return support, repair diagnostics, figures, tests, and an explicit claim audit.

## Quick Start

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
python -m experiments.run_cartpole_benchmark
bash scripts/run_claim_audit.sh
pytest
```

The Bash scripts are written to work on this Windows checkout as well as conventional Unix-style Python setups.

## Main Outputs

- `figures/return_conditioning_fantasy.*`
- `figures/repair_comparison.*`
- `figures/target_return_sweep.*`
- `figures/support_diagnostics.*`
- `figures/exact_accounting_validation.*`
- `figures/figure6_candidate_count_256.png`
- `figures/figure7_target_overshoot.png`
- `figures/figure8_repair_ladder_256.png`
- `figures/figure9_pilot_noise_stress.png`
- `figures/figure10_cartpole_benchmark.png`
- `results/claims_status.md`
- `results/claims_status.json`
- `results/expansion/expanded_summary.csv`
- `results/expansion/claims.json`
- `results/cartpole_benchmark/cartpole_summary.csv`
- `results/cartpole_benchmark/claims.json`
- `docs/final_audit.md`
- `paper/final/best_of_n_decision_transformer-v4.pdf`

## What Is Implemented

- Synthetic offline trajectories with scarce high-return support.
- A small learned return-to-go-conditioned sequence policy in `TinyDecisionTransformer`.
- Candidate-count evaluation for `N = {1, 2, 4, 8, 16, 32, 64}` in the full run.
- Expanded candidate-count evaluation through `N = 256` with target-return overshoot sweeps, pilot-label sensitivity, noise stress, and a repair ladder at the high-N tail.
- A CPU-light CartPole-v1 benchmark tier with raw proxy selection, support filtering, conservative support penalties, behavior cloning, random, and oracle diagnostic selectors.
- Separate internal score/proxy return `S` and real utility `R`.
- Exact tie-aware finite-pool accounting plus Monte Carlo validation.
- Return Fantasy Microscope diagnostics: prompt satisfaction gap, support gap, and real utility gap.
- Tail Phase Diagram labels: helps, saturates, or hurts.
- DT-specific repair ladder: support-aware return calibration, behavior-likelihood constrained selection, conservative target-return gating, pilot real-utility calibration, and an oracle diagnostic upper bound.
- Claim audit with forbidden-overclaim scanning and a final PDF page-count gate.

## Scope

This is a controlled submission artifact. It is designed to make one mechanism measurable and reproducible, not to establish D4RL, MuJoCo, Atari, or deployment-scale offline RL dominance. The paper now includes a standard CartPole-v1 control benchmark tier while keeping larger Transformers, heavyweight offline-RL benchmark suites, and real deployment labels outside the claim boundary. The v4 acceptance standard is intentionally strict: the final manuscript must be 25 or more pages, the expanded experiment suite and CartPole benchmark must pass, and the audit must select `submission-ready v4`.
