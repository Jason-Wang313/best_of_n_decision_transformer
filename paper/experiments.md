# Experiments

Source of truth: `paper/sections/experiments.tex`, included by `paper/main.tex`.

The LaTeX experiments section is organized around seven questions:

- Does naive top-score selection improve proxy return while harming real utility out of support?
- Does the exact accounting match Monte Carlo?
- Do in-support and anti-aligned controls behave as predicted?
- Do support, likelihood, gate, and pilot repairs reduce the failure?
- Does the same failure-and-repair pattern survive a CPU-light CartPole-v1 benchmark tier?
- Do pilot-label and rollout-noise stresses expose operational limits?
- Does the final claim audit gate all reported evidence?

It includes all required figures from `figures/` and tables with numbers from `results/selection_summary.csv`, `results/target_return_sweep.csv`, `results/exact_accounting_validation.csv`, `results/expansion/claims.json`, `results/cartpole_benchmark/claims.json`, and `results/claims_status.md`.
