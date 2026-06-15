from pathlib import Path

from experiments.run_cartpole_benchmark import run as run_cartpole_benchmark
from experiments.run_expansion_suite import run as run_expansion_suite
from return_support_audit.audit import run_claim_audit
from return_support_audit.config import ExperimentConfig
from return_support_audit.experiments import run_experiment


if __name__ == "__main__":
    root = Path.cwd()
    summary = run_experiment(ExperimentConfig.full(root))
    expansion = run_expansion_suite(quick=False)
    cartpole = run_cartpole_benchmark(quick=False)
    audit = run_claim_audit(root)
    print(
        f"full run complete: {summary['audit_verdict']}; "
        f"expansion={expansion['all_passed']}; "
        f"cartpole={cartpole['all_passed']}; final={audit['verdict']}"
    )
