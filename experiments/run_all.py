from pathlib import Path

from return_support_audit.config import ExperimentConfig
from return_support_audit.experiments import run_experiment


if __name__ == "__main__":
    summary = run_experiment(ExperimentConfig.full(Path.cwd()))
    print(f"full run complete: {summary['audit_verdict']}")

