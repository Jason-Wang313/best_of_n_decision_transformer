from pathlib import Path

from return_support_audit.config import ExperimentConfig
from return_support_audit.experiments import run_experiment


if __name__ == "__main__":
    summary = run_experiment(ExperimentConfig.smoke(Path.cwd()))
    print(f"smoke complete: {summary['audit_verdict']}")

