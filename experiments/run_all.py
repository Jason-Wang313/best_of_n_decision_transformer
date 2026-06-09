from pathlib import Path

from decision_transformer_best_of_n.config import ExperimentConfig
from decision_transformer_best_of_n.experiments import run_experiment


if __name__ == "__main__":
    summary = run_experiment(ExperimentConfig.full(Path.cwd()))
    print(f"full run complete: {summary['audit_verdict']}")

