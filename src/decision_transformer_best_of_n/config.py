"""Experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Small, reproducible configuration for the synthetic study."""

    seed: int = 313
    horizon: int = 12
    n_trajectories: int = 600
    n_replicates: int = 180
    n_pilot: int = 24
    n_values: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    safe_action: float = 0.72
    action_clip: float = 1.65
    model_noise_scale: float = 0.32
    ridge: float = 1e-3
    output_dir: Path = Path(".")
    mode: str = "full"
    figure_formats: tuple[str, ...] = ("png", "pdf")
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def smoke(cls, output_dir: Path | str = ".") -> "ExperimentConfig":
        return cls(
            seed=313,
            n_trajectories=96,
            n_replicates=8,
            n_pilot=8,
            n_values=(1, 4, 16, 64),
            output_dir=Path(output_dir),
            mode="smoke",
            figure_formats=("png",),
        )

    @classmethod
    def full(cls, output_dir: Path | str = ".") -> "ExperimentConfig":
        return cls(
            n_trajectories=450,
            n_replicates=64,
            n_pilot=16,
            output_dir=Path(output_dir),
            mode="full",
        )

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["n_values"] = list(self.n_values)
        payload["figure_formats"] = list(self.figure_formats)
        return payload
