"""Run the CPU-light CartPole-v1 return-support benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from return_support_audit.cartpole_benchmark import (  # noqa: E402
    claim_gates,
    make_figure,
    run_cartpole_benchmark,
    summarize,
    write_outputs,
)


def run(quick: bool = False, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path.cwd()
    artifact_dir = Path(output_dir) if output_dir is not None else root / "results" / "cartpole_benchmark"
    figure_path = (
        artifact_dir / "figures" / "figure10_cartpole_benchmark.png"
        if output_dir is not None
        else root / "figures" / "figure10_cartpole_benchmark.png"
    )
    trials = 12 if quick else 32
    rows, meta = run_cartpole_benchmark(trials=trials, seed=2026)
    summary = summarize(rows)
    make_figure(summary, figure_path)
    manifest = write_outputs(rows, meta, artifact_dir, figure_path)
    manifest_path = artifact_dir / "manifest.json"
    return {**manifest, "manifest": str(manifest_path)}


if __name__ == "__main__":
    payload = run(quick="--quick" in sys.argv)
    claims = json.loads(Path(payload["claims"]).read_text(encoding="utf-8"))
    print(f"CartPole-v1 benchmark: {claims['summary']}")
    print(f"all_passed={claims['all_passed']}")
    print(f"Manifest: {payload['manifest']}")
