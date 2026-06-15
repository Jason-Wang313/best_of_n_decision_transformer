import importlib.util
import json
from pathlib import Path

from return_support_audit.cartpole_benchmark import MAX_STEPS, X_THRESHOLD


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_cartpole_benchmark",
    ROOT / "experiments" / "run_cartpole_benchmark.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
run = MODULE.run


def test_cartpole_benchmark_quick_writes_passing_claims(tmp_path):
    manifest = run(quick=True, output_dir=tmp_path / "cartpole")

    assert MAX_STEPS == 500
    assert X_THRESHOLD == 2.4
    assert Path(manifest["summary"]).exists()
    assert Path(manifest["trials"]).exists()
    assert Path(manifest["claims"]).exists()
    assert Path(manifest["figure"]).exists()
    assert Path(manifest["manifest"]).exists()
    assert json.loads(Path(manifest["claims"]).read_text(encoding="utf-8"))["all_passed"]
