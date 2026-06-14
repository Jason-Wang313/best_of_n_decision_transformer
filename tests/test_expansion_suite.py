import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_expansion_suite",
    ROOT / "experiments" / "run_expansion_suite.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
run = MODULE.run


def test_expansion_suite_quick_writes_passing_claims(tmp_path):
    manifest = run(quick=True, output_dir=tmp_path / "expansion")

    assert Path(manifest["summary"]).exists()
    assert Path(manifest["trials"]).exists()
    assert Path(manifest["claims"]).exists()
    assert Path(manifest["manifest"]).exists()
    assert json.loads(Path(manifest["claims"]).read_text(encoding="utf-8"))["all_passed"]
