from pathlib import Path

from decision_transformer_best_of_n.audit import scan_forbidden_claims


def test_forbidden_claim_scanner(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "paper").mkdir()
    (tmp_path / "README.md").write_text("A careful controlled study.", encoding="utf-8")
    assert scan_forbidden_claims(tmp_path) == []

    (tmp_path / "paper" / "bad.md").write_text("Best-of-N always hurts.", encoding="utf-8")
    hits = scan_forbidden_claims(tmp_path)
    assert hits and hits[0]["claim"] == "best-of-n always hurts"

