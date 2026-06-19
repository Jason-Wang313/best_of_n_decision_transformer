# Final Audit

## Final Artifact and Provenance

- Paper: `best_of_n_decision_transformer-v4.pdf`
- Source folder: `C:\Users\wangz\best_of_n_decision_transformer`
- GitHub remote: `https://github.com/Jason-Wang313/best_of_n_decision_transformer.git`
- Repository PDF: `paper/final/best_of_n_decision_transformer-v4.pdf`
- Visible Desktop PDF: `C:\Users\wangz\OneDrive\Desktop\best_of_n_decision_transformer-v4.pdf`
- SHA256: `5DB0F4B3548DCEF1C6E463A9D2198E806165C3E387E7B15131FB0C2272C01515`
- Page count: 27
- Repo/Desktop hash match: yes
- Verified on: 2026-06-19

## Final Verification

```powershell
python -m compileall src experiments tests -q
python -m pytest -q
python -m experiments.run_claim_audit
powershell -ExecutionPolicy Bypass -File paper\build_submission.ps1 -DesktopCopy "C:\Users\wangz\OneDrive\Desktop\best_of_n_decision_transformer-v4.pdf"
rg -n "undefined|Citation.*undefined|Reference.*undefined|Rerun to get|Overfull|LaTeX Warning|Package natbib Warning" "build\latex\main.log"
pdfinfo "paper\final\best_of_n_decision_transformer-v4.pdf"
pdftoppm -png "paper\final\best_of_n_decision_transformer-v4.pdf" "tmp\pdfs\decision_transformer_v4\page"
```

Results:

- Compile check: passed.
- Unit tests: 15 passed.
- Claim audit: `submission-ready v4`.
- Final LaTeX log scan: no unresolved citations, unresolved references, rerun warnings, overfull boxes, or natbib warnings.
- PDF render: all 27 pages rendered.
- Visual QA: pages 1, 5, 8, 9, 12, 14, 18, 20, 24, and 27 inspected for title/abstract, main figures, benchmark/stress tables, references, appendix diagnostics, fresh-agent reproduction manifest, clipping, and readability.

Chosen verdict: `submission-ready v4`.

This repository is a scoped submission-ready v4 study. It validates finite-pool accounting, the synthetic DT-style return-conditioning failure mode, expanded N=256 stress tests, support-target sweeps, pilot-size ablations, noise stress, a CPU-light CartPole-v1 benchmark tier, and the listed repairs in this environment. It does not claim D4RL, MuJoCo, Atari, or deployment-scale dominance.

Audit rule: the verdict must be exactly one of `submission-ready v4`, `needs stronger learned model`, `needs benchmark validation`, or `redesign required`.
