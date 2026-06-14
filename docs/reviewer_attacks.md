# Reviewer Attacks

## Attack: The model is too small to be a Decision Transformer.

Response: Correct as a limitation. The repo uses a small learned return-to-go-conditioned sequence policy, not a benchmark-scale Transformer. The paper should frame this as a controlled mechanism study and reserve architecture-scale claims for future work.

v3 strengthening: The manuscript now treats this as a mechanism-identification paper rather than a benchmark-performance paper. The expanded suite asks whether the same DT-style return-conditioning interface fails under larger candidate pools, target overshoot, score noise, and limited pilot labels.

## Attack: The environment is synthetic.

Response: Correct. The synthetic environment is chosen because the proxy/utility split and support boundary are measurable. The audit should not claim benchmark generality.

v3 strengthening: The expanded experiment suite is deliberately larger without pretending to be a benchmark: it increases candidate counts to 256, sweeps target-return gaps, adds noise stress, and records exact CSV/JSON evidence for every claim.

## Attack: The repairs are tuned to the toy environment.

Response: The repairs are diagnostics, not universal prescriptions. The pilot real-utility calibration uses a small set of labels to show how hidden utility can change selection. The oracle selector is clearly marked as an upper bound.

v3 strengthening: The high-N repair ladder separates naive selection, proxy quantile truncation, behavior-likelihood filters, pilot-calibrated selection, and the oracle upper bound so reviewers can see which part of the repair actually changes selected utility.

## Attack: Candidate-count selection can help in many settings.

Response: The in-support negative control is included precisely to show that the result is conditional. The finite-pool accounting says the outcome depends on the score/utility tail relation.

v3 strengthening: The paper reports both the original candidate sweep and the expanded `N = 256` tail, making clear that the failure is not "larger N is bad" but "larger N amplifies whatever score-utility relation lives in the selected tail."

## Attack: The support estimator is crude.

Response: Yes. It is deliberately simple and auditable: empirical return quantiles plus marginal action likelihood. Stronger density models are future work.

v3 strengthening: The support estimator is stress-tested through target overshoot, behavior-likelihood thresholds, pilot-label sizes, and score-noise scale. The manuscript describes what the estimator detects, what it misses, and why stronger density models are outside the present claim.

## Attack: This looks like another wrapper around a generic best-of-N theorem.

Response: The paper has been refactored around the Decision Transformer-specific object: return-to-go prompting outside offline support. The finite-pool identity is only the accounting layer; the empirical contribution is the measured interaction among target returns, behavior support, selected proxy return, and selected real utility.

v3 strengthening: The final manuscript foregrounds return-support audits, target overshoot, repair ladders, and pilot real-utility labels. Those are specific to the DT setting and should remain distinguishable when placed next to other candidate-selection papers.
