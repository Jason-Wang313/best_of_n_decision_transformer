# Reviewer Attacks

## Attack: The model is too small to be a Decision Transformer.

Response: Correct as a limitation. The repo uses a small learned return-to-go-conditioned sequence policy, not a benchmark-scale Transformer. The paper frames this as a controlled mechanism study plus a CPU-light CartPole-v1 benchmark tier, and reserves architecture-scale claims for future work.

Strengthening: The manuscript treats this as a mechanism-identification paper rather than a benchmark-performance paper. The expanded suite asks whether the same DT-style return-conditioning interface fails under larger candidate pools, target overshoot, score noise, limited pilot labels, and standard CartPole dynamics.

## Attack: The environment is synthetic.

Response: Partly correct. The synthetic environment is still the main microscope because the proxy/utility split and support boundary are exactly measurable. The added CartPole-v1 tier checks whether the same failure and repair pattern survives a recognized control benchmark without claiming D4RL-scale generality.

Strengthening: The expanded experiment suite increases candidate counts to 256, sweeps target-return gaps, adds noise stress, records exact CSV/JSON evidence, and adds a CartPole-v1 claim gate with raw, support-aware, conservative, behavior-cloning, random, and oracle diagnostic selectors.

## Attack: The repairs are tuned to the toy environment.

Response: The repairs are diagnostics, not universal prescriptions. The pilot real-utility calibration uses a small set of labels to show how hidden utility can change selection. The oracle selector is clearly marked as an upper bound.

Strengthening: The high-N repair ladder separates naive selection, proxy quantile truncation, behavior-likelihood filters, pilot-calibrated selection, and the oracle upper bound so reviewers can see which part of the repair actually changes selected utility. The CartPole tier adds the same selector separation under external dynamics.

## Attack: Candidate-count selection can help in many settings.

Response: The in-support negative control is included precisely to show that the result is conditional. The finite-pool accounting says the outcome depends on the score/utility tail relation.

Strengthening: The paper reports both the original candidate sweep and the expanded `N = 256` tail, making clear that the failure is not "larger N is bad" but "larger N amplifies whatever score-utility relation lives in the selected tail." CartPole-v1 repeats this as a raw proxy versus true episode-return split.

## Attack: The support estimator is crude.

Response: Yes. It is deliberately simple and auditable: empirical return quantiles plus marginal action likelihood. Stronger density models are future work.

Strengthening: The support estimator is stress-tested through target overshoot, behavior-likelihood thresholds, pilot-label sizes, score-noise scale, and CartPole action-imbalance support. The manuscript describes what the estimator detects, what it misses, and why stronger density models are outside the present claim.

## Attack: This looks like another wrapper around a generic best-of-N theorem.

Response: The paper has been refactored around the Decision Transformer-specific object: return-to-go prompting outside offline support. The finite-pool identity is only the accounting layer; the empirical contribution is the measured interaction among target returns, behavior support, selected proxy return, and selected real utility.

Strengthening: The final manuscript foregrounds return-support audits, target overshoot, repair ladders, pilot real-utility labels, and a CartPole-v1 benchmark gate. Those are specific to the DT setting and should remain distinguishable when placed next to other candidate-selection papers.
