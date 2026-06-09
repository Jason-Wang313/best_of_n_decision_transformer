# Reviewer Attacks

## Attack: The model is too small to be a Decision Transformer.

Response: Correct as a limitation. The repo uses a small learned return-to-go-conditioned sequence policy, not a benchmark-scale Transformer. The paper should frame this as a controlled mechanism study and reserve architecture-scale claims for future work.

## Attack: The environment is synthetic.

Response: Correct. The synthetic environment is chosen because the proxy/utility split and support boundary are measurable. The audit should not claim benchmark generality.

## Attack: The repairs are tuned to the toy environment.

Response: The repairs are diagnostics, not universal prescriptions. The pilot real-utility calibration uses a small set of labels to show how hidden utility can change selection. The oracle selector is clearly marked as an upper bound.

## Attack: Best-of-N can help in many settings.

Response: The in-support negative control is included precisely to show that the result is conditional. The finite-N law says the outcome depends on the score/utility tail relation.

## Attack: The support estimator is crude.

Response: Yes. It is deliberately simple and auditable: empirical return quantiles plus marginal action likelihood. Stronger density models are future work.

