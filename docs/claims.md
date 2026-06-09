# Claims

The claim audit in `results/claims_status.*` is the source of truth for which claims are currently supported.

## Supported Claims

- The exact finite-N tie-aware law matches Monte Carlo estimates in the generated candidate pool.
- Under the out-of-support target in this controlled environment, naive Best-of-N increases the selected proxy return while real utility does not track that increase.
- Under an in-support target, naive Best-of-N is a negative control: it does not show the same failure pattern.
- An anti-aligned scorer is a negative control showing that the tail relation between score and utility matters.
- The repair ladder improves real utility for the controlled out-of-support case, with the oracle selector included only as a diagnostic upper bound.
- The deployment gate returns exactly one of `allow_high_n`, `lower_target_return`, `collect_pilot_labels`, or `block_high_n`.

## Explicit Non-Claims

- This study does not establish a general solution for offline RL.
- It does not claim every use of Best-of-N is harmful.
- It does not claim calibration is universally sufficient.
- It does not replace benchmark validation or real deployment labels.

