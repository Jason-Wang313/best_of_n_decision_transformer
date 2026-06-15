# Theory

## Tie-Aware Top-Score Accounting

Let candidates be drawn iid from a finite distribution over pairs `(S, R)`, where `S` is the selection score and `R` is real utility. Top-score selection chooses an item with maximal `S`; if several sampled items tie for the maximal score, the winner is chosen uniformly among those tied items.

Group candidates by score level `s_g`. Let `q_g = P(S = s_g)`, `F_g = P(S <= s_g)`, and `u_g = E[R | S = s_g]`. For `N >= 1`, the expected selected real utility is:

```text
E[R_selected_N] = sum_g (F_g^N - F_{g-1}^N) u_g
```

The selected score expectation replaces `u_g` with `s_g`. This is exact for finite iid sampling with replacement and handles ties because, conditional on a winning score group, uniform tie-breaking gives the group-average utility.

## Why the Law Matters Here

The identity separates selection pressure from utility alignment. Increasing `N` shifts mass toward larger `S`. Whether real utility rises depends on the conditional tail relation between `S` and `R`. If the high-`S` tail is outside offline support and carries hidden risk, `E[S_selected_N]` can rise while `E[R_selected_N]` saturates or falls.

## DT-Specific Mechanism

Decision Transformer-style policies condition actions on desired return-to-go. When prompted above the support of offline returns, a learned policy can extrapolate toward action patterns that satisfy the proxy return signal. In this controlled environment, the proxy underprices risky actions, while real utility penalizes those actions convexly.

The expanded suite turns this mechanism into five measurable axes:

- Candidate-count pressure: increasing `N` moves selected samples toward the highest proxy tail.
- Target-return overshoot: increasing the prompt beyond the observed return frontier increases the chance that selected trajectories satisfy the prompt through unsupported actions.
- Behavior support: likelihood and empirical-return filters test whether the selected tail remains behavior-plausible.
- Pilot real utility: small labeled sets test whether direct utility feedback can correct proxy-tail selection.
- CartPole-v1 control: standard dynamics test whether the same proxy/support/return split survives outside the synthetic one-dimensional action process.

## Finite-Pool Variant

The repo also includes exact enumeration for small finite pools sampled without replacement. That function is used in tests to verify tie handling under a second sampling protocol.
