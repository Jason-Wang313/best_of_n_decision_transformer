# Method

We generate offline trajectories from a safe behavior policy with limited high-return support. A small learned sequence policy conditions each action on state, previous action, time, and desired return-to-go. At evaluation time, the model rolls out candidates for a requested target return.

For each candidate we record:

- `S`: internal score/proxy return.
- `R`: real utility with a convex off-support penalty.
- behavior likelihood under a simple support estimator.
- prompt satisfaction gap, support gap, and real utility gap.

Strategies include naive Best-of-N, support-calibrated target returns, behavior-likelihood constrained selection, conservative target gating, pilot-calibrated selection, and an oracle real-utility diagnostic.

