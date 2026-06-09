# Experiments

The full experiment evaluates `N = {1, 2, 4, 8, 16, 32, 64}` across in-support, edge-support, and out-of-support target returns. The primary figure, `return_conditioning_fantasy`, shows selected proxy return and real utility under the out-of-support prompt. `target_return_sweep` varies the target return across and beyond the offline support boundary. `repair_comparison` compares the repair ladder at high `N`. `exact_law_validation` compares the tie-aware law to Monte Carlo estimates.

The in-support and anti-aligned scorer controls are included to make the claim conditional rather than universal.

