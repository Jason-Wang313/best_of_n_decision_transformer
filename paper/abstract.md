# Abstract

Return-conditioned sequence policies invite a simple inference-time improvement: sample multiple trajectories and select the one with the highest predicted or proxy return. We study a controlled failure mode where this Best-of-N procedure improves prompt satisfaction while real utility fails to improve. The mechanism is support mismatch: high target returns push the policy into an extrapolative tail where the proxy return underprices risk. We provide an exact finite-N tie-aware selection law, a small learned Decision Transformer-style synthetic study, diagnostic figures, and a repair ladder based on support-aware calibration, behavior likelihood, conservative gating, and pilot real-utility labels.

