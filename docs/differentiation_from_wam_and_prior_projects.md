# Differentiation From WAM And Prior Projects

This repo is not a general reward-model alignment benchmark and not a broad offline-RL benchmark suite. Its contribution is narrower:

- It studies return-conditioned sequence policies rather than generic response reranking.
- It separates prompt satisfaction/proxy return from real utility in a controlled environment.
- It gives an exact finite-pool accounting identity for selected utility under tie-aware top-score sampling.
- It tests support-aware repairs that are specific to target-return conditioning.
- It adds a CPU-light CartPole-v1 benchmark tier while keeping D4RL, MuJoCo, Atari, and large-Transformer claims out of scope.
- It ships a claim audit that fails unsupported paper claims instead of relying on prose alone.

Compared with prior local projects about candidate reranking or weighted alignment methods, this project centers the offline dataset support boundary. The key diagnostic question is not whether reranking can improve a score, but whether the selected high-score tail remains behavior-supported and utility-aligned when the DT prompt asks for high returns.
