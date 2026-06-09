# Differentiation From WAM And Prior Projects

This repo is not a general reward-model alignment benchmark and not a broad offline-RL benchmark suite. Its contribution is narrower:

- It studies return-conditioned sequence policies rather than generic response reranking.
- It separates prompt satisfaction/proxy return from real utility in a controlled environment.
- It gives an exact finite-N law for the selected utility under tie-aware Best-of-N sampling.
- It tests support-aware repairs that are specific to target-return conditioning.
- It ships a claim audit that fails unsupported paper claims instead of relying on prose alone.

Compared with prior local projects about Best-of-N or weighted alignment methods, this project centers the offline dataset support boundary. The key diagnostic question is not whether reranking can improve a score, but whether the selected high-score tail remains behavior-supported and utility-aligned when the DT prompt asks for high returns.

