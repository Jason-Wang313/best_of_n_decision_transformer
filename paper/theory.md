# Theory

Source of truth: `paper/sections/theory.tex`, with proof details in `paper/sections/appendix.tex`.

The manuscript states the tie-aware finite-N law for grouped score levels:

`E[R_BoN(N)] = sum_g (F_g^N - F_{g-1}^N) mu_g`.

The section emphasizes that Best-of-N shifts mass toward the high-score tail. Whether real utility improves is determined by the conditional tail means `mu_g = E[R | S = z_g]`, not by `N` alone.
