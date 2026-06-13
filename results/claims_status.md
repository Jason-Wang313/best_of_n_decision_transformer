# Claim Status

Final audit verdict: `submission-ready v2`.

- `finite_pool_accounting_validated`: PASS. Evidence: `{"max_abs_error_real": 0.0053447338100474084}`
- `fantasy_detected_out_of_support`: PASS. Evidence: `{"proxy_gain_nmax_vs_n1": 0.16847673584024037, "real_gain_nmax_vs_n1": -0.44415727051562026}`
- `in_support_negative_control`: PASS. Evidence: `{"real_gain_nmax_vs_n1": 0.15442027723823415}`
- `anti_aligned_scorer_control`: PASS. Evidence: `{"real_gain_nmax_vs_n1": -0.18845364523758334}`
- `repair_ladder_improves_real_utility`: PASS. Evidence: `{"oracle_gain_over_naive": 1.0850698431040113, "pilot_gain_over_naive": 1.0087409919042}`
- `deployment_gate_valid`: PASS. Evidence: `{"gate_decisions": ["allow_candidate_sweep", "collect_pilot_labels"]}`
- `forbidden_claims_absent`: PASS. Evidence: `{"hits": []}`

The audit only supports the controlled synthetic claims listed above.
