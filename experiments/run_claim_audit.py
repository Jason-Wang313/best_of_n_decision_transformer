from decision_transformer_best_of_n.audit import run_claim_audit


if __name__ == "__main__":
    status = run_claim_audit(".")
    print(f"claim audit complete: {status['verdict']}")

