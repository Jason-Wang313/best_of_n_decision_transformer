from return_support_audit.metrics import phase_label, return_fantasy_microscope


def test_return_fantasy_microscope_gaps():
    microscope = return_fantasy_microscope(
        target_return=10.0,
        selected_proxy_return=8.5,
        selected_real_utility=4.0,
        support_boundary=7.0,
        baseline_real_utility=5.0,
    )

    assert microscope.prompt_satisfaction_gap == 1.5
    assert microscope.support_gap == 3.0
    assert microscope.real_utility_gap == -1.0


def test_phase_labels():
    assert phase_label([1, 2, 4], [1.0, 1.4, 1.8], [1.0, 1.3, 1.6]) == "helps"
    assert phase_label([1, 2, 4], [1.0, 1.4, 1.8], [1.0, 0.98, 0.95]) == "saturates"
    assert phase_label([1, 2, 4], [1.0, 1.4, 1.8], [1.0, 0.8, 0.5]) == "hurts"

