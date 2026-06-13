import numpy as np

from return_support_audit.data import generate_offline_dataset, rollout_from_actions
from return_support_audit.repairs import (
    GATE_DECISIONS,
    SupportEstimator,
    deployment_gate,
    make_candidate,
    select_behavior_likelihood_constrained,
    select_naive,
    select_pilot_calibrated,
)


def test_support_estimator_calibrates_target():
    batch = generate_offline_dataset(n_trajectories=80, seed=1)
    support = SupportEstimator.fit(batch)

    high_target = support.q95 + 10.0

    assert support.calibrate_target(high_target) == support.q95
    assert support.support_gap(high_target) > 0.0


def test_behavior_likelihood_constraint_prefers_feasible_candidate():
    batch = generate_offline_dataset(n_trajectories=80, seed=2)
    support = SupportEstimator.fit(batch)
    safe = make_candidate(rollout_from_actions([0.45] * 12, target_return=5.0), support)
    risky = make_candidate(rollout_from_actions([1.45] * 12, target_return=12.0), support)

    assert select_naive([safe, risky]) is risky
    assert select_behavior_likelihood_constrained([safe, risky], support) is safe


def test_pilot_calibration_penalizes_risk():
    batch = generate_offline_dataset(n_trajectories=80, seed=3)
    support = SupportEstimator.fit(batch)
    safe = make_candidate(rollout_from_actions([0.55] * 12, target_return=6.0), support)
    risky = make_candidate(rollout_from_actions([1.35] * 12, target_return=12.0), support)
    pilot = [
        make_candidate(rollout_from_actions([0.45] * 12, target_return=5.0), support),
        make_candidate(rollout_from_actions([0.65] * 12, target_return=7.0), support),
        make_candidate(rollout_from_actions([1.40] * 12, target_return=12.0), support),
        make_candidate(rollout_from_actions([1.55] * 12, target_return=13.0), support),
    ]

    assert select_pilot_calibrated([safe, risky], pilot) is safe


def test_deployment_gate_returns_exactly_one_allowed_value():
    batch = generate_offline_dataset(n_trajectories=80, seed=4)
    support = SupportEstimator.fit(batch)

    decisions = {
        deployment_gate(support.q80, support, 64),
        deployment_gate(support.q95 + 0.3, support, 64),
        deployment_gate(support.q95 + 3.0, support, 64),
        deployment_gate(support.q95 + 3.0, support, 64, pilot_delta_real=-0.5),
    }

    assert decisions.issubset(set(GATE_DECISIONS))
    assert len(decisions) >= 3


def test_action_log_likelihood_orders_safe_above_risky():
    batch = generate_offline_dataset(n_trajectories=80, seed=5)
    support = SupportEstimator.fit(batch)
    safe_logp = support.action_log_likelihood(np.array([0.50] * 12))
    risky_logp = support.action_log_likelihood(np.array([1.50] * 12))

    assert safe_logp > risky_logp

