import numpy as np

from decision_transformer_best_of_n.data import generate_offline_dataset
from decision_transformer_best_of_n.model import TinyDecisionTransformer
from decision_transformer_best_of_n.repairs import SupportEstimator, make_candidate


def test_tiny_dt_trains_and_rolls_out():
    batch = generate_offline_dataset(n_trajectories=96, seed=11)
    support = SupportEstimator.fit(batch)
    model = TinyDecisionTransformer(horizon=batch.horizon).fit(batch)

    rollout = model.rollout(target_return=support.q80, rng=np.random.default_rng(12))
    candidate = make_candidate(rollout, support)

    assert rollout.actions.shape == (batch.horizon,)
    assert np.isfinite(candidate.proxy)
    assert np.isfinite(candidate.real)
    assert np.isfinite(candidate.behavior_logp)

