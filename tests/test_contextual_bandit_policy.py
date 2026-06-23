"""Tests for the learned contextual-bandit market-making policy."""

from __future__ import annotations

import numpy as np

from lob_forecasting.backtesting.market_making.environment import ACTION_BOTH, ACTION_NEITHER, ACTIONS
from lob_forecasting.backtesting.market_making.learned_policies import ContextualBanditMM


class _Arrays:
    """Minimal stand-in exposing a state_matrix for the bandit."""

    def __init__(self, states):
        self.state_matrix = states


def test_bandit_learns_dominant_best_action():
    rng = np.random.default_rng(0)
    n = 200
    states = rng.normal(size=(n, 4))
    rewards = rng.normal(size=(n, len(ACTIONS)))
    rewards[:, ACTION_BOTH] += 5.0  # action "both" is clearly best everywhere
    bandit = ContextualBanditMM().fit([(states, rewards)])
    arrays = _Arrays(states)
    chosen = [bandit.act(arrays, t, 0.0) for t in range(n)]
    assert np.mean(np.array(chosen) == ACTION_BOTH) > 0.8


def test_bandit_handles_single_class_target():
    n = 50
    states = np.zeros((n, 3))
    rewards = np.zeros((n, len(ACTIONS)))
    rewards[:, ACTION_NEITHER] = 1.0  # only NEITHER is ever best
    bandit = ContextualBanditMM().fit([(states, rewards)])
    assert bandit.act(_Arrays(states), 0, 0.0) == ACTION_NEITHER


def test_bandit_trains_only_on_supplied_examples():
    # the fit signature consumes train examples only; no test rewards are passed.
    rng = np.random.default_rng(1)
    train = [(rng.normal(size=(30, 4)), rng.normal(size=(30, len(ACTIONS))))]
    bandit = ContextualBanditMM().fit(train)
    # predicting on unseen states must work without ever having seen their rewards
    out = bandit.act(_Arrays(rng.normal(size=(5, 4))), 0, 0.0)
    assert out in ACTIONS


def test_empty_training_falls_back_to_constant():
    bandit = ContextualBanditMM().fit([])
    assert bandit.act(_Arrays(np.zeros((3, 4))), 0, 0.0) == ACTION_NEITHER
