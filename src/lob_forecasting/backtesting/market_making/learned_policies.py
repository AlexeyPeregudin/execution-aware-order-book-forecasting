"""A small learned contextual-bandit market-making policy (exploratory).

For each training/validation event we simulate every action independently under
the same fill model and record its realised reward. The target is the
best-reward action, and the policy is a multinomial classifier over the discrete
actions, trained with cross-entropy weighted by the reward gap. The bandit never
sees test rewards: only the train (and, for selection, validation) days are
passed in.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .environment import ACTION_NEITHER, ACTIONS, MarketArrays, MarketMakingEnv
from .policies import Policy


class ContextualBanditMM(Policy):
    name = "contextual_bandit_mm"

    def __init__(self, **params: float) -> None:
        super().__init__(**params)
        self.clf: LogisticRegression | None = None
        self.constant_action: int = ACTION_NEITHER
        self.n_features: int = 0

    def fit(self, train_examples: list[tuple[np.ndarray, np.ndarray]]) -> "ContextualBanditMM":
        """train_examples: list of (state_matrix, per_action_rewards) per day."""
        X_parts, y_parts, w_parts = [], [], []
        for states, rewards in train_examples:
            if states is None or len(states) == 0:
                continue
            best = rewards.argmax(axis=1)
            gap = rewards[np.arange(len(rewards)), best] - rewards.mean(axis=1)
            X_parts.append(states)
            y_parts.append(best)
            w_parts.append(np.maximum(gap, 0.0) + 1e-6)
        if not X_parts:
            self.constant_action = ACTION_NEITHER
            return self
        X = np.vstack(X_parts).astype("float64")
        y = np.concatenate(y_parts).astype("int64")
        w = np.concatenate(w_parts).astype("float64")
        self.n_features = X.shape[1]
        if len(np.unique(y)) < 2:
            self.constant_action = int(y[0])
            return self
        self.clf = LogisticRegression(max_iter=200, C=float(self.params.get("C", 1.0)))
        self.clf.fit(X, y, sample_weight=w)
        return self

    def act(self, arrays: MarketArrays, t: int, inventory: float) -> int:
        if self.clf is None or arrays.state_matrix is None:
            return self.constant_action
        x = arrays.state_matrix[t].reshape(1, -1)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x)
        return int(self.clf.predict(x)[0])


def build_training_examples(
    envs: list[MarketMakingEnv], states: list[np.ndarray], decision_interval: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pair each day's decision-event states with their per-action reward matrix."""
    examples: list[tuple[np.ndarray, np.ndarray]] = []
    for env, s in zip(envs, states):
        idx, rewards = env.per_action_rewards(decision_interval)
        sub = s[idx] if s is not None else None
        examples.append((sub, rewards))
    return examples
