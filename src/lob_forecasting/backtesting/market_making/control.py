"""Control-style quote optimiser.

At each decision we score a small grid of quote actions (sides x levels) by
expected risk-adjusted reward and pick the best one, after dropping any action
whose worst-case fills could breach the inventory limit. No RL here, just a
grid search over ex-ante quantities: the fitted fill-probability model, the
model's predicted markout / adverse / quantile width, and the risk weights.

The score for an action combines expected edge G, an inventory penalty on the
projected next inventory, and turnover/uncertainty/action penalties:

    J = G - lambda_inv * x_next^2 - lambda_turn * turnover
          - lambda_unc * uncertainty - lambda_act * (action != hold)
"""

from __future__ import annotations

import numpy as np

from .environment import MarketArrays
from .policies import Policy

_EPS = 1e-8

# action id -> submitted (side, level) quotes
CONTROL_ACTIONS: dict[int, tuple[tuple[str, int], ...]] = {
    0: (),
    1: (("bid", 1),),
    2: (("ask", 1),),
    3: (("bid", 1), ("ask", 1)),
    4: (("bid", 2),),
    5: (("ask", 2),),
    6: (("bid", 2), ("ask", 2)),
    7: (("bid", 1), ("ask", 2)),
    8: (("bid", 2), ("ask", 1)),
}


def _actions_for_levels(levels: list[int]) -> list[int]:
    """Restrict the action grid to those using only the configured levels."""
    allowed = set(levels)
    out = []
    for a, quotes in CONTROL_ACTIONS.items():
        if all(lv in allowed for _, lv in quotes):
            out.append(a)
    return out


class ControlQuoteOptimizer(Policy):
    """Grid-based control optimiser over quote actions."""

    name = "control_quote_optimizer"

    def __init__(self, fill_model=None, **params) -> None:
        super().__init__(**params)
        self.fill_model = fill_model
        self.quote_size = float(params.get("quote_size", 1.0))
        self.max_inventory = float(params.get("max_inventory", 1.0))
        self.horizon = int(params.get("horizon", 50))
        self.maker_fee = float(params.get("maker_fee", 0.0))
        self.lambda_inv = float(params.get("lambda_inv", 0.01))
        self.lambda_turn = float(params.get("lambda_turn", 0.001))
        self.lambda_adv = float(params.get("lambda_adv", 0.25))
        self.lambda_unc = float(params.get("lambda_unc", 0.0))
        self.lambda_act = float(params.get("lambda_act", 0.0))
        self.actions = _actions_for_levels(params.get("action_levels", [1, 2]))

    # level-adjusted execution quantities

    def _markout_adverse(self, arrays: MarketArrays, t: int, side: str, level: int) -> tuple[float, float]:
        m1 = arrays.signal(f"markout_{side}", t)
        a1 = arrays.signal(f"adv_{side}", t)
        if level == 1:
            return m1, max(0.0, a1)
        best = arrays.quote_price(side, 1, t)
        lvl = arrays.quote_price(side, level, t)
        dprice = abs(best - lvl) if np.isfinite(best) and np.isfinite(lvl) else 0.0
        # quoting deeper improves markout and reduces adverse selection by dprice
        return m1 + dprice, max(0.0, a1 - dprice)

    def _fill_prob(self, arrays: MarketArrays, t: int, side: str, level: int) -> float:
        if self.fill_model is None:
            return 0.0
        return self.fill_model.predict_one(arrays.fill_feature_row(t, side, level))

    def _uncertainty(self, arrays: MarketArrays, t: int) -> float:
        q05 = arrays.signal("pred_q05", t)
        q50 = arrays.signal("pred_q50", t)
        q95 = arrays.signal("pred_q95", t)
        width = q95 - q05
        if width <= 0:
            width = arrays.signal("interval_width", t)
        return width / (abs(q50) + _EPS)

    def _feasible(self, action_quotes, inventory: float) -> bool:
        q = self.quote_size
        n_bid = sum(1 for s, _ in action_quotes if s == "bid")
        n_ask = sum(1 for s, _ in action_quotes if s == "ask")
        # worst case: only bids fill (max long) or only asks fill (max short)
        if inventory + q * n_bid > self.max_inventory + 1e-9:
            return False
        if inventory - q * n_ask < -self.max_inventory - 1e-9:
            return False
        return True

    def score_actions(self, arrays: MarketArrays, t: int, inventory: float) -> dict[int, float]:
        q = self.quote_size
        unc = self._uncertainty(arrays, t)
        scores: dict[int, float] = {}
        for a in self.actions:
            quotes = CONTROL_ACTIONS[a]
            if not self._feasible(quotes, inventory):
                continue
            g = 0.0
            x_next = inventory
            turnover = 0.0
            u_pen = 0.0
            for side, level in quotes:
                p = self._fill_prob(arrays, t, side, level)
                m_adj, a_adj = self._markout_adverse(arrays, t, side, level)
                g += p * q * (m_adj - self.lambda_adv * a_adj - self.maker_fee)
                x_next += q * p if side == "bid" else -q * p
                turnover += q * p
                u_pen += p * unc
            j = (g - self.lambda_inv * x_next**2 - self.lambda_turn * turnover
                 - self.lambda_unc * u_pen - (self.lambda_act if a != 0 else 0.0))
            scores[a] = j
        return scores

    def act(self, arrays: MarketArrays, t: int, inventory: float):
        scores = self.score_actions(arrays, t, inventory)
        if not scores:
            return []  # nothing feasible -> quote neither
        best_a = max(scores, key=scores.get)
        return list(CONTROL_ACTIONS[best_a])
