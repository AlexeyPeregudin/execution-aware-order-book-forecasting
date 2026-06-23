"""The offline market-making replay environment.

For one monthly day it holds the best-level book path, the forecast/feature
signals used by the policies, and a numeric state matrix used by the learned
policy. `simulate_day` replays the day: at each decision event the policy quotes
bid/ask/both/neither, fills are resolved by the conservative fill model, and
inventory, cash, wealth and reward are accounted. Inventory never carries across
monthly days (each day is simulated from flat), per the calendar rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .accounting import Portfolio
from .fills import FILL_ASSUMPTION_VERSION, resolve_fill
from .queue_fill import QUEUE_AWARE_FILL_VERSION, resolve_queue_fill

ACTION_NEITHER = 0
ACTION_BID = 1
ACTION_ASK = 2
ACTION_BOTH = 3
ACTION_NAMES = {0: "neither", 1: "bid_only", 2: "ask_only", 3: "both"}
ACTIONS = (ACTION_NEITHER, ACTION_BID, ACTION_ASK, ACTION_BOTH)

# legacy integer action -> list of (side, level) quotes (all at level 1)
_LEGACY_QUOTES: dict[int, tuple[tuple[str, int], ...]] = {
    ACTION_NEITHER: (),
    ACTION_BID: (("bid", 1),),
    ACTION_ASK: (("ask", 1),),
    ACTION_BOTH: (("bid", 1), ("ask", 1)),
}


def action_to_quotes(action) -> list[tuple[str, int]]:
    """Normalise a policy action into a list of (side, level) quotes.

    Accepts the legacy integer actions (0-3, level-1 quotes) or, for the control
    optimiser, an explicit iterable of `(side, level)` pairs.
    """
    if isinstance(action, (int, np.integer)):
        return list(_LEGACY_QUOTES.get(int(action), ()))
    return [(str(s), int(lv)) for s, lv in action]


def _action_label(quotes: list[tuple[str, int]]) -> str:
    if not quotes:
        return "neither"
    return "+".join(f"{s}{lv}" for s, lv in quotes)


@dataclass
class MarketArrays:
    """Per-day arrays the simulator and policies consume."""

    mid: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    event_id: np.ndarray
    timestamp: np.ndarray
    tick_size: float
    monthly_date: str
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    state_matrix: np.ndarray | None = None  # (n, S) static state, inventory appended at runtime
    state_names: list[str] = field(default_factory=list)
    # top-K level prices/sizes (n, K), level 1 in column 0; used by the
    # queue-aware partial-fill model and the control optimiser's quote levels.
    bid_px_levels: np.ndarray | None = None
    bid_qty_levels: np.ndarray | None = None
    ask_px_levels: np.ndarray | None = None
    ask_qty_levels: np.ndarray | None = None
    # market-only feature matrix (no forecasts) for the ex-ante fill-prob model
    fill_features: np.ndarray | None = None
    fill_feature_names: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.mid)

    def displayed_qty(self, side: str, level: int, t: int) -> float:
        """Displayed size at `side` `level` at event `t` (0 if absent)."""
        col = level - 1
        arr = self.bid_qty_levels if side == "bid" else self.ask_qty_levels
        if arr is None or col >= arr.shape[1]:
            return 0.0
        v = arr[t, col]
        return float(v) if np.isfinite(v) else 0.0

    def fill_feature_row(self, t: int, side: str, level: int) -> np.ndarray:
        """Feature vector for the ex-ante fill-probability model at (t, side, level).

        Market features at `t` (no forecasts) augmented with the quote side,
        quote level and displayed depth at the quoted level.
        """
        base = self.fill_features[t] if self.fill_features is not None else np.zeros(0)
        is_bid = 1.0 if side == "bid" else 0.0
        depth = self.displayed_qty(side, level, t)
        return np.concatenate([np.nan_to_num(base), [is_bid, float(level), np.log1p(depth)]])

    def has_levels(self) -> bool:
        return self.bid_px_levels is not None and self.ask_px_levels is not None

    def quote_price(self, side: str, level: int, t: int) -> float:
        """Displayed price for `side` at `level` (1-based) at event `t`."""
        col = level - 1
        if side == "bid":
            arr = self.bid_px_levels if self.has_levels() else None
            return float(arr[t, col]) if arr is not None and col < arr.shape[1] else float(self.bid[t])
        arr = self.ask_px_levels if self.has_levels() else None
        return float(arr[t, col]) if arr is not None and col < arr.shape[1] else float(self.ask[t])

    def signal(self, name: str, t: int) -> float:
        arr = self.signals.get(name)
        if arr is None:
            return 0.0
        v = arr[t]
        return float(v) if np.isfinite(v) else 0.0

    def state_row(self, t: int, inventory: float) -> np.ndarray:
        if self.state_matrix is None:
            return np.array([inventory], dtype="float32")
        return np.concatenate([self.state_matrix[t], [np.float32(inventory)]])


@dataclass
class _RewardWeights:
    lambda_inv: float
    lambda_turn: float
    lambda_dd: float
    lambda_adv: float


@dataclass
class FillOutcome:
    """Unified result of resolving one quote under either fill model."""

    filled: bool
    fill_index: int
    fill_qty: float
    fill_fraction: float
    queue_ahead: float
    cum_depletion: float
    reason: str
    fill_model_name: str
    version: str


def resolve_fill_outcome(
    arrays: MarketArrays, side: str, level: int, t: int, horizon: int, quote_price: float,
    quote_size: float, fill_model: str, queue_kwargs: dict | None = None,
) -> FillOutcome:
    """Resolve a resting quote, dispatching on the configured fill model.

    The queue-aware model produces a (possibly partial) fill quantity and queue
    diagnostics; the legacy models fill the whole quote size or not at all.
    """
    if fill_model == "queue_aware_partial" and arrays.has_levels():
        qk = queue_kwargs or {}
        r = resolve_queue_fill(
            side, level, t, horizon,
            bid_px=arrays.bid_px_levels, bid_qty=arrays.bid_qty_levels,
            ask_px=arrays.ask_px_levels, ask_qty=arrays.ask_qty_levels,
            order_size=quote_size, tick=arrays.tick_size,
            queue_position=qk.get("queue_position", "back"),
            kappa=float(qk.get("kappa", 0.5)),
            full_cross_fill=bool(qk.get("full_cross_fill", True)),
        )
        return FillOutcome(r.filled, r.fill_index, r.fill_qty, r.fill_fraction, r.queue_ahead,
                           r.cum_effective_depletion, r.reason, fill_model, QUEUE_AWARE_FILL_VERSION)
    filled, fidx = resolve_fill(arrays.mid, arrays.bid, arrays.ask, t, horizon, quote_price, side, fill_model)
    return FillOutcome(filled, fidx, quote_size if filled else 0.0, 1.0 if filled else 0.0,
                       float("nan"), float("nan"), "full_cross" if filled else "expired",
                       fill_model, FILL_ASSUMPTION_VERSION)


def simulate_day(
    policy,
    arrays: MarketArrays,
    *,
    horizon: int,
    quote_size: float,
    distance_ticks: int,
    max_inventory: float,
    maker_fee_rate: float,
    reward_weights: _RewardWeights,
    fill_model: str,
    fold_id: int,
    model_name: str,
    policy_name: str,
    policy_params: dict | None = None,
    decision_interval: int = 1,
    queue_kwargs: dict | None = None,
    fill_model_params_hash: str = "",
) -> dict[str, list[dict]]:
    """Replay one monthly day with a policy; return order/fill/inventory/reward rows.

    The policy acts only every `decision_interval` events (a market maker requotes
    periodically, not on every micro-update), which keeps the replay tractable on
    high-frequency real data while inventory and wealth still carry between
    decisions.
    """
    n = arrays.n
    port = Portfolio(max_inventory=max_inventory, maker_fee_rate=maker_fee_rate)
    orders: list[dict] = []
    fills: list[dict] = []
    inventory_path: list[dict] = []
    rewards: list[float] = []

    tick = arrays.tick_size
    w_prev = 0.0
    prev_inventory = 0.0
    params_json = str(policy_params or {})
    base = {
        "fold_id": fold_id, "monthly_date": arrays.monthly_date,
        "model_name": model_name, "policy_name": policy_name,
    }

    for t in range(0, n, max(1, decision_interval)):
        mid_t = arrays.mid[t]
        if not np.isfinite(mid_t):
            continue

        raw_action = policy.act(arrays, t, port.inventory)
        quotes = action_to_quotes(raw_action)  # list of (side, level)
        quote_bid = any(s == "bid" for s, _ in quotes)
        quote_ask = any(s == "ask" for s, _ in quotes)
        bid_quote_price = float("nan")
        ask_quote_price = float("nan")

        inv_before = port.inventory
        realized_adverse = 0.0

        for side, level in quotes:
            qprice = arrays.quote_price(side, level, t)
            if level == 1 and distance_ticks:
                qprice += -distance_ticks * tick if side == "bid" else distance_ticks * tick
            if not np.isfinite(qprice):
                continue
            if side == "bid":
                bid_quote_price = float(qprice)
            else:
                ask_quote_price = float(qprice)
            extra = {"quote_level": int(level), "fill_model_name": fill_model,
                     "fill_model_params_hash": fill_model_params_hash}
            if not port.can_fill(side, quote_size):
                continue
            outcome = resolve_fill_outcome(arrays, side, level, t, horizon, float(qprice),
                                           quote_size, fill_model, queue_kwargs)
            if not outcome.filled or outcome.fill_qty <= 0:
                fills.append({**base, "decision_event_id": int(arrays.event_id[t]),
                              "fill_event_id": -1, "side": side, "quote_price": float(qprice),
                              "fill_price": float("nan"), "fill_occurred": False,
                              "fill_latency_events": -1, "mark_mid_horizon": float("nan"),
                              "gross_pnl": 0.0, "fee": 0.0, "net_pnl": 0.0,
                              "adverse_selection_cost": 0.0,
                              "fill_model": fill_model, "fill_assumption_version": outcome.version,
                              "queue_position": (queue_kwargs or {}).get("queue_position") if fill_model == "queue_aware_partial" else None,
                              "queue_ahead": outcome.queue_ahead, "cum_effective_depletion": outcome.cum_depletion,
                              "fill_fraction": 0.0, "fill_reason": outcome.reason, **extra})
                continue
            fill_qty = float(outcome.fill_qty)
            applied = port.apply_fill(side, fill_qty, float(qprice))
            if not applied:
                continue
            fidx = outcome.fill_index
            mark_idx = min(fidx + horizon, n - 1)
            mark_mid = float(arrays.mid[mark_idx])
            if side == "bid":
                adverse = max(0.0, float(qprice) - mark_mid) * fill_qty
                gross = (mid_t - float(qprice)) * fill_qty
            else:
                adverse = max(0.0, mark_mid - float(qprice)) * fill_qty
                gross = (float(qprice) - mid_t) * fill_qty
            realized_adverse += adverse
            fee = maker_fee_rate * float(qprice) * fill_qty
            fills.append({**base, "decision_event_id": int(arrays.event_id[t]),
                          "fill_event_id": int(arrays.event_id[fidx]), "side": side,
                          "quote_price": float(qprice), "fill_price": float(qprice),
                          "fill_occurred": True, "fill_latency_events": int(fidx - t),
                          "mark_mid_horizon": mark_mid, "gross_pnl": float(gross),
                          "fee": float(fee), "net_pnl": float(gross - fee),
                          "adverse_selection_cost": float(adverse),
                          "fill_model": fill_model, "fill_assumption_version": outcome.version,
                          "queue_position": (queue_kwargs or {}).get("queue_position") if fill_model == "queue_aware_partial" else None,
                          "queue_ahead": outcome.queue_ahead, "cum_effective_depletion": outcome.cum_depletion,
                          "fill_fraction": float(outcome.fill_fraction), "fill_reason": outcome.reason, **extra})

        # marked wealth change since the previous decision (inventory carry + fill edge)
        w_after = port.wealth(mid_t)
        d_wealth = w_after - w_prev
        d_inv = port.inventory - prev_inventory
        dd = port.drawdown(mid_t)
        rw = reward_weights
        reward = (
            d_wealth
            - rw.lambda_inv * port.inventory**2
            - rw.lambda_turn * abs(d_inv)
            - rw.lambda_dd * dd
            - rw.lambda_adv * realized_adverse
        )
        port.mark(mid_t)
        rewards.append(reward)

        orders.append({**base, "event_id": int(arrays.event_id[t]),
                       "timestamp_exchange_ns": int(arrays.timestamp[t]),
                       "action": _action_label(quotes), "quote_bid": quote_bid, "quote_ask": quote_ask,
                       "bid_quote_price": bid_quote_price if quote_bid else float("nan"),
                       "ask_quote_price": ask_quote_price if quote_ask else float("nan"),
                       "bid_distance_ticks": distance_ticks if quote_bid else -1,
                       "ask_distance_ticks": distance_ticks if quote_ask else -1,
                       "inventory_before": float(inv_before), "inventory_after_action": float(port.inventory),
                       "reward": float(reward), "policy_params_json": params_json})
        inventory_path.append({**base, "event_id": int(arrays.event_id[t]),
                               "timestamp_exchange_ns": int(arrays.timestamp[t]),
                               "cash": float(port.cash), "inventory": float(port.inventory),
                               "mid": float(mid_t), "wealth": float(w_after), "drawdown": float(dd)})
        w_prev = w_after
        prev_inventory = port.inventory

    return {"orders": orders, "fills": fills, "inventory": inventory_path,
            "rewards": rewards, "n_inventory_rejects": port.n_inventory_rejects}


class MarketMakingEnv:
    """Thin wrapper exposing per-action reward simulation for the contextual bandit.

    Given a day's arrays it can replay a fixed action at every event under the
    same fill model, which the learned policy uses to build reward-labelled
    targets without ever touching test rewards.
    """

    def __init__(self, arrays: MarketArrays, *, horizon: int, quote_size: float,
                 distance_ticks: int, max_inventory: float, maker_fee_rate: float,
                 reward_weights: _RewardWeights, fill_model: str,
                 decision_interval: int = 1) -> None:
        self.arrays = arrays
        self.decision_interval = decision_interval
        self.kw = dict(horizon=horizon, quote_size=quote_size, distance_ticks=distance_ticks,
                       max_inventory=max_inventory, maker_fee_rate=maker_fee_rate,
                       reward_weights=reward_weights, fill_model=fill_model)

    def per_action_rewards(self, decision_interval: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Reward of each fixed action at the decision events, inventory held flat.

        Each action is evaluated independently from flat inventory at that event,
        so the targets isolate the immediate decision value.
        Returns (decision_indices, rewards[len(idx), |A|]).
        """
        a = self.arrays
        n = a.n
        idx = np.arange(0, n, max(1, decision_interval), dtype="int64")
        out = np.zeros((len(idx), len(ACTIONS)), dtype="float64")
        h = self.kw["horizon"]
        q = self.kw["quote_size"]
        d = self.kw["distance_ticks"]
        rw: _RewardWeights = self.kw["reward_weights"]
        fm = self.kw["fill_model"]
        tick = a.tick_size
        for row, t in enumerate(idx):
            mid_t = a.mid[t]
            if not np.isfinite(mid_t):
                continue
            bid_price = a.bid[t] - d * tick
            ask_price = a.ask[t] + d * tick
            edge = {}
            adv = {}
            for side, qprice in (("bid", bid_price), ("ask", ask_price)):
                filled, fidx = resolve_fill(a.mid, a.bid, a.ask, t, h, float(qprice), side, fm)
                if filled and np.isfinite(qprice):
                    mark = a.mid[min(fidx + h, n - 1)]
                    if side == "bid":
                        edge["bid"] = (mid_t - qprice) * q
                        adv["bid"] = max(0.0, qprice - mark)
                    else:
                        edge["ask"] = (qprice - mid_t) * q
                        adv["ask"] = max(0.0, mark - qprice)
            for ai, action in enumerate(ACTIONS):
                e = 0.0
                av = 0.0
                inv = 0.0
                if action in (ACTION_BID, ACTION_BOTH) and "bid" in edge:
                    e += edge["bid"]
                    av += adv["bid"]
                    inv += q
                if action in (ACTION_ASK, ACTION_BOTH) and "ask" in edge:
                    e += edge["ask"]
                    av += adv["ask"]
                    inv -= q
                out[row, ai] = e - rw.lambda_inv * inv**2 - rw.lambda_turn * abs(inv) - rw.lambda_adv * av
        return idx, out
