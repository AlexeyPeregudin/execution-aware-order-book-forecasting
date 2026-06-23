"""The shared prediction format every model writes.

Because all models use the same columns, evaluation and the backtest don't need
to know anything about how a model works. Classification-only models leave
pred_return null; regression-only models leave the class columns null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# execution-aware extension columns (nullable; only the multi-task model fills them)
EXEC_PREDICTION_COLUMNS: tuple[str, ...] = (
    "pred_q05",
    "pred_q50",
    "pred_q95",
    "true_markout_bid",
    "true_markout_ask",
    "pred_markout_bid",
    "pred_markout_ask",
    "true_adverse_bid",
    "true_adverse_ask",
    "pred_adverse_bid",
    "pred_adverse_ask",
    "pred_interval_width",
    "pred_uncertainty_score",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model_name",
    "model_version",
    "split",
    "venue",
    "symbol",
    "event_id",
    "timestamp_exchange_ns",
    "horizon",
    "true_return",
    "true_direction",
    "pred_return",
    "pred_down",
    "pred_neutral",
    "pred_up",
    "pred_class",
    *EXEC_PREDICTION_COLUMNS,
    "pred_return_source",
    "prediction_available",
)

# how pred_return was produced; "" / NA means a plain model return (or none)
_STRING_COLS = ("run_id", "model_name", "model_version", "split", "venue", "symbol")
_INT_COLS = ("event_id", "timestamp_exchange_ns", "horizon")
_FLOAT_COLS = (
    "true_return",
    "true_direction",
    "pred_return",
    "pred_down",
    "pred_neutral",
    "pred_up",
    "pred_class",
    *EXEC_PREDICTION_COLUMNS,
)


def enforce_prediction_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder/retype df to the prediction schema."""
    out = df.copy()
    for col in PREDICTION_COLUMNS:
        if col not in out.columns:
            # nullable float columns default to NaN (the exec columns are often absent)
            out[col] = np.nan if col in _FLOAT_COLS else pd.NA
    out = out[list(PREDICTION_COLUMNS)]
    for col in _STRING_COLS:
        out[col] = out[col].astype("string")
    for col in _INT_COLS:
        out[col] = out[col].astype("int64")
    for col in _FLOAT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["pred_return_source"] = out["pred_return_source"].astype("string")
    out["prediction_available"] = out["prediction_available"].astype("bool")
    return out


def build_predictions(
    *,
    run_id: str,
    model_name: str,
    model_version: str,
    split: str,
    ids: pd.DataFrame,
    horizons: list[int],
    true_return: dict[int, np.ndarray],
    true_direction: dict[int, np.ndarray],
    pred_return: dict[int, np.ndarray] | None = None,
    pred_proba: dict[int, np.ndarray] | None = None,
    pred_class: dict[int, np.ndarray] | None = None,
    prediction_available: dict[int, np.ndarray] | None = None,
    exec_columns: dict[int, dict[str, np.ndarray]] | None = None,
    pred_return_source: str | None = None,
) -> pd.DataFrame:
    """Build the long prediction table (one row per event per horizon).

    ids has venue/symbol/event_id/timestamp_exchange_ns, one row per event.
    pred_proba, if given, maps horizon -> an (n, 3) array of [down, neutral, up].
    """
    n = len(ids)
    blocks: list[pd.DataFrame] = []
    for h in horizons:
        block = pd.DataFrame(
            {
                "run_id": run_id,
                "model_name": model_name,
                "model_version": model_version,
                "split": split,
                "venue": ids["venue"].to_numpy(),
                "symbol": ids["symbol"].to_numpy(),
                "event_id": ids["event_id"].to_numpy(),
                "timestamp_exchange_ns": ids["timestamp_exchange_ns"].to_numpy(),
                "horizon": h,
                "true_return": true_return[h],
                "true_direction": true_direction[h],
            }
        )
        block["pred_return"] = pred_return[h] if pred_return is not None else np.nan
        if pred_proba is not None:
            proba = np.asarray(pred_proba[h], dtype="float64")
            block["pred_down"] = proba[:, 0]
            block["pred_neutral"] = proba[:, 1]
            block["pred_up"] = proba[:, 2]
        else:
            block["pred_down"] = np.nan
            block["pred_neutral"] = np.nan
            block["pred_up"] = np.nan
        block["pred_class"] = pred_class[h] if pred_class is not None else np.nan
        h_exec = (exec_columns or {}).get(h, {})
        for col in EXEC_PREDICTION_COLUMNS:
            block[col] = h_exec[col] if col in h_exec else np.nan
        block["pred_return_source"] = pred_return_source if pred_return_source is not None else pd.NA
        if prediction_available is not None:
            block["prediction_available"] = prediction_available[h]
        else:
            block["prediction_available"] = np.ones(n, dtype=bool)
        blocks.append(block)

    combined = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    return enforce_prediction_schema(combined)


# event_id resets per monthly day, so the sidecar merge keys on the timestamp
_SIDECAR_KEYS = ["venue", "symbol", "split", "timestamp_exchange_ns", "horizon"]


def merge_ridge_sidecar(neural_preds: pd.DataFrame, ridge_preds: pd.DataFrame) -> pd.DataFrame:
    """Fill `pred_return` from a ridge model's predictions.

    Used by the `ridge_sidecar` return variant: the neural model emits NaN
    `pred_return` and tags `pred_return_source = "ridge_sidecar"`; here we join
    the ridge model's point return on (venue, symbol, split, timestamp, horizon)
    -- never event_id, which resets per monthly day -- and copy it in.
    """
    if neural_preds.empty or ridge_preds.empty:
        return neural_preds
    ridge = ridge_preds[[*_SIDECAR_KEYS, "pred_return"]].rename(
        columns={"pred_return": "_ridge_pred_return"}
    )
    merged = neural_preds.merge(ridge, on=_SIDECAR_KEYS, how="left")
    merged["pred_return"] = merged["_ridge_pred_return"]
    merged = merged.drop(columns=["_ridge_pred_return"])
    merged["pred_return_source"] = "ridge_sidecar"
    return enforce_prediction_schema(merged)
