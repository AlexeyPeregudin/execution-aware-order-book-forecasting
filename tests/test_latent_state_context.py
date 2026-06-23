"""Latent state-space context tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_forecasting.datasets import build_datasets, generate_folds
from lob_forecasting.features.latent_state import (
    fit_latent_state,
    filtered_context,
    latent_state_columns,
    load_ssm,
    save_ssm,
)
from lob_forecasting.models import load_model_data

from ._monthly_helpers import monthly_config, run_to_labels

OBS = ["imbalance_l1", "imbalance_lK", "ofi_10", "return_lag_10",
       "realised_vol_10", "relative_spread", "regime_depth"]


def _synthetic_blocks(n_days=3, T=200, P=7, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_days):
        z = np.zeros((T, 2))
        for t in range(1, T):
            z[t] = 0.9 * z[t - 1] + rng.normal(0, 0.3, 2)
        load = rng.normal(0, 1, (2, P))
        y = z @ load + rng.normal(0, 0.2, (T, P))
        blocks.append(y)
    return blocks


def test_filter_is_causal_no_future_leak():
    blocks = _synthetic_blocks()
    ssm = fit_latent_state(blocks, OBS, state_dim=4, max_em_iterations=5)
    Y = blocks[0].copy()
    out0 = ssm.filter(Y)
    t = 50
    Y2 = Y.copy()
    Y2[t + 5:] += 10.0  # perturb only the future
    out1 = ssm.filter(Y2)
    # filtered states up to and including t must be unchanged
    assert np.allclose(out0["z"][: t + 1], out1["z"][: t + 1])
    # and the future is affected (sanity: the perturbation actually matters)
    assert not np.allclose(out0["z"][t + 6:], out1["z"][t + 6:])


def test_filter_resets_each_monthly_day():
    blocks = _synthetic_blocks(n_days=2)
    ssm = fit_latent_state(blocks, OBS, state_dim=4, max_em_iterations=5)
    T = len(blocks[0])
    frame = pd.DataFrame(np.vstack(blocks), columns=OBS)
    frame["monthly_date"] = ["2025-07-01"] * T + ["2025-08-01"] * len(blocks[1])
    ctx = filtered_context(frame, ssm, day_column="monthly_date", reset_each_day=True)
    # day 2 filtered alone must match the reset-filtered day-2 slice
    day2_alone = ssm.filter(blocks[1])
    z_cols = [f"ssm_z_{i+1}" for i in range(4)]
    assert np.allclose(ctx.iloc[T:][z_cols].to_numpy(), day2_alone["z"])


def test_parameters_fit_on_supplied_blocks_only():
    blocks = _synthetic_blocks()
    ssm = fit_latent_state(blocks, OBS, state_dim=4, max_em_iterations=5)
    # observation standardisation reflects the training blocks
    stacked = np.vstack(blocks)
    assert np.allclose(ssm.obs_mean, stacked.mean(axis=0), atol=1e-6)
    assert ssm.A.shape == (4, 4) and ssm.C.shape == (7, 4)
    # diagonal A with stable spectral radius (<1)
    assert np.all(np.abs(np.diag(ssm.A)) < 1.0)


def test_save_load_reproduces_filtered_states(tmp_path):
    blocks = _synthetic_blocks()
    ssm = fit_latent_state(blocks, OBS, state_dim=4, max_em_iterations=5)
    out0 = ssm.filter(blocks[0])
    p = tmp_path / "ssm.pkl"
    save_ssm(ssm, p)
    out1 = load_ssm(p).filter(blocks[0])
    assert np.allclose(out0["z"], out1["z"])
    assert np.allclose(out0["var"], out1["var"])


def test_context_columns_match_state_dim():
    assert latent_state_columns(4) == [f"ssm_z_{i}" for i in range(1, 5)] + [f"ssm_var_{i}" for i in range(1, 5)]
    assert len(latent_state_columns(3)) == 6


@pytest.fixture(scope="module")
def ssm_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ssm_ds")
    cfg = monthly_config(latent_state={"enabled": True, "state_dim": 4, "observation_columns": OBS})
    labelled = run_to_labels(cfg, root)
    fold = generate_folds(cfg)[0]
    idx = build_datasets(cfg, labelled, "ssm_run", root, fold=fold)
    return cfg, root, idx


def test_build_writes_latent_context_and_transform(ssm_dataset):
    cfg, root, idx = ssm_dataset
    assert idx.latent_context_path and (root / idx.latent_context_path).exists()
    assert idx.latent_transform_path and (root / idx.latent_transform_path).exists()
    assert idx.latent_state_dim == 4
    ctx = pd.read_parquet(root / idx.latent_context_path)
    for c in latent_state_columns(4):
        assert c in ctx.columns


def test_latent_context_appended_only_when_requested(ssm_dataset):
    cfg, root, idx = ssm_dataset
    without = load_model_data(cfg, idx, "train", root, with_sequences=True, include_latent_context=False)
    with_ssm = load_model_data(cfg, idx, "train", root, with_sequences=True, include_latent_context=True)
    extra = with_ssm.context_dim - without.context_dim
    assert extra == 8  # 2 * state_dim (z + var)
    assert any(name.startswith("ssm_z_") for name in (with_ssm.context_names or []))
