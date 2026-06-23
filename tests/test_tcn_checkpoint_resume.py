"""Crash-resume tests for the multi-task TCN's per-epoch checkpointing."""

from __future__ import annotations

import torch

from lob_forecasting.datasets import build_datasets, generate_folds
from lob_forecasting.models import build_model, load_model_data

from ._monthly_helpers import monthly_config, run_to_labels


def _data(tmp_path):
    cfg = monthly_config()
    labelled = run_to_labels(cfg, tmp_path)
    fold = generate_folds(cfg)[0]
    idx = build_datasets(cfg, labelled, "ckpt_run", tmp_path, fold=fold)
    tr = load_model_data(cfg, idx, "train", tmp_path, with_sequences=True)
    va = load_model_data(cfg, idx, "validation", tmp_path, with_sequences=True)
    te = load_model_data(cfg, idx, "test", tmp_path, with_sequences=True)
    return cfg, tr, va, te


def _two_phase(ckpt, resume, a=3, b=1):
    return {
        "channels": 8, "num_layers": 2, "_checkpoint_path": str(ckpt), "_resume": resume,
        "two_phase": {"enabled": True,
                      "phase_a": {"epochs": a, "min_epochs": a, "patience": a,
                                  "learning_rate": 5e-4, "weight_decay": 5e-4},
                      "phase_b": {"epochs": b, "learning_rate": 1e-4, "weight_decay": 1e-5,
                                  "weights": {"direction": 0.0, "quantile": 2.0,
                                              "markout": 1.0, "adverse": 1.0}}},
    }


def test_checkpoint_written_each_epoch(tmp_path):
    cfg, tr, va, te = _data(tmp_path)
    ckpt = tmp_path / "ck.pt"
    m = build_model("tcn_exec_multitask", "configs/model", overrides=_two_phase(ckpt, False))
    m.fit(tr, va, cfg)
    assert ckpt.exists()
    st = torch.load(ckpt, weights_only=False)
    # after a full two-phase fit the checkpoint sits at the final phase boundary
    assert st["phase_idx"] == 1
    assert "net_state" in st and "training_log" in st


def test_resume_at_phase_boundary_reproduces(tmp_path):
    cfg, tr, va, te = _data(tmp_path)
    ckpt = tmp_path / "ck.pt"
    m = build_model("tcn_exec_multitask", "configs/model", overrides=_two_phase(ckpt, False))
    m.fit(tr, va, cfg)
    p0 = m.predict(te, cfg)["pred_q50"].to_numpy()
    # a fresh model that resumes from the completed checkpoint should not retrain
    m2 = build_model("tcn_exec_multitask", "configs/model", overrides=_two_phase(ckpt, True))
    m2.fit(tr, va, cfg)
    p1 = m2.predict(te, cfg)["pred_q50"].to_numpy()
    assert (abs(p0 - p1) < 1e-6).all()


def test_resume_midphase_continues_to_completion(tmp_path):
    cfg, tr, va, te = _data(tmp_path)
    ckpt = tmp_path / "ck.pt"
    # simulate a crash after 1 of 3 Phase-A epochs: hand-craft a partial checkpoint
    seed = build_model("tcn_exec_multitask", "configs/model", overrides=_two_phase(ckpt, False))
    seed.in_features = len(tr.feature_names)
    seed.context_dim = tr.context_dim
    seed.horizons = list(tr.horizons)
    seed.net = seed._build_net()
    torch.save({"phase_idx": 0, "epoch_in_phase": 1, "net_state": seed.net.state_dict(),
                "best_score": -1e9, "best_state": None, "no_improve": 0, "training_log": []},
               ckpt)
    m = build_model("tcn_exec_multitask", "configs/model", overrides=_two_phase(ckpt, True))
    m.fit(tr, va, cfg)
    # it resumed mid-Phase-A and ran the remaining epochs through Phase B
    phases = {r["phase"] for r in m.training_log}
    assert "head_calibration" in phases
    preds = m.predict(te, cfg)
    assert preds["pred_q50"].notna().any()
