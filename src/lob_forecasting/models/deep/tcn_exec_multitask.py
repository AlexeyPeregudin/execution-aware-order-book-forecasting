"""Compact execution-aware multi-task TCN.

A causal dilated-convolution encoder feeds a temporal pooling layer (gated by
default), which is fused with a regime/context embedding and read out by
per-horizon heads. Each head predicts, for every horizon:

    return (Huber), 3-class direction (cross-entropy),
    q05/q50/q95 return quantiles (monotone, pinball loss),
    bid/ask passive markout (Huber), bid/ask adverse selection (>=0, Huber).

The point is not size but objective: the model emits exactly the quantities an
execution-aware policy consumes. Training uses the existing lazy batched window
loader, so no dense (n, L, F) tensor is ever materialised.

The model supports:

  - per-head enable/disable and per-head loss weights via `execution_heads`;
  - a return head that can be detached from the encoder (its loss trains the head
    but not the shared trunk) or omitted entirely (point return supplied by a
    ridge sidecar merged downstream);
  - a composite validation score that balances direction skill, calibration and
    execution-head accuracy (`selection.composite_score`); and
  - an optional two-phase schedule: joint training, then a short execution-head
    calibration phase with the encoder frozen.

Legacy configs that only set `objective_weights` keep working: the weights are
mapped onto the per-head wiring with all heads enabled and a neural return head.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ...config.schema import ExecutionHeadsConfig
from ..base import ForecastModel, register_model
from ..data import ModelData
from ..prediction import build_predictions
from .selection import HorizonScales, composite_score, fit_scales

# shared head output layout (per horizon): direction(3) | quantile raw(3) | markout(2) | adverse(2)
_SHARED_WIDTH = 3 + 3 + 2 + 2  # = 10

DEFAULT_PARAMS = {
    "channels": 16,
    "kernel_size": 7,
    "num_layers": 5,
    "sequence_length": 100,
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "dropout": 0.05,
    "weight_decay": 1e-4,
    "pooling": "gated",  # gated | last_step | attention
    "context_hidden": 16,
    "fusion_hidden": 32,
    "early_stopping": True,
    "patience": 999,          # epochs without improvement before stopping
    "min_epochs": 0,          # never stop before this many epochs regardless of patience
    "restore_best_validation": True,
    "selection_metric": "composite",  # composite | rank_ic_plus_accuracy (legacy)
    "optimizer": "adamw",     # adamw | adam
    # legacy weighting (used only when execution_heads is absent)
    "objective_weights": {
        "return": 1.0,
        "direction": 0.5,
        "quantile": 0.5,
        "markout": 0.5,
        "adverse": 0.25,
    },
    # new per-head wiring; None => derive from objective_weights
    "execution_heads": None,
    # optional two-phase schedule; disabled by default
    "two_phase": {
        "enabled": False,
        "phase_a": {"epochs": 30, "min_epochs": 12, "patience": 8,
                    "learning_rate": 5e-4, "weight_decay": 5e-4},
        "phase_b": {"epochs": 8, "learning_rate": 1e-4, "weight_decay": 1e-5,
                    "weights": {"direction": 0.0, "quantile": 2.0, "markout": 1.0, "adverse": 1.0}},
    },
}

_QUANTILE_TAUS = (0.05, 0.50, 0.95)
_ALL_HEADS = ("direction", "quantile", "markout", "adverse")


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class _GatedPool(nn.Module):
    """Attention-style gated pooling over time; logs alpha stats for diagnostics."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Linear(channels, 1)
        self.last_alpha_mean: float = 0.0

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(h.transpose(1, 2)).squeeze(-1))  # (batch, length)
        alpha = g / (g.sum(dim=1, keepdim=True) + 1e-8)
        self.last_alpha_mean = float(alpha.mean().detach())
        return torch.einsum("bcl,bl->bc", h, alpha)


class _AttentionPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        s = self.score(h.transpose(1, 2)).squeeze(-1)
        alpha = torch.softmax(s, dim=1)
        return torch.einsum("bcl,bl->bc", h, alpha)


class _MultiTaskNet(nn.Module):
    """Encoder + pooling + context fusion + per-horizon shared and return heads.

    The point-return head is a separate linear layer so its gradient can be
    detached from the shared trunk independently of the other heads.
    """

    def __init__(
        self,
        in_features: int,
        context_dim: int,
        channels: int,
        kernel: int,
        num_layers: int,
        dropout: float,
        pooling: str,
        context_hidden: int,
        fusion_hidden: int,
        horizons: list[int],
    ) -> None:
        super().__init__()
        self.horizons = list(horizons)
        self.pooling = pooling

        layers: list[nn.Module] = []
        ch_in = in_features
        for i in range(num_layers):
            layers.append(_CausalConv1d(ch_in, channels, kernel, dilation=2**i))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            ch_in = channels
        self.body = nn.Sequential(*layers)

        if pooling == "gated":
            self.pool: nn.Module | None = _GatedPool(channels)
        elif pooling == "attention":
            self.pool = _AttentionPool(channels)
        else:
            self.pool = None  # last-step pooling

        fusion_in = channels
        self.ctx: nn.Module | None = None
        if context_dim > 0:
            self.ctx = nn.Sequential(nn.Linear(context_dim, context_hidden), nn.ReLU())
            fusion_in += context_hidden

        self.fusion = nn.Sequential(nn.Linear(fusion_in, fusion_hidden), nn.ReLU())
        self.shared_heads = nn.ModuleDict(
            {str(h): nn.Linear(fusion_hidden, _SHARED_WIDTH) for h in horizons}
        )
        self.return_heads = nn.ModuleDict(
            {str(h): nn.Linear(fusion_hidden, 1) for h in horizons}
        )

    def encode(self, x: torch.Tensor, ctx: torch.Tensor | None) -> torch.Tensor:
        h = self.body(x.transpose(1, 2))  # (batch, channels, length)
        z = h[:, :, -1] if self.pool is None else self.pool(h)
        if self.ctx is not None and ctx is not None and ctx.shape[1] > 0:
            z = torch.cat([z, self.ctx(ctx)], dim=1)
        return self.fusion(z)

    def encoder_modules(self) -> list[nn.Module]:
        """Trunk modules frozen during the execution-head calibration phase."""
        mods: list[nn.Module] = [self.body, self.fusion]
        if self.pool is not None:
            mods.append(self.pool)
        if self.ctx is not None:
            mods.append(self.ctx)
        return mods

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.shared_heads.parameters()) + list(self.return_heads.parameters())

    def forward(
        self, x: torch.Tensor, ctx: torch.Tensor | None, detach_return: bool = False
    ) -> dict[int, dict[str, torch.Tensor]]:
        u = self.encode(x, ctx)
        u_ret = u.detach() if detach_return else u
        out: dict[int, dict[str, torch.Tensor]] = {}
        for hh in self.horizons:
            raw = self.shared_heads[str(hh)](u)
            direction = raw[:, 0:3]
            q_raw = raw[:, 3:6]
            q05 = q_raw[:, 0]
            q50 = q05 + F.softplus(q_raw[:, 1])
            q95 = q50 + F.softplus(q_raw[:, 2])
            markout = raw[:, 6:8]
            adverse = F.softplus(raw[:, 8:10])
            ret = self.return_heads[str(hh)](u_ret).squeeze(-1)
            out[hh] = {
                "return": ret,
                "direction": direction,
                "q05": q05, "q50": q50, "q95": q95,
                "markout": markout,
                "adverse": adverse,
            }
        return out


def _pinball(r: torch.Tensor, q: torch.Tensor, tau: float) -> torch.Tensor:
    diff = r - q
    return torch.maximum(tau * diff, (tau - 1.0) * diff)


def _resolve_heads(params: dict) -> ExecutionHeadsConfig:
    """Per-head wiring from `execution_heads` or, failing that, legacy weights."""
    eh = params.get("execution_heads")
    if eh:
        return ExecutionHeadsConfig.model_validate(eh)
    ow = {**DEFAULT_PARAMS["objective_weights"], **params.get("objective_weights", {})}
    return ExecutionHeadsConfig.model_validate({
        "return_head": {"enabled": ow.get("return", 0.0) > 0, "loss_weight": ow.get("return", 0.0),
                        "detach_from_encoder": False, "prediction_source": "neural_head"},
        "direction_head": {"enabled": True, "loss_weight": ow.get("direction", 1.0)},
        "quantile_head": {"enabled": True, "loss_weight": ow.get("quantile", 1.0)},
        "markout_head": {"enabled": True, "loss_weight": ow.get("markout", 0.5)},
        "adverse_head": {"enabled": True, "loss_weight": ow.get("adverse", 0.25)},
    })


@register_model
class TCNExecMultiTaskModel(ForecastModel):
    """The compact execution-aware multi-task TCN."""

    name = "tcn_exec_multitask"
    version = "2.0"
    requires_sequences = True

    def __init__(self, **params: object) -> None:
        self.params = {**DEFAULT_PARAMS, **params}
        self.net: _MultiTaskNet | None = None
        self.in_features: int | None = None
        self.context_dim: int = 0
        self.horizons: list[int] = []
        self.alpha_mean_log: float | None = None
        self.heads = _resolve_heads(self.params)
        self.scales: dict[int, HorizonScales] = {}
        self.training_log: list[dict] = []

    # helpers

    def _build_net(self) -> _MultiTaskNet:
        return _MultiTaskNet(
            in_features=int(self.in_features),
            context_dim=int(self.context_dim),
            channels=int(self.params["channels"]),
            kernel=int(self.params["kernel_size"]),
            num_layers=int(self.params["num_layers"]),
            dropout=float(self.params["dropout"]),
            pooling=str(self.params["pooling"]),
            context_hidden=int(self.params["context_hidden"]),
            fusion_hidden=int(self.params["fusion_hidden"]),
            horizons=self.horizons,
        )

    def _enabled_heads(self) -> set[str]:
        return {h for h in _ALL_HEADS if getattr(self.heads, f"{h}_head").enabled}

    def _loss_weights(self, override: dict | None = None) -> dict[str, float]:
        """Per-objective loss weights (0 disables a term).

        `override` (Phase B re-weighting) replaces the named weights, but a
        disabled head is always pinned to 0 regardless of the override.
        """
        w = {
            obj: (getattr(self.heads, f"{obj}_head").loss_weight if self._head_enabled(obj) else 0.0)
            for obj in ("return", "direction", "quantile", "markout", "adverse")
        }
        if override:
            for k, v in override.items():
                if k in w:
                    w[k] = float(v) if self._head_enabled(k) else 0.0
        return w

    def _head_enabled(self, name: str) -> bool:
        if name == "return":
            return self.heads.return_head.enabled
        return getattr(self.heads, f"{name}_head").enabled

    def _batch_tensors(self, data: ModelData, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor | None]:
        xb = torch.from_numpy(data.seq_window_batch(idx))
        cb = None
        if self.context_dim > 0:
            cb = torch.from_numpy(data.seq_context_batch(idx))
        return xb, cb

    def _loss_for_batch(self, data: ModelData, idx: np.ndarray, targets: dict, weights: dict) -> torch.Tensor:
        xb, cb = self._batch_tensors(data, idx)
        detach_return = self.heads.return_head.detach_from_encoder
        out = self.net(xb, cb, detach_return=detach_return)
        lam = 1.0 / len(self.horizons)
        total = torch.zeros(())
        for h in self.horizons:
            o = out[h]
            avail = targets["avail"][h][idx]
            mk_avail = targets["mk_avail"][h][idx]
            loss_h = torch.zeros(())
            if bool(avail.any()):
                r = targets["ret"][h][idx][avail]
                y = (targets["dir"][h][idx][avail] + 1).long()
                if weights["return"] > 0:
                    loss_h = loss_h + weights["return"] * F.huber_loss(o["return"][avail], r)
                if weights["direction"] > 0:
                    loss_h = loss_h + weights["direction"] * F.cross_entropy(o["direction"][avail], y)
                if weights["quantile"] > 0:
                    ql = torch.zeros(())
                    for tau, qcol in zip(_QUANTILE_TAUS, ("q05", "q50", "q95")):
                        ql = ql + _pinball(r, o[qcol][avail], tau).mean()
                    loss_h = loss_h + weights["quantile"] * ql
            if bool(mk_avail.any()) and (weights["markout"] > 0 or weights["adverse"] > 0):
                mb = targets["mk_bid"][h][idx][mk_avail]
                ma = targets["mk_ask"][h][idx][mk_avail]
                ab = targets["adv_bid"][h][idx][mk_avail]
                aa = targets["adv_ask"][h][idx][mk_avail]
                if weights["markout"] > 0:
                    mk_loss = F.huber_loss(o["markout"][mk_avail][:, 0], mb) + F.huber_loss(
                        o["markout"][mk_avail][:, 1], ma)
                    loss_h = loss_h + weights["markout"] * mk_loss
                if weights["adverse"] > 0:
                    adv_loss = F.huber_loss(o["adverse"][mk_avail][:, 0], ab) + F.huber_loss(
                        o["adverse"][mk_avail][:, 1], aa)
                    loss_h = loss_h + weights["adverse"] * adv_loss
            total = total + lam * loss_h
        return total

    def _gather_targets(self, data: ModelData) -> dict:
        t = lambda a, dt: torch.tensor(a, dtype=dt)  # noqa: E731
        return {
            "ret": {h: t(data.seq_true_return(h), torch.float32) for h in self.horizons},
            "dir": {h: t(data.seq_true_direction(h), torch.float32) for h in self.horizons},
            "avail": {h: t(data.seq_available(h), torch.bool) for h in self.horizons},
            "mk_avail": {h: t(data.seq_markout_available(h), torch.bool) for h in self.horizons},
            "mk_bid": {h: t(np.nan_to_num(data.seq_markout("bid", h)), torch.float32) for h in self.horizons},
            "mk_ask": {h: t(np.nan_to_num(data.seq_markout("ask", h)), torch.float32) for h in self.horizons},
            "adv_bid": {h: t(np.nan_to_num(data.seq_adverse("bid", h)), torch.float32) for h in self.horizons},
            "adv_ask": {h: t(np.nan_to_num(data.seq_adverse("ask", h)), torch.float32) for h in self.horizons},
        }

    def _make_optimizer(self, parameters, lr: float, weight_decay: float):
        if str(self.params.get("optimizer", "adamw")).lower() == "adam":
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    # ForecastModel API

    def _phase_plan(self) -> list[dict]:
        """The ordered training phases (one entry for single-phase training)."""
        two_phase = dict(self.params.get("two_phase") or {})
        if two_phase.get("enabled"):
            pa = {**DEFAULT_PARAMS["two_phase"]["phase_a"], **two_phase.get("phase_a", {})}
            pb = {**DEFAULT_PARAMS["two_phase"]["phase_b"], **two_phase.get("phase_b", {})}
            return [
                {"phase": "joint", "weights": self._loss_weights(), "head_only": False,
                 "epochs": int(pa["epochs"]), "min_epochs": int(pa["min_epochs"]),
                 "patience": int(pa["patience"]), "lr": float(pa["learning_rate"]),
                 "weight_decay": float(pa["weight_decay"])},
                {"phase": "head_calibration", "weights": self._loss_weights(pb.get("weights")),
                 "head_only": True,
                 "epochs": int(pb["epochs"]), "min_epochs": int(pb["epochs"]),
                 "patience": int(pb["epochs"]), "lr": float(pb["learning_rate"]),
                 "weight_decay": float(pb["weight_decay"])},
            ]
        return [{"phase": "joint", "weights": self._loss_weights(), "head_only": False,
                 "epochs": int(self.params["epochs"]),
                 "min_epochs": int(self.params.get("min_epochs", 0)),
                 "patience": int(self.params.get("patience", 999)),
                 "lr": float(self.params["learning_rate"]),
                 "weight_decay": float(self.params["weight_decay"])}]

    def fit(self, train: ModelData, validation: ModelData, config) -> None:
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        self.in_features = len(train.feature_names)
        self.context_dim = train.context_dim
        self.horizons = list(train.horizons)
        self.net = self._build_net()
        self.training_log = []

        n = train.n_sequences
        if n == 0:
            return

        targets = self._gather_targets(train)
        self.scales = fit_scales(
            self.horizons,
            {h: targets["ret"][h].numpy() for h in self.horizons},
            {h: data_or_nan(train.seq_markout, "bid", h) for h in self.horizons},
            {h: data_or_nan(train.seq_markout, "ask", h) for h in self.horizons},
            {h: data_or_nan(train.seq_adverse, "bid", h) for h in self.horizons},
            {h: data_or_nan(train.seq_adverse, "ask", h) for h in self.horizons},
        )

        plan = self._phase_plan()
        # resume from a per-epoch checkpoint if one exists for this model (crash safety)
        ckpt_path = self.params.get("_checkpoint_path")
        start_pi, start_ep, resume_state = 0, 0, None
        if ckpt_path and bool(self.params.get("_resume")) and Path(ckpt_path).exists():
            from ...utils.progress import log
            st = torch.load(ckpt_path, weights_only=False)
            self.net.load_state_dict(st["net_state"])
            self.training_log = st.get("training_log", [])
            start_pi, start_ep = int(st["phase_idx"]), int(st["epoch_in_phase"])
            resume_state = {"best_score": st["best_score"], "best_state": st["best_state"],
                            "no_improve": st["no_improve"]}
            log(f"{self.name}: resuming from checkpoint (phase {start_pi}, epoch {start_ep})")

        for pi, spec in enumerate(plan):
            if pi < start_pi:
                continue
            if spec["head_only"]:
                self._freeze_encoder()
                params = self.net.head_parameters()
            else:
                params = list(self.net.parameters())
            s_ep = start_ep if pi == start_pi else 0
            r_state = resume_state if (pi == start_pi and s_ep > 0) else None
            self._train_phase(
                train, validation, config, targets, parameters=params,
                phase=spec["phase"], weights=spec["weights"], epochs=spec["epochs"],
                min_epochs=spec["min_epochs"], patience=spec["patience"],
                lr=spec["lr"], weight_decay=spec["weight_decay"],
                phase_idx=pi, start_epoch=s_ep, resume_state=r_state, ckpt_path=ckpt_path,
            )
            if spec["head_only"]:
                self._unfreeze_encoder()

        if isinstance(self.net.pool, _GatedPool):
            self.alpha_mean_log = self.net.pool.last_alpha_mean

    def _freeze_encoder(self) -> None:
        for mod in self.net.encoder_modules():
            for p in mod.parameters():
                p.requires_grad_(False)

    def _unfreeze_encoder(self) -> None:
        for mod in self.net.encoder_modules():
            for p in mod.parameters():
                p.requires_grad_(True)

    def _save_ckpt(self, ckpt_path, phase_idx, epoch_in_phase, best_score, best_state, no_improve) -> None:
        if not ckpt_path:
            return
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(ckpt_path) + ".tmp"
        torch.save({"phase_idx": phase_idx, "epoch_in_phase": epoch_in_phase,
                    "net_state": self.net.state_dict(), "best_score": best_score,
                    "best_state": best_state, "no_improve": no_improve,
                    "training_log": self.training_log}, tmp)
        Path(tmp).replace(ckpt_path)  # atomic, so a crash mid-write can't corrupt it

    def _train_phase(self, train, validation, config, targets, *, phase, weights,
                     parameters, epochs, min_epochs, patience, lr, weight_decay,
                     phase_idx=0, start_epoch=0, resume_state=None, ckpt_path=None) -> None:
        from ...utils.progress import log, progress

        n = train.n_sequences
        batch = int(self.params["batch_size"])
        n_batches = (n + batch - 1) // batch
        gen = torch.Generator().manual_seed(config.random_seed + phase_idx)
        opt = self._make_optimizer([p for p in parameters if p.requires_grad], lr, weight_decay)
        early = bool(self.params.get("early_stopping")) and validation.n_sequences > 0
        best_score = resume_state["best_score"] if resume_state else -np.inf
        best_state = resume_state["best_state"] if resume_state else None
        no_improve = resume_state["no_improve"] if resume_state else 0
        for ep in range(start_epoch, epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen).numpy()
            t0 = time.time()
            run_loss = 0.0
            for s in progress(range(0, n, batch),
                              desc=f"{self.name} [{phase}] epoch {ep+1}/{epochs}", total=n_batches):
                idx = perm[s : s + batch]
                loss = self._loss_for_batch(train, idx, targets, weights)
                if loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    run_loss += float(loss.detach())
            rec: dict = {"phase": phase, "epoch": ep + 1,
                         "train_loss_total": round(run_loss / max(1, n_batches), 6)}
            msg = (f"{self.name} [{phase}] epoch {ep+1}/{epochs}: "
                   f"loss={rec['train_loss_total']:.4f} ({time.time()-t0:.0f}s, {n:,} windows)")
            if early:
                score, comps = self._validation_score(validation)
                rec["val_score_composite"] = round(score, 6)
                rec.update({k: round(v, 6) for k, v in comps.items()})
                msg += f" val_score={score:.4f}"
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    no_improve = 0
                    rec["checkpoint_saved"] = True
                else:
                    no_improve += 1
                    rec["checkpoint_saved"] = False
                    msg += f" (no_improve={no_improve}/{patience})"
            self.training_log.append(rec)
            log(msg)
            # persist after every epoch so a crash loses at most the current epoch
            self._save_ckpt(ckpt_path, phase_idx, ep + 1, best_score, best_state, no_improve)
            if early and no_improve >= patience and (ep + 1) >= min_epochs:
                log(f"{self.name} [{phase}]: early stopping at epoch {ep+1} "
                    f"(patience={patience}, min_epochs={min_epochs})")
                break
        if best_state is not None and self.params.get("restore_best_validation", True):
            self.net.load_state_dict(best_state)
        # checkpoint the best-restored net at the phase boundary (epoch_in_phase=epochs)
        self._save_ckpt(ckpt_path, phase_idx, epochs, best_score, best_state, no_improve)

    def _validation_score(self, validation: ModelData) -> tuple[float, dict]:
        """Composite validation score; legacy metric on request."""
        preds = self.predict(validation, None, run_id="")
        if str(self.params.get("selection_metric", "composite")) == "composite" and self.scales:
            return composite_score(preds, horizons=self.horizons, scales=self.scales,
                                   enabled_heads=self._enabled_heads())
        # legacy: mean rank-IC(return) + accuracy over horizons
        scores: list[float] = []
        for h in self.horizons:
            sub = preds[preds["horizon"] == h]
            m = sub[sub["true_return"].notna() & sub["pred_return"].notna()]
            if len(m) > 1:
                ic = pd.Series(m["true_return"].to_numpy()).rank().corr(
                    pd.Series(m["pred_return"].to_numpy()).rank())
                acc = float((m["pred_class"] == m["true_direction"]).mean())
                if np.isfinite(ic):
                    scores.append(float(ic) + acc)
        return (float(np.mean(scores)) if scores else 0.0), {}

    def predict(self, data: ModelData, config=None, run_id: str = "") -> pd.DataFrame:
        assert self.net is not None, "Model must be fitted before predict()"
        ids = data.seq_ids()
        n = len(ids)
        batch = int(self.params["batch_size"])
        eps = 1e-8
        return_source = self.heads.return_head.prediction_source
        emit_neural_return = return_source == "neural_head"

        proba: dict[int, list] = {h: [] for h in data.horizons}
        ret: dict[int, list] = {h: [] for h in data.horizons}
        q: dict[int, dict[str, list]] = {h: {"q05": [], "q50": [], "q95": []} for h in data.horizons}
        mk: dict[int, dict[str, list]] = {h: {"bid": [], "ask": []} for h in data.horizons}
        adv: dict[int, dict[str, list]] = {h: {"bid": [], "ask": []} for h in data.horizons}

        self.net.eval()
        with torch.no_grad():
            for s in range(0, n, batch):
                idx = np.arange(s, min(s + batch, n))
                xb, cb = self._batch_tensors(data, idx)
                out = self.net(xb, cb)
                for h in data.horizons:
                    o = out[h]
                    proba[h].append(torch.softmax(o["direction"], dim=1).numpy())
                    ret[h].append(o["return"].numpy())
                    q[h]["q05"].append(o["q05"].numpy())
                    q[h]["q50"].append(o["q50"].numpy())
                    q[h]["q95"].append(o["q95"].numpy())
                    mk[h]["bid"].append(o["markout"][:, 0].numpy())
                    mk[h]["ask"].append(o["markout"][:, 1].numpy())
                    adv[h]["bid"].append(o["adverse"][:, 0].numpy())
                    adv[h]["ask"].append(o["adverse"][:, 1].numpy())

        pred_proba: dict[int, np.ndarray] = {}
        pred_cls: dict[int, np.ndarray] = {}
        pred_ret: dict[int, np.ndarray] = {}
        exec_cols: dict[int, dict[str, np.ndarray]] = {}
        for h in data.horizons:
            if n > 0:
                p = np.concatenate(proba[h], axis=0)
                q05 = np.concatenate(q[h]["q05"]); q50 = np.concatenate(q[h]["q50"]); q95 = np.concatenate(q[h]["q95"])
                neural_ret = np.concatenate(ret[h])
                mkb = np.concatenate(mk[h]["bid"]); mka = np.concatenate(mk[h]["ask"])
                avb = np.concatenate(adv[h]["bid"]); ava = np.concatenate(adv[h]["ask"])
            else:
                p = np.zeros((0, 3)); q05 = q50 = q95 = np.zeros(0)
                neural_ret = np.zeros(0); mkb = mka = avb = ava = np.zeros(0)
            pred_proba[h] = p
            pred_cls[h] = np.array([-1, 0, 1])[p.argmax(axis=1)].astype("float64") if n else np.zeros(0)
            pred_ret[h] = neural_ret if emit_neural_return else np.full(n, np.nan)
            width = q95 - q05
            exec_cols[h] = {
                "pred_q05": q05, "pred_q50": q50, "pred_q95": q95,
                "true_markout_bid": data.seq_markout("bid", h),
                "true_markout_ask": data.seq_markout("ask", h),
                "pred_markout_bid": mkb, "pred_markout_ask": mka,
                "true_adverse_bid": data.seq_adverse("bid", h),
                "true_adverse_ask": data.seq_adverse("ask", h),
                "pred_adverse_bid": avb, "pred_adverse_ask": ava,
                "pred_interval_width": width,
                "pred_uncertainty_score": width / (np.abs(q50) + eps),
            }

        source_tag = "neural_head" if emit_neural_return else (
            "ridge_sidecar" if return_source == "ridge_sidecar" else "none")
        return build_predictions(
            run_id=run_id,
            model_name=self.name,
            model_version=self.version,
            split=data.split,
            ids=ids,
            horizons=data.horizons,
            true_return={h: data.seq_true_return(h) for h in data.horizons},
            true_direction={h: data.seq_true_direction(h) for h in data.horizons},
            pred_return=pred_ret,
            pred_proba=pred_proba,
            pred_class=pred_cls,
            exec_columns=exec_cols,
            pred_return_source=source_tag,
        )

    def hyperparameters(self) -> dict:
        hp = {k: v for k, v in self.params.items() if not k.startswith("_")}
        hp["execution_heads_effective"] = self.heads.model_dump()
        if self.alpha_mean_log is not None:
            hp["gated_alpha_mean"] = self.alpha_mean_log
        if self.training_log:
            hp["training_log"] = self.training_log
        return hp

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict() if self.net is not None else None,
                "params": self.params,
                "in_features": self.in_features,
                "context_dim": self.context_dim,
                "horizons": self.horizons,
                "scales": {h: vars(s) for h, s in self.scales.items()},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TCNExecMultiTaskModel":
        state = torch.load(path, weights_only=False)
        model = cls(**state["params"])
        model.in_features = state["in_features"]
        model.context_dim = state["context_dim"]
        model.horizons = state["horizons"]
        model.scales = {int(h): HorizonScales(**v) for h, v in state.get("scales", {}).items()}
        if state["state_dict"] is not None:
            model.net = model._build_net()
            model.net.load_state_dict(state["state_dict"])
            model.net.eval()
        return model


def data_or_nan(getter, side: str, h: int) -> np.ndarray:
    """Fetch a sequence target, tolerating absence (returns NaNs)."""
    try:
        return getter(side, h)
    except Exception:  # pragma: no cover - defensive
        return np.array([np.nan])
