"""A small temporal convolutional network over the book windows.

Two causal conv layers, take the last time step, then a linear head per horizon
with a softmax over {-1, 0, 1}. The loss is the average cross-entropy across
horizons, only counting events whose label is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .base import ForecastModel, register_model
from .data import ModelData
from .prediction import build_predictions

DEFAULT_PARAMS = {
    "channels": 16,
    "kernel_size": 3,
    "num_layers": 2,
    "epochs": 5,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "dropout": 0.0,
    "weight_decay": 0.0,
    "early_stopping": False,
    "patience": 3,
    "restore_best_validation": True,
    "selection_metric": "mean_validation_direction_accuracy",
}


class _CausalConv1d(nn.Module):
    """A Conv1d that only looks at the past: we pad on the left and don't pad right."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class _TCNNet(nn.Module):
    def __init__(self, in_features: int, channels: int, kernel: int, num_layers: int, horizons: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.horizons = list(horizons)
        layers: list[nn.Module] = []
        ch_in = in_features
        for i in range(num_layers):
            layers.append(_CausalConv1d(ch_in, channels, kernel, dilation=2**i))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            ch_in = channels
        self.body = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({str(h): nn.Linear(channels, 3) for h in horizons})

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        # x comes in as (batch, length, features); conv wants (batch, features, length)
        h = self.body(x.transpose(1, 2))
        last = h[:, :, -1]  # take the last time step
        return {hh: self.heads[str(hh)](last) for hh in self.horizons}


@register_model
class TCNModel(ForecastModel):
    """The small TCN. Predicts a direction for each horizon."""

    name = "tcn_small"
    version = "1.0"
    requires_sequences = True

    def __init__(self, **params: object) -> None:
        self.params = {**DEFAULT_PARAMS, **params}
        self.net: _TCNNet | None = None
        self.in_features: int | None = None
        self.horizons: list[int] = []

    def fit(self, train: ModelData, validation: ModelData, config) -> None:
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        n = train.n_sequences
        f = len(train.feature_names)
        self.in_features = f
        self.horizons = list(train.horizons)

        self.net = _TCNNet(
            in_features=f,
            channels=int(self.params["channels"]),
            kernel=int(self.params["kernel_size"]),
            num_layers=int(self.params["num_layers"]),
            horizons=self.horizons,
            dropout=float(self.params.get("dropout", 0.0)),
        )
        if n == 0:
            return  # no windows to train on; leave the net as initialised

        # labels {-1,0,1} -> {0,1,2} for cross-entropy, plus the availability masks
        targets = {h: torch.tensor(train.seq_true_direction(h), dtype=torch.float32) for h in self.horizons}
        avail = {h: torch.tensor(train.seq_available(h), dtype=torch.bool) for h in self.horizons}

        opt = torch.optim.Adam(
            self.net.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )
        lam = 1.0 / len(self.horizons)
        batch = int(self.params["batch_size"])
        gen = torch.Generator().manual_seed(config.random_seed)

        from ..utils.progress import log, progress

        import time as _time
        epochs = int(self.params["epochs"])
        patience = int(self.params.get("patience", 999))
        n_batches = (n + batch - 1) // batch
        best_score = -np.inf
        best_state = None
        no_improve = 0
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            t0 = _time.time()
            for s in progress(range(0, n, batch), desc=f"{self.name} epoch {ep+1}/{epochs}", total=n_batches):
                idx = perm[s : s + batch]
                xb = torch.from_numpy(train.seq_window_batch(idx.numpy()))
                logits = self.net(xb)
                loss = torch.zeros(())
                for h in self.horizons:
                    m = avail[h][idx]
                    if not bool(m.any()):
                        continue
                    y = (targets[h][idx][m] + 1).long()  # {-1,0,1} -> {0,1,2}
                    loss = loss + lam * F.cross_entropy(logits[h][m], y)
                if loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            msg = f"{self.name} epoch {ep+1}/{epochs} done ({_time.time()-t0:.0f}s, {n:,} windows)"
            if self.params.get("early_stopping") and validation.n_sequences > 0:
                score = self._validation_score(validation)
                msg += f" val_acc={score:.4f}"
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    msg += f" (no_improve={no_improve}/{patience})"
            log(msg)
            if self.params.get("early_stopping") and no_improve >= patience:
                log(f"{self.name}: early stopping at epoch {ep+1} (patience={patience})")
                break
        if best_state is not None and self.params.get("restore_best_validation", True):
            self.net.load_state_dict(best_state)

    def _validation_score(self, validation: ModelData) -> float:
        """Mean direction accuracy across horizons on the validation set."""
        self.net.eval()
        batch = int(self.params["batch_size"])
        n = validation.n_sequences
        accs: list[float] = []
        parts: dict[int, list[np.ndarray]] = {h: [] for h in self.horizons}
        with torch.no_grad():
            for s in range(0, n, batch):
                idx = np.arange(s, min(s + batch, n))
                xb = torch.from_numpy(validation.seq_window_batch(idx))
                logits = self.net(xb)
                for h in self.horizons:
                    parts[h].append(torch.softmax(logits[h], dim=1).numpy())
        for h in self.horizons:
            p = np.concatenate(parts[h], axis=0) if n > 0 else np.zeros((0, 3))
            pred_cls = np.array([-1, 0, 1])[p.argmax(axis=1)] if n > 0 else np.zeros(0)
            true_dir = validation.seq_true_direction(h)
            avail = validation.seq_available(h)
            m = avail.astype(bool)
            if m.sum() > 0:
                accs.append(float((pred_cls[m] == true_dir[m]).mean()))
        return float(np.mean(accs)) if accs else 0.0

    def predict(self, data: ModelData, config, run_id: str = "") -> pd.DataFrame:
        assert self.net is not None, "Model must be fitted before predict()"
        ids = data.seq_ids()
        n = len(ids)
        proba: dict[int, np.ndarray] = {}
        cls: dict[int, np.ndarray] = {}

        self.net.eval()
        if n > 0:
            batch = int(self.params["batch_size"])
            parts: dict[int, list[np.ndarray]] = {h: [] for h in data.horizons}
            with torch.no_grad():
                for s in range(0, n, batch):
                    idx = np.arange(s, min(s + batch, n))
                    xb = torch.from_numpy(data.seq_window_batch(idx))
                    logits = self.net(xb)
                    for h in data.horizons:
                        parts[h].append(torch.softmax(logits[h], dim=1).numpy())
            for h in data.horizons:
                p = np.concatenate(parts[h], axis=0)
                proba[h] = p
                cls[h] = np.array([-1, 0, 1])[p.argmax(axis=1)].astype("float64")
        else:
            for h in data.horizons:
                proba[h] = np.zeros((0, 3))
                cls[h] = np.zeros(0)

        return build_predictions(
            run_id=run_id,
            model_name=self.name,
            model_version=self.version,
            split=data.split,
            ids=ids,
            horizons=data.horizons,
            true_return={h: data.seq_true_return(h) for h in data.horizons},
            true_direction={h: data.seq_true_direction(h) for h in data.horizons},
            pred_proba=proba,
            pred_class=cls,
        )

    def hyperparameters(self) -> dict:
        return dict(self.params)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict() if self.net is not None else None,
                "params": self.params,
                "in_features": self.in_features,
                "horizons": self.horizons,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TCNModel":
        state = torch.load(path, weights_only=False)
        model = cls(**state["params"])
        model.in_features = state["in_features"]
        model.horizons = state["horizons"]
        if state["state_dict"] is not None:
            model.net = _TCNNet(
                in_features=state["in_features"],
                channels=int(model.params["channels"]),
                kernel=int(model.params["kernel_size"]),
                num_layers=int(model.params["num_layers"]),
                horizons=state["horizons"],
                dropout=float(model.params.get("dropout", 0.0)),
            )
            model.net.load_state_dict(state["state_dict"])
            model.net.eval()
        return model
