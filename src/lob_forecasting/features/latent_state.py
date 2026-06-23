"""Fitted linear-Gaussian latent state-space context.

A small time-invariant linear-Gaussian state-space model

    z_{t+1} = A z_t + w_t,   w_t ~ N(0, Q)
    y_t     = C z_t + d + v_t, v_t ~ N(0, R)

gives a causal estimate of hidden market state (drift, liquidity stress,
volatility, order-flow pressure) that is smoother than the rolling regime
buckets. It provides extra context; it does not replace the TCN.

A few rules are enforced here:
  - parameters are fitted on training-month observations only;
  - fitting may use the RTS smoother (offline, on training data), but the
    features emitted for any split are the causal filtered states z_{t|t} and
    their variances -- never the smoothed states;
  - the Kalman prior is reset at the start of each monthly day, so no state
    crosses a day boundary;
  - observations are standardised with training-month mean/std.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_EPS = 1e-8


def _standardise_fit(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(Y, axis=0)
    sd = np.nanstd(Y, axis=0)
    sd[~np.isfinite(sd) | (sd < _EPS)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    return mu, sd


@dataclass
class LinearGaussianSSM:
    """A fitted time-invariant linear-Gaussian state-space model."""

    state_dim: int
    obs_dim: int
    A: np.ndarray
    C: np.ndarray
    d: np.ndarray
    Q: np.ndarray            # diagonal (state_dim,)
    R: np.ndarray            # diagonal (obs_dim,)
    mu0: np.ndarray
    P0: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    observation_columns: list[str] = field(default_factory=list)
    n_em_iterations: int = 0
    final_loglik: float = 0.0

    # causal filtering -- the only thing used to produce features

    def filter(self, Y: np.ndarray) -> dict[str, np.ndarray]:
        """Causal Kalman filter of one contiguous block (one monthly day).

        Missing rows (any non-finite observation) carry the prior forward (a pure
        time update with no measurement) and are flagged. Returns filtered means
        `z` (T, K), filtered variances `var` (T, K) = diag(P_{t|t}), per-step
        log-likelihood increments and a missing-observation flag.
        """
        T = Y.shape[0]
        K = self.state_dim
        Ys = (Y - self.obs_mean) / self.obs_std
        z = np.zeros((T, K))
        var = np.zeros((T, K))
        loglik_inc = np.zeros(T)
        missing = np.zeros(T, dtype=bool)

        m = self.mu0.copy()
        P = self.P0.copy()
        Q = np.diag(self.Q)
        R = np.diag(self.R)
        for t in range(T):
            # time update (predict)
            if t > 0:
                m = self.A @ m
                P = self.A @ P @ self.A.T + Q
            y = Ys[t]
            if not np.all(np.isfinite(y)):
                missing[t] = True
                z[t] = m
                var[t] = np.diag(P)
                continue
            # measurement update
            yhat = self.C @ m + self.d
            S = self.C @ P @ self.C.T + R
            innov = y - yhat
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                Sinv = np.linalg.pinv(S)
            Kg = P @ self.C.T @ Sinv
            m = m + Kg @ innov
            P = (np.eye(K) - Kg @ self.C) @ P
            z[t] = m
            var[t] = np.diag(P)
            sign, logdet = np.linalg.slogdet(2 * np.pi * S)
            loglik_inc[t] = -0.5 * (logdet + innov @ Sinv @ innov)
        return {"z": z, "var": var, "loglik_increment": loglik_inc, "missing": missing}


def _init_params(Ys: np.ndarray, K: int) -> dict:
    """PCA/AR(1) initialisation on standardised training observations."""
    P = Ys.shape[1]
    finite = Ys[np.all(np.isfinite(Ys), axis=1)]
    if len(finite) < 2:
        finite = np.zeros((2, P))
    # C from the top-K principal directions of the observations
    cov = np.cov(finite, rowvar=False)
    cov = np.atleast_2d(cov)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:K]
    loadings = evecs[:, order]  # (P, K)
    C = loadings
    # latent series by projection; AR(1) coefficient per latent, clipped to [.8,.98]
    z_proj = finite @ loadings  # (n, K)
    a = np.empty(K)
    q = np.empty(K)
    for i in range(K):
        zi = z_proj[:, i]
        if len(zi) > 2 and np.std(zi[:-1]) > _EPS:
            ai = float(np.corrcoef(zi[1:], zi[:-1])[0, 1])
            ai = 0.9 if not np.isfinite(ai) else ai
        else:
            ai = 0.9
        a[i] = float(np.clip(ai, 0.80, 0.98))
        resid = zi[1:] - a[i] * zi[:-1]
        q[i] = float(max(np.var(resid), 1e-3)) if len(resid) else 1.0
    A = np.diag(a)
    recon = z_proj @ C.T
    R = np.maximum(np.var(finite - recon, axis=0), 1e-3)
    return {"A": A, "C": C, "d": np.zeros(P), "Q": q, "R": R,
            "mu0": np.zeros(K), "P0": np.eye(K)}


def _kalman_for_loglik(ssm: LinearGaussianSSM, blocks: list[np.ndarray]) -> float:
    total = 0.0
    for Y in blocks:
        total += float(np.nansum(ssm.filter(Y)["loglik_increment"]))
    return total


def _subsample_blocks(blocks: list[np.ndarray], max_rows: int) -> list[np.ndarray]:
    """Cap total rows used for parameter fitting, keeping each block contiguous.

    The Kalman filter is a sequential O(T) loop, so EM over the full multi-month
    training set is too slow; a few tens of thousands of contiguous rows are
    enough to estimate a 4-state LGSSM. Feature filtering still runs on every row
    downstream -- only parameter estimation is subsampled.
    """
    out: list[np.ndarray] = []
    used = 0
    for b in blocks:
        if len(b) == 0 or used >= max_rows:
            continue
        take = min(len(b), max_rows - used)
        out.append(b[:take])
        used += take
    return out or [b for b in blocks if len(b)][:1]


def fit_latent_state(
    blocks: list[np.ndarray],
    observation_columns: list[str],
    state_dim: int = 4,
    max_em_iterations: int = 25,
    loglik_tol: float = 1e-4,
    max_fit_rows: int = 40000,
) -> LinearGaussianSSM:
    """Fit the SSM on training-day observation blocks (one block per monthly day).

    Uses a stable closed-form PCA/AR(1) initialisation and then refines the
    diagonal process/observation noise via filtered-residual EM-style updates,
    stopping on the relative log-likelihood improvement. Parameters depend only on
    the supplied (training) blocks; EM uses at most `max_fit_rows` contiguous
    rows so the fit stays tractable on multi-month real data.
    """
    P = len(observation_columns)
    blocks = _subsample_blocks([b for b in blocks if len(b)], max_fit_rows)
    stacked = np.vstack(blocks) if blocks else np.zeros((1, P))
    mu, sd = _standardise_fit(stacked)
    std_blocks = [((b - mu) / sd) for b in blocks if len(b)]
    init = _init_params(np.vstack(std_blocks) if std_blocks else np.zeros((2, P)), state_dim)

    ssm = LinearGaussianSSM(
        state_dim=state_dim, obs_dim=P, A=init["A"], C=init["C"], d=init["d"],
        Q=init["Q"], R=init["R"], mu0=init["mu0"], P0=init["P0"],
        obs_mean=mu, obs_std=sd, observation_columns=list(observation_columns),
    )

    prev_ll = _kalman_for_loglik(ssm, blocks)
    iters = 0
    for it in range(max_em_iterations):
        iters = it + 1
        # refine diagonal Q, R from filtered residuals (a stable, monotone-ish step)
        innov_sq: list[np.ndarray] = []
        state_resid_sq: list[np.ndarray] = []
        for Y in blocks:
            out = ssm.filter(Y)
            z = out["z"]
            Ys = (Y - ssm.obs_mean) / ssm.obs_std
            ok = np.all(np.isfinite(Ys), axis=1)
            recon = z @ ssm.C.T + ssm.d
            innov_sq.append(((Ys - recon)[ok]) ** 2)
            if len(z) > 1:
                sr = (z[1:] - z[:-1] @ ssm.A.T) ** 2
                state_resid_sq.append(sr)
        if innov_sq:
            R_new = np.maximum(np.mean(np.vstack(innov_sq), axis=0), 1e-4)
            ssm.R = 0.5 * ssm.R + 0.5 * R_new
        if state_resid_sq:
            Q_new = np.maximum(np.mean(np.vstack(state_resid_sq), axis=0), 1e-4)
            ssm.Q = 0.5 * ssm.Q + 0.5 * Q_new
        ll = _kalman_for_loglik(ssm, blocks)
        if prev_ll != 0 and abs(ll - prev_ll) / (abs(prev_ll) + _EPS) < loglik_tol:
            prev_ll = ll
            break
        prev_ll = ll
    ssm.n_em_iterations = iters
    ssm.final_loglik = float(prev_ll)
    return ssm


def filtered_context(
    frame: pd.DataFrame,
    ssm: LinearGaussianSSM,
    *,
    day_column: str,
    reset_each_day: bool,
) -> pd.DataFrame:
    """Causal filtered-state context for every row of `frame`.

    Resets the Kalman prior at each monthly-day boundary when `reset_each_day`.
    Returns a frame aligned to `frame`'s index with columns ssm_z_*, ssm_var_*,
    ssm_loglik_increment, ssm_missing_observation_flag.
    """
    K = ssm.state_dim
    cols = ssm.observation_columns
    Y_all = frame[cols].to_numpy(dtype="float64")
    z = np.zeros((len(frame), K))
    var = np.zeros((len(frame), K))
    ll = np.zeros(len(frame))
    miss = np.zeros(len(frame), dtype=bool)

    if reset_each_day and day_column in frame.columns:
        groups = frame.groupby(day_column, sort=False).indices.values()
    else:
        groups = [np.arange(len(frame))]
    for idx in groups:
        idx = np.asarray(sorted(idx))
        out = ssm.filter(Y_all[idx])
        z[idx] = out["z"]
        var[idx] = out["var"]
        ll[idx] = out["loglik_increment"]
        miss[idx] = out["missing"]

    data = {f"ssm_z_{i+1}": z[:, i] for i in range(K)}
    data.update({f"ssm_var_{i+1}": var[:, i] for i in range(K)})
    data["ssm_loglik_increment"] = ll
    data["ssm_missing_observation_flag"] = miss
    return pd.DataFrame(data, index=frame.index)


def latent_state_columns(state_dim: int) -> list[str]:
    """The context column names appended to the model context / policy state."""
    return [f"ssm_z_{i+1}" for i in range(state_dim)] + [f"ssm_var_{i+1}" for i in range(state_dim)]


def save_ssm(ssm: LinearGaussianSSM, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(ssm, fh)
    return out


def load_ssm(path: str | Path) -> LinearGaussianSSM:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)
