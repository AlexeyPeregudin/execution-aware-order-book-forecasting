# Execution-Aware Limit Order Book Forecasting

This repository contains a research pipeline for short-horizon forecasting on the BTCUSDT top-5 limit order book. It builds causal features and labels from order book snapshots, trains several baseline and sequence models, and evaluates the resulting forecasts both with predictive metrics and with simple execution tests.

The reported real-data experiments use Binance USDT-margined futures snapshots from Tardis (`book_snapshot_5`), sampled every 10th book update, with horizons of 10, 50, and 200 events. The models show a small short-horizon directional edge, but the edge does not produce a robust taker or passive quoting policy under the costs and fill assumptions used here.

The repository treats forecasting and execution as separate evaluations. A model can improve direction accuracy and still fail once fees, spread, fill uncertainty, and adverse selection are included. That is what happens in the reported BTCUSDT experiments.

The accompanying [Technical Report](TechnicalReport.pdf) (37 pages) gives the full formulation, derivations, and results. This README explains how the repository is organised and how to reproduce the runs.

---

## Main result

On the reported BTCUSDT real-data run (Binance USDT-margined futures, top-5 levels, twelve first-of-month snapshots, about 1.6M event rows, expanding monthly walk-forward over three held-out months):

- `tcn_small` is the best short-horizon direction model at `h=10`, with mean accuracy 0.533 against a majority-class baseline of 0.374, and it beats the per-month baseline in all three held-out months. The multi-task model gives calibrated 90% return intervals, with empirical coverage between 0.84 and 0.95.
- The latent state-space context is the largest useful architectural change in the ablation runs, worth +9.7 points of direction accuracy at `h=10` — more than added network capacity.
- The execution tests are negative: no model clears a 5 bps taker fee, and no forecast-driven passive quoting policy beats the `no_quote` baseline across the held-out months. Realised adverse selection dominates spread capture at these horizons and costs.

The forecasting result and the execution result are separate. The forecasts contain some signal, but the tested execution policies do not turn that signal into positive realised reward. The same execution conclusion holds across all three held-out months: the taker rule loses after fees, and the passive quoting policies do not beat `no_quote`.

The passive simulator is not a soft test. It includes queue-aware fills, a fitted fill-probability model, and a control-style optimiser, so the forecasts get a reasonable chance to be useful. They still do not beat standing aside.

### Where the forecast stops being useful

The same forecast, viewed at four stages of execution:

| Stage | Best available evidence | Result |
|---|---|---|
| Raw predictive signal | `tcn_small` `h=10` accuracy 0.533 against baseline 0.374; 3 of 3 months | Passes predictive check |
| Calibrated uncertainty | 90% interval coverage 0.84 to 0.95 | Passes calibration check |
| Taker edge (5 bps) | best net PnL −26,740 (ridge) | Fails after taker fee |
| Passive-quoting edge | best quoting reward −427; `no_quote` = 0 | Does not beat no-quote baseline |

---

## Quick start

Requires Python 3.11+. A fresh checkout runs the whole pipeline on synthetic data, with no external downloads and no credentials.

```bash
# 1. Install (editable, with dev extras for the test suite)
pip install -e ".[dev]"

# 2. Run the test suite (it asserts the leakage checks directly)
make test            # or: pytest tests/

# 3. Full synthetic run: ingest, features and labels, train, evaluate, backtest, report
make mvp             # uses configs/experiment/mvp.yaml

# 4. The monthly walk-forward study on synthetic snapshots, fast smoke run
make monthly QUICK=1 # uses configs/experiment/btcusdt_top5_monthly.yaml
```

The monthly driver writes its report to `reports/monthly_results.md`, tables to `reports/tables/`, figures to `reports/figures/`, and all run artefacts under `artefacts/runs/{run_id}/`. After a run, start with `reports/monthly_results.md`; the supporting CSVs are in `reports/tables/`.

The synthetic runs are smoke tests for the pipeline. They will not reproduce the headline numbers — those come from the real-data run below.

---

## Running on real data

The reported results use Tardis `book_snapshot_5` daily archives for Binance-futures BTCUSDT. Real data is expected to be present locally: drop the gzipped CSVs into the expected layout and point ingestion at it.

```
data/raw/binance-futures/book_snapshot_5/{YYYY}/{MM}/{DD}/BTCUSDT.csv.gz
```

```bash
# 12-month expanding walk-forward on real data (July 2025 to June 2026)
python scripts/run_monthly_robustness.py \
    --config configs/experiment/btcusdt_top5_monthly_real_12m.yaml
```

The real-data run is substantially slower than the synthetic smoke run; training and the bootstrap are the slow parts. Use `--quick` while checking that the pipeline works end to end, and the full config for the reported results.

Useful driver flags: `--quick` (fewer epochs and bootstrap), `--folds 2,3,4`, `--exec-variants` (return-head and latent-SSM ablations), `--resume` (continue an interrupted run), and `--models ... --reuse-predictions-from {run_id}` (selective retraining).

---

## Repository layout

```
src/lob_forecasting/
  config/         schema.py loader.py            # typed config + hard invariants
  ingestion/      sources.py manifest.py ingest.py
  normalisation/  event_table.py adapters.py normalise.py
  orderbook/      book_state.py reconstruct.py   # vectorised fast path + striding
  features/       compute.py regimes.py latent_state.py build.py
  labels/         compute.py quantiles.py markout.py build.py
  datasets/       splits.py monthly_splits.py scaler.py build.py
  models/         registry.py baselines.py linear.py gbm.py tcn.py variants.py
                  deep/tcn_exec_multitask.py     # multi-task execution model
  training/       orchestrate.py
  evaluation/     metrics.py distributional.py robustness.py bootstrap.py
  backtesting/    engine.py run.py
                  market_making/                 # fills, accounting, policies, control
  diagnostics/    tables.py figures.py monthly_report.py

configs/
  experiment/     mvp.yaml, btcusdt_top5_monthly.yaml, btcusdt_top5_monthly_real_12m.yaml, ...
  model/          tcn_exec_multitask.yaml, tcn_small.yaml, lightgbm.yaml, ...
scripts/          one CLI per pipeline stage + run_monthly_robustness.py (the full driver)
tests/            32 files, one per stage (260+ tests)
data_manifest/    storage schema, source manifests, SHA-256 checksums
TechnicalReport.pdf
```

---

## Pipeline

The pipeline is split into stages. Each stage reads files from the previous stage, writes its own outputs (Parquet for data, small YAML index files for metadata), and validates the output schema before downstream steps consume it. This makes it possible to rerun or replace individual stages, resume a run from any completed stage, and re-evaluate one run's frozen predictions from another without retraining.

| Stage | Package | What it does |
|---|---|---|
| Ingestion | `ingestion/` | Synthetic / local / URL / Tardis adapters; atomic writes; a SHA-256 manifest of every raw file |
| Normalisation | `normalisation/` | Raw messages to a canonical event table; venue adapters; quality flags (crossed book, missing levels, gaps) that survive downstream |
| Order book | `orderbook/` | Reconstructs the top-5 book; vectorised snapshot fast path (1272 s to 4 s per day, about 300x, verified identical); event-time striding |
| Features | `features/` | Causal microstructure features, regime descriptors, and a fitted linear-Gaussian latent state-space context |
| Labels | `labels/` | Forward return, 3-class direction with a spread-scaled neutral band, return quantiles, passive markout and adverse-selection targets |
| Datasets | `datasets/` | Expanding monthly walk-forward folds; embargo; leakage-safe sequence windows; per-fold feature scaler |
| Models | `models/`, `models/deep/` | Baselines, linear, LightGBM, direction-only TCN, multi-task execution-aware TCN (plus variants) |
| Training | `training/` | Orchestration, two-phase schedule, composite validation selection, crash-resumable checkpoints |
| Evaluation | `evaluation/` | Classification, return, and distributional metrics; moving-block bootstrap; cross-month robustness |
| Backtesting | `backtesting/`, `backtesting/market_making/` | Taker-cost engine; passive market-making simulator (policies, queue-aware fills, fill-probability model, control optimiser, accounting) |
| Diagnostics | `diagnostics/` | Assembles `reports/monthly_results.md` plus CSV tables and figures, with a fixed-rule verdict |

Intermediate outputs are stored as Parquet files and YAML metadata files. Stages use atomic writes, and raw inputs are tracked with SHA-256 manifests, so a half-written or altered input is caught before it is used.

---

## Models

Every model implements one interface (`fit`, `predict`, `save`, `load`, `hyperparameters`, and a `requires_sequences` flag) and registers itself by name. From the trivial baseline to the multi-task TCN, all of them write the same long-format prediction table (one row per event and horizon, with nullable columns for the execution heads only some models fill). The evaluation, backtest, and market-making layer reads this table and does not need to know which model produced a row.

| Model | Type | Role |
|---|---|---|
| `no_change` | trivial anchor | predicts neutral, zero return, and so reveals the majority-class bar per horizon |
| `imbalance_rule` | interpretable microstructure rule | trades on the sign of level-1 imbalance; threshold tuned on validation |
| `logistic_regression` | linear 3-class (tabular) | the linear direction ceiling on the full feature set |
| `ridge_regression` | linear return (tabular) | the linear return ceiling, and the only model with usable point-return skill |
| `lightgbm` | gradient-boosted trees | non-linear tabular reference |
| `tcn_small` | causal dilated TCN, direction-only | best direction model at `h=10` |
| **`tcn_exec_multitask`** | **multi-task TCN (sequence + context)** | **the multi-task sequence model used in the execution experiments** |

**The execution-aware multi-task TCN.** A shared causal dilated-convolution encoder feeds a learned gated temporal pooling layer, which is fused with the regime/context vector (and the latent SSM state) and read out by per-horizon heads that emit the quantities used by the passive-quoting policy:

- a point return (Huber),
- 3-class direction (cross-entropy),
- monotone q05 / q50 / q95 return quantiles (pinball loss; monotonicity guaranteed by cumulative softplus increments),
- bid/ask markout (Huber),
- bid/ask adverse-selection cost (softplus, so it stays non-negative).

The network is small (32 channels, 5 layers). Its main role is to output the quantities the execution layer uses, so that layer can read a sign, a calibrated interval, and a per-side markout and adverse estimate directly off the model.

The class also supports return-head ablations (return head on, off, gradient-detached, or replaced by a ridge sidecar), per-head loss weights, an optional two-phase schedule (joint training, then an encoder-frozen calibration pass), and latent-SSM variants. These are driven by a config-level experiment matrix.

**Latent state-space context.** A small time-invariant linear-Gaussian state-space model (4 latent states, 7 observations) fitted by a stable EM-style loop on training months only. The network receives the causal filtered states and their variances. Smoothed states are not used because they condition on later observations. The Kalman prior is reset at each monthly-day boundary to avoid carrying state across independent episodes.

---

## Evaluation protocol and leakage checks

The measured signal is small, so the evaluation uses explicit leakage checks, and each one is asserted by the test suite. The main restrictions are:

- **Calendar-discontinuity check.** Each first-of-month day is an independent episode. No feature window, label horizon, or sequence window crosses a day boundary. Admission is enforced with `O(1)` prefix-sum tests, and an embargo (at least the longest horizon) drops rows whose label would otherwise reach into the next split.
- **Causal fitted transforms.** The direction threshold εₕ and the regime-bucket edges are fitted once, on the earliest fold's training months, and then frozen. The feature scaler is re-fitted per fold on that fold's training rows only. Statistics used for transforms are computed from training rows only.
- **Join on time, not on `event_id`.** `event_id` resets per day, so all joins key on `(venue, symbol, timestamp_exchange_ns)`.
- **Fail-fast config.** A typed Pydantic schema validates the whole run once and rejects configurations outside the study's scope (non-BTCUSDT, `top_k` other than 5, non-event-time sampling, an embargo shorter than the longest horizon, market-making without markout targets, a date that is not the first of its month, and so on).
- **Expanding walk-forward.** 6 train / 1 validation / 1 test months, stepping one month at a time. Training begins at the earliest month and grows; each fold's test month becomes the next fold's validation month, so no month is used for both selection and reporting in the same fold.
- **Validation-only selection.** Model and policy choices are made on the validation month; the test month is used only for final reporting. The multi-task network selects on a composite validation score that balances direction skill against calibration and execution-head accuracy. This prevents selecting a checkpoint whose direction score is good but whose auxiliary heads are poorly trained.
- **Robustness reporting.** Results are summarised across held-out months (mean and standard deviation, best and worst month, fraction beating the per-month baseline) with moving-block-bootstrap confidence intervals that preserve short-range autocorrelation.

---

## Selected results

**Direction accuracy** (mean over three held-out months; read against the per-horizon majority baseline, since the neutral band collapses as the horizon grows):

| Model | *h*=10 | *h*=50 | *h*=200 | Beats baseline (h50) |
|---|:--:|:--:|:--:|:--:|
| *majority-class baseline* | 0.374 | 0.481 | 0.500 | n/a |
| `imbalance_rule` | 0.497 | **0.532** | 0.516 | 3 of 3 |
| `logistic_regression` | 0.515 | 0.531 | **0.517** | 3 of 3 |
| `lightgbm` | 0.529 | 0.519 | 0.508 | 3 of 3 |
| **`tcn_small`** | **0.533** | 0.531 | 0.513 | 3 of 3 |
| `tcn_exec_base` (multi-task) | 0.408 | 0.490 | 0.498 | 2 of 3 |
| `tcn_exec_ret0` (no return head) | 0.410 | 0.497 | 0.513 | 3 of 3 |
| `tcn_exec_ret0_ssm` (+ latent SSM) | 0.507 | 0.519 | 0.502 | 3 of 3 |

**Execution.** The ridge return model — the only model with usable point-return skill — has positive gross PnL, but net PnL is negative after a 5 bps taker fee (−26,740). The multi-task return head produces too many trades and nets −642,809, which is why the return head was removed in later variants. In the passive market-making simulator, every quoting policy posts a negative total reward, and the ranking is governed almost entirely by adverse-selection cost. A forecast-driven control optimiser, using markout, adverse-selection, fill-probability, and uncertainty estimates, beats the naive maker in 3 of 3 months but beats `no_quote` in 0 of 3. In these runs, the forecast mainly reduces quote frequency rather than improving realised reward.

The full tables, figures, regime breakdowns, and the development campaign are in the [Technical Report](TechnicalReport.pdf), and every run regenerates them into `reports/`.

---

## Reproducibility

- **Performance.** Vectorised snapshot reconstruction (about 300x, verified column-for-column identical); event-time striding applied once at the book stage so the features, backtest, and simulator share one sampled stream; column-pruned market-making frames; selective retraining with prediction reuse.
- **Self-contained runs.** Each run is written under its own `run_id` (frozen resolved config, per-fold scaler and split definitions, model checkpoints, the long prediction table, structured per-epoch logs, market-making artefacts), so independent runs sit side by side on disk and can be compared without re-execution.
- **Resuming after a crash.** Runs can resume at three levels: skip completed folds and models, restore the multi-task model from a per-epoch checkpoint, and reuse existing fold datasets, scalers, and latent-state files.
- **Reproducible inputs.** Raw files are immutable after ingestion and SHA-256-checksummed; every derived table is reproducible from the raw files and the config.
- **Testing.** 260+ tests, one file per stage. The tests focus on errors that would affect the empirical result: no window or label crosses a monthly day, every fitted transform sees only training rows, the multi-task save/load reproduces predictions and keeps its quantiles monotone, and policy selection touches only the validation split.

---

## Scope and limitations

This repository is intended for research and reproduction, not live trading. The reported results are limited to one symbol, one venue, top-5 depth, event-time sampling, and twelve first-of-month snapshots. The held-out evaluation covers three test months, which is more informative than a single split but still a handful of regime snapshots rather than continuous history, so the cross-month standard deviations are indicative rather than tight.

The stream is thinned by event striding (a horizon of 50 events is roughly 500 raw updates, on the order of tens of seconds). The passive fill model is approximate because top-5 snapshots do not reveal true queue position, so the queue-aware fill logic should be read as an execution proxy, not as observed fills. The decision rules are also intentionally simple: a 4-state linear filter and a transparent action grid, not a learned policy.

No live-trading profitability is claimed.
