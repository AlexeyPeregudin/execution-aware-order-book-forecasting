"""End-to-end monthly robustness pipeline.

    python scripts/run_monthly_robustness.py --config configs/experiment/btcusdt_top5_monthly.yaml

Runs ingestion -> normalisation -> order books -> features/labels once, then for
each expanding monthly fold: builds fold datasets, trains the models (+ TCN
pooling ablations), computes month/regime/distributional metrics, runs the taker
sanity backtest and the passive market-making simulator (validation-only
selection), and finally writes reports/monthly_results.md with block-bootstrap
confidence intervals.

Use --quick to shrink epochs/bootstrap for a fast smoke run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# so we can run this without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from lob_forecasting.backtesting import run_backtest, run_market_making
from lob_forecasting.config import generate_run_id, load_config, parse_cli_overrides, save_resolved_config
from lob_forecasting.datasets import DatasetIndex, build_datasets, generate_folds
from lob_forecasting.diagnostics.monthly_report import MonthlyReportData, build_monthly_report
from lob_forecasting.evaluation import (
    bootstrap_from_config,
    distributional_metrics,
    load_context,
    month_stability_summary,
    monthly_and_regime_metrics,
)
from lob_forecasting.evaluation import metrics as M
from lob_forecasting.features import build_features
from lob_forecasting.ingestion import RawDataManifest, ingest_raw_data
from lob_forecasting.labels import LabelledTableIndex, build_labels
from lob_forecasting.normalisation import EventTableIndex, normalise_events
from lob_forecasting.orderbook import BookTableIndex, build_order_books
from lob_forecasting.training import train_and_predict

MULTITASK = "tcn_exec_multitask"


def _ensure_features(config, root: Path, run_id: str) -> LabelledTableIndex:
    """Run the data stages up to features+labels if not already present."""
    labelled_path = root / config.data.processed_dir.parent / "features" / "labelled_index.yaml"
    if labelled_path.exists():
        lab = LabelledTableIndex.load(labelled_path)
        if len(lab):
            print(f"[pipeline] reusing existing features/labels ({lab.total_rows()} rows)")
            return lab

    print("[pipeline] ingesting raw data")
    ingest_raw_data(config, root)
    manifest = RawDataManifest.load(root / config.ingestion.manifest_path)
    print("[pipeline] normalising events")
    normalise_events(config, manifest, root)
    events = EventTableIndex.load(root / config.data.processed_dir / "events" / "events_index.yaml")
    print("[pipeline] reconstructing order books (top-5)")
    build_order_books(config, events, root)
    books = BookTableIndex.load(root / config.data.processed_dir / "books" / "books_index.yaml")
    print("[pipeline] building features + labels (per-day, regime + markout)")
    features = build_features(config, books, root)
    return build_labels(config, features, run_id, root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monthly robustness pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", help="Use a specific run id (default: generate one).")
    parser.add_argument("--quick", action="store_true", help="Fewer epochs / bootstrap for a smoke run.")
    parser.add_argument("--tcn-epochs", type=int, help="Override epochs for the TCN models.")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable TCN early stopping (skips per-epoch validation predict).")
    parser.add_argument("--no-ablation", action="store_true", help="Skip the TCN pooling ablation variant.")
    parser.add_argument("--exec-variants", action="store_true",
                        help="Train the execution-aware multi-task variants (return-head ablations, "
                             "+ latent-SSM variants when latent_state is enabled).")
    parser.add_argument("--variants", help="Comma-separated subset of variant output names to run "
                        "(default: the full matrix). Implies --exec-variants.")
    parser.add_argument("--folds", help="Comma-separated fold ids to run (default: all).")
    parser.add_argument("--phase-a-epochs", type=int, help="Override variant two-phase Phase A max epochs.")
    parser.add_argument("--phase-b-epochs", type=int, help="Override variant two-phase Phase B epochs.")
    parser.add_argument("--mm-model", default=MULTITASK,
                        help="Prediction model name that drives the market-making simulator "
                             "(default: tcn_exec_multitask; use a variant name when the base isn't trained).")
    parser.add_argument("--no-market-making", action="store_true", help="Skip the market-making simulator.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run: skip models/datasets already on disk and "
                             "continue any partially-trained deep model from its last epoch checkpoint.")
    parser.add_argument("--models", help="Comma-separated list of models to train (others reused from --reuse-predictions-from).")
    parser.add_argument("--reuse-predictions-from", dest="reuse_from", help="Run ID whose saved predictions to reuse for models not in --models.")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.overrides)
    config, _ = load_config(args.config, overrides=overrides)
    if not config.splits.is_monthly:
        print("[pipeline] config is not a monthly split mode; aborting.")
        return 1
    for w in config.config_warnings():
        print(f"[pipeline] WARNING: {w}")

    root = Path.cwd()
    run_id = args.run_id or generate_run_id()
    save_resolved_config(config, run_id, root)
    print(f"[pipeline] run_id={run_id}")

    from lob_forecasting.utils import log

    log("[pipeline] building features + labels (once)")
    labelled = _ensure_features(config, root, run_id)
    books = BookTableIndex.load(root / config.data.processed_dir / "books" / "books_index.yaml")
    log(f"[pipeline] loading context + features-labels ({labelled.total_rows():,} rows)")
    context = load_context(config, root)
    # only the columns the market-making simulator needs (keeps memory bounded on
    # multi-million-row real data)
    mm_cols = [
        "event_id", "timestamp_exchange_ns", "monthly_date", "mid", "spread",
        "bid_px_1", "ask_px_1", "imbalance_l1", "imbalance_lK",
        "ofi_10", "ofi_50", "ofi_200", "relative_spread",
        "return_lag_10", "realised_vol_200", "regime_depth",
        "vol_regime", "spread_regime", "liq_regime", "time_of_day_bucket",
    ]
    # the queue-aware fill model and control optimiser need the raw top-K levels
    k = config.data.top_k
    level_cols = [f"{stem}_{i}" for stem in ("bid_px", "bid_qty", "ask_px", "ask_qty")
                  for i in range(1, k + 1)]
    mm_cols = list(dict.fromkeys(mm_cols + level_cols))

    import pyarrow.parquet as pq

    def _read_mm(path: Path) -> pd.DataFrame:
        avail = set(pq.ParquetFile(path).schema.names)
        cols = [c for c in mm_cols if c in avail]
        return pd.read_parquet(path, columns=cols)

    fl_full = pd.concat(
        [_read_mm(root / lp.file_path) for lp in labelled.partitions],
        ignore_index=True,
    )

    folds = generate_folds(config)
    if args.folds:
        wanted = {int(x) for x in args.folds.split(",")}
        folds = [f for f in folds if f.fold_id in wanted]
    log(f"[pipeline] {len(folds)} monthly fold(s): {[f.fold_id for f in folds]}")

    # TCN training overrides (epochs / early stopping) without touching the model yamls
    tcn_overrides: dict = {}
    if args.quick:
        tcn_overrides["epochs"] = 2
    if args.tcn_epochs is not None:
        tcn_overrides["epochs"] = args.tcn_epochs
    if args.no_early_stop:
        tcn_overrides["early_stopping"] = False
    model_names = list(config.models.run)
    models_to_train = set(args.models.split(",")) if args.models else set(model_names)
    reuse_from = args.reuse_from  # run ID to load saved predictions from, or None

    monthly_all, regime_all, dist_all, taker_all, policy_all = [], [], [], [], []
    inventory_all: list[pd.DataFrame] = []

    for fold in folds:
        log(f"[fold {fold.fold_id}] train={[d.isoformat() for d in fold.train_dates]} "
            f"val={[d.isoformat() for d in fold.validation_dates]} "
            f"test={[d.isoformat() for d in fold.test_dates]}")
        subdir = f"folds/{fold.name}"
        meta_path = root / config.data.artefact_dir / "runs" / run_id / subdir / "dataset_metadata.yaml"
        if args.resume and meta_path.exists():
            from lob_forecasting.datasets import DatasetIndex
            idx = DatasetIndex.load(meta_path)
            log(f"[fold {fold.fold_id}] [resume] reusing existing datasets")
        else:
            log(f"[fold {fold.fold_id}] building datasets")
            idx = build_datasets(config, labelled, run_id, root, fold=fold)
        for s in idx.stats:
            log(f"  {s.split}: {s.n_tabular:,} tabular / {s.n_sequence:,} sequence windows")

        predictions: list[pd.DataFrame] = []
        # reuse saved predictions for models not in the training set
        if reuse_from:
            reuse_dir = root / config.data.artefact_dir / "runs" / reuse_from / f"folds/{fold.name}" / "predictions"
            for name in model_names:
                if name not in models_to_train:
                    pred_path = reuse_dir / f"{name}.parquet"
                    if pred_path.exists():
                        p = pd.read_parquet(pred_path)
                        p["fold_id"] = fold.fold_id
                        predictions.append(p)
                        log(f"  reused {name} predictions from {reuse_from} ({len(p):,} rows)")
                    else:
                        log(f"  WARNING: no saved predictions for {name} in {reuse_from}/{fold.name}; skipping")
        train_names = [n for n in model_names if n in models_to_train]
        for i, name in enumerate(train_names, 1):
            ov = dict(tcn_overrides) if (tcn_overrides and name.startswith("tcn")) else None
            log(f"[fold {fold.fold_id}] training model {i}/{len(train_names)}: {name}")
            preds = train_and_predict(config, idx, name, root, overrides=ov, run_subdir=subdir,
                                      resume=args.resume)
            predictions.append(preds)
            log(f"  {name}: {len(preds):,} pred rows")
        # pooling ablation for the multi-task model (last-step vs gated default)
        if MULTITASK in models_to_train and not args.no_ablation:
            ov = {"pooling": "last_step", **tcn_overrides}
            log(f"[fold {fold.fold_id}] training tcn_exec_multitask_last_step (ablation)")
            preds_ls = train_and_predict(config, idx, MULTITASK, root, overrides=ov,
                                         run_subdir=subdir, output_name=f"{MULTITASK}_last_step",
                                         resume=args.resume)
            predictions.append(preds_ls)

        # execution-aware variants: return-head ablations (+ latent-SSM variants)
        if args.exec_variants or args.variants:
            from lob_forecasting.models import experiment_matrix, merge_ridge_sidecar
            ridge_pred = next((p for p in predictions
                               if len(p) and str(p["model_name"].iloc[0]) == "ridge_regression"), None)
            variants = experiment_matrix(include_ssm=config.latent_state.enabled,
                                         two_phase=not args.quick)
            if args.variants:
                wanted_v = set(args.variants.split(","))
                variants = [v for v in variants if v.output_name in wanted_v]
            for v in variants:
                ov = _variant_overrides(v, tcn_overrides, args.quick,
                                        phase_a=args.phase_a_epochs, phase_b=args.phase_b_epochs)
                log(f"[fold {fold.fold_id}] training variant {v.output_name} "
                    f"(ssm={v.needs_ssm}, return={v.return_source})")
                preds_v = train_and_predict(config, idx, v.base_model, root, overrides=ov,
                                            run_subdir=subdir, output_name=v.output_name,
                                            include_latent_context=v.needs_ssm, resume=args.resume)
                if v.return_source == "ridge_sidecar" and ridge_pred is not None:
                    preds_v = merge_ridge_sidecar(preds_v, ridge_pred)
                    pred_path = (root / config.data.artefact_dir / "runs" / run_id / subdir
                                 / "predictions" / f"{v.output_name}.parquet")
                    preds_v.to_parquet(pred_path, engine="pyarrow", index=False)
                predictions.append(preds_v)

        # tag predictions with the fold and collect month/regime/distributional metrics
        for p in predictions:
            p["fold_id"] = fold.fold_id
        log(f"[fold {fold.fold_id}] computing monthly + regime + distributional metrics")
        mo, rg = monthly_and_regime_metrics(config, predictions, fold.fold_id, context=context)
        dist = distributional_metrics(config, predictions, fold.fold_id, context=context)
        monthly_all.append(mo); regime_all.append(rg); dist_all.append(dist)

        if not any(len(p) for p in predictions):
            log(f"[fold {fold.fold_id}] WARNING: all predictions empty (no usable rows for these dates); skipping backtests")
            continue

        # taker sanity backtest (test + validation) for this fold
        if config.backtest.run_taker_sanity:
            log(f"[fold {fold.fold_id}] taker sanity backtest")
            res = run_backtest(config, predictions, books, root, write=False)
            tm = res.metrics.copy()
            if len(tm):
                tm["fold_id"] = fold.fold_id
                taker_all.append(tm)

        # passive market-making simulator on the chosen execution-aware model
        multitask_preds = [p for p in predictions
                           if len(p) and str(p["model_name"].iloc[0]) == args.mm_model] if predictions else []
        if config.market_making.enabled and not args.no_market_making and multitask_preds:
            log(f"[fold {fold.fold_id}] market-making simulator (decision_interval={config.market_making.decision_interval})")
            mt_pred = multitask_preds[0]
            fl_mm = _merge_latent_for_policy(fl_full, idx, config, root)
            mm_res = run_market_making(config, fold, mt_pred, fl_mm, run_id, root)
            mm_res.save(root / config.data.artefact_dir / "runs" / run_id / subdir / "market_making")
            policy_all.append(mm_res.policy_metrics)
            inv_test = mm_res.inventory[mm_res.inventory["monthly_date"].isin(
                [d.isoformat() for d in fold.test_dates])]
            inventory_all.append(inv_test)
            print(f"  market-making: {len(mm_res.policy_metrics)} policy-metric rows")

    # aggregate
    monthly_metrics = _concat(monthly_all)
    regime_metrics = _concat(regime_all)
    distributional = _concat(dist_all)
    taker_metrics = _concat(taker_all)
    policy_metrics = _concat(policy_all)
    baseline = _baseline_accuracy(monthly_metrics, context, root, config)
    stability = month_stability_summary(monthly_metrics, baseline_accuracy=baseline)

    # write aggregated metric artefacts
    metrics_dir = root / config.data.artefact_dir / "runs" / run_id / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _write(monthly_metrics, metrics_dir / "monthly_predictive_metrics.parquet")
    _write(regime_metrics, metrics_dir / "regime_predictive_metrics.parquet")
    _write(distributional, metrics_dir / "monthly_distributional_metrics.parquet")
    _write(stability, metrics_dir / "monthly_robustness_summary.parquet")

    bootstrap = _bootstrap_headline(config, monthly_metrics, policy_metrics, predictions_context=context,
                                    quick=args.quick)
    label_dist = _label_distribution_table(labelled)
    data_summary = _data_summary(labelled, books)

    data = MonthlyReportData(
        folds=[f.as_dict() for f in folds],
        monthly_metrics=monthly_metrics, regime_metrics=regime_metrics,
        distributional=distributional, stability=stability,
        taker_metrics=taker_metrics, policy_metrics=policy_metrics,
        bootstrap=bootstrap, data_summary=data_summary,
        label_distribution=label_dist, baseline_accuracy=baseline,
        inventory_paths=_concat(inventory_all),
    )
    out = build_monthly_report(config, run_id, data, root)
    print(f"\n[pipeline] report -> {out}")
    print(f"[pipeline] tables -> reports/tables/  figures -> reports/figures/")
    return 0


def _variant_overrides(variant, tcn_overrides: dict, quick: bool,
                       phase_a: int | None = None, phase_b: int | None = None) -> dict:
    """Variant params, shrunk to a fast two-phase schedule under --quick.

    `phase_a` / `phase_b` cap the two-phase epoch budgets (for fitting a long
    run into a time budget); min_epochs/patience scale down with Phase A.
    """
    ov = dict(variant.overrides)
    if quick:
        ov["two_phase"] = {
            "enabled": True,
            "phase_a": {"epochs": 2, "min_epochs": 1, "patience": 2,
                        "learning_rate": 5e-4, "weight_decay": 5e-4},
            "phase_b": {"epochs": 1, "learning_rate": 1e-4, "weight_decay": 1e-5,
                        "weights": {"direction": 0.0, "quantile": 2.0, "markout": 1.0, "adverse": 1.0}},
        }
    elif (phase_a or phase_b) and ov.get("two_phase", {}).get("enabled"):
        tp = ov["two_phase"]
        if phase_a:
            tp["phase_a"] = {**tp["phase_a"], "epochs": phase_a,
                             "min_epochs": min(tp["phase_a"]["min_epochs"], max(1, phase_a // 2)),
                             "patience": min(tp["phase_a"]["patience"], max(2, phase_a // 2))}
        if phase_b:
            tp["phase_b"] = {**tp["phase_b"], "epochs": phase_b}
    # epoch overrides from the CLI apply to the single-phase fallback only
    for k in ("epochs",):
        if k in tcn_overrides and not ov.get("two_phase", {}).get("enabled"):
            ov[k] = tcn_overrides[k]
    return ov


def _merge_latent_for_policy(fl: pd.DataFrame, idx, config, root: Path) -> pd.DataFrame:
    """Append the fold's filtered SSM columns to the MM features frame (policy state)."""
    ls = config.latent_state
    if not (ls.enabled and ls.append_to_policy_state and getattr(idx, "latent_context_path", "")):
        return fl
    path = root / idx.latent_context_path
    if not path.exists():
        return fl
    ctx = pd.read_parquet(path)
    ssm_cols = [c for c in ctx.columns if c.startswith("ssm_z_") or c.startswith("ssm_var_")]
    ctx = ctx[["timestamp_exchange_ns", *ssm_cols]].drop_duplicates("timestamp_exchange_ns")
    return fl.merge(ctx, on="timestamp_exchange_ns", how="left")


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _write(df: pd.DataFrame, path: Path) -> None:
    if df is not None and len(df):
        df.to_parquet(path, index=False)


def _baseline_accuracy(monthly_metrics, context, root, config) -> pd.DataFrame:
    """Majority-class accuracy per test month/horizon, from the true-direction mix."""
    if monthly_metrics.empty:
        return pd.DataFrame()
    fl = context  # has timestamp + regimes but not labels; load labels instead
    rows = []
    for path in (root / config.data.processed_dir.parent / "features").glob(
        "venue=*/symbol=*/features_labels.parquet"
    ):
        df = pd.read_parquet(path, columns=["monthly_date", "y_dir_h50", "label_available_h50"])
        df = df[df["label_available_h50"]]
        for md, g in df.groupby("monthly_date"):
            counts = g["y_dir_h50"].value_counts(normalize=True)
            rows.append({"monthly_date": str(md), "baseline_accuracy": float(counts.max()) if len(counts) else 0.5})
    return pd.DataFrame(rows)


def _label_distribution_table(labelled: LabelledTableIndex) -> pd.DataFrame:
    rows = []
    for p in labelled.partitions:
        for h, d in p.label_distribution.items():
            rows.append({"symbol": p.symbol, "horizon": h, **d})
    return pd.DataFrame(rows)


def _data_summary(labelled: LabelledTableIndex, books: BookTableIndex) -> pd.DataFrame:
    rows = [{"stage": "features_labels_rows", "count": labelled.total_rows()},
            {"stage": "book_rows", "count": books.total_rows()},
            {"stage": "book_partitions", "count": len(books)}]
    return pd.DataFrame(rows)


def _bootstrap_headline(config, monthly_metrics, policy_metrics, predictions_context, quick) -> pd.DataFrame:
    """Block-bootstrap CIs for accuracy/rank_ic per test month from monthly metrics.

    Operates on the per-month metric values across folds (treating each test-month
    observation as a sample), giving a coarse stability interval per model/metric.
    """
    cfg = config.robustness.block_bootstrap
    if not cfg.enabled or monthly_metrics.empty:
        return pd.DataFrame()
    rows = []
    test = monthly_metrics[(monthly_metrics["split"] == "test")
                           & (monthly_metrics["metric_name"].isin(["accuracy", "rank_ic", "r2_oos"]))]
    bs = config.robustness.block_bootstrap
    n_boot = 100 if quick else bs.n_bootstrap
    for (model, metric, horizon), g in test.groupby(["model_name", "metric_name", "horizon"]):
        vals = g.dropna(subset=["metric_value"])[["metric_value"]].reset_index(drop=True)
        if len(vals) < 2:
            continue
        res = bootstrap_from_config(
            vals, lambda d: float(d["metric_value"].mean()),
            type(bs)(enabled=True, n_bootstrap=n_boot, block_size_events=1,
                     confidence_level=bs.confidence_level),
        )
        rows.append({"model_name": model, "metric_name": metric, "horizon": int(horizon),
                     "point": round(res["point"], 4), "lower": round(res["lower"], 4),
                     "upper": round(res["upper"], 4), "n_months": res["n"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
