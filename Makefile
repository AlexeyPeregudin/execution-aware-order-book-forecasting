CONFIG ?= configs/experiment/mvp.yaml
MONTHLY_CONFIG ?= configs/experiment/btcusdt_top5_monthly.yaml

.PHONY: download-sample-data normalise-events build-orderbooks \
        build-features-labels build-datasets \
        train-baselines train-deep-model \
        evaluate backtest report \
        monthly monthly-market-making \
        test mvp clean

## ── Pipeline steps ───────────────────────────────────────────────────────────

download-sample-data:
	python scripts/download_sample_data.py --config $(CONFIG)

normalise-events:
	python scripts/normalise_events.py --config $(CONFIG)

build-orderbooks:
	python scripts/build_orderbooks.py --config $(CONFIG)

build-features-labels:
	python scripts/build_features_labels.py --config $(CONFIG)

build-datasets:
	python scripts/build_datasets.py --config $(CONFIG)

train-baselines:
	python scripts/train_model.py --config $(CONFIG) --model no_change
	python scripts/train_model.py --config $(CONFIG) --model imbalance_rule
	python scripts/train_model.py --config $(CONFIG) --model logistic_regression
	python scripts/train_model.py --config $(CONFIG) --model ridge_regression
	python scripts/train_model.py --config $(CONFIG) --model lightgbm

train-deep-model:
	python scripts/train_model.py --config $(CONFIG) --model tcn_small

evaluate:
	python scripts/evaluate_predictions.py --config $(CONFIG)

backtest:
	python scripts/run_backtest.py --config $(CONFIG)

report:
	python scripts/make_report_assets.py --config $(CONFIG)

## ── Full MVP run ─────────────────────────────────────────────────────────────

mvp: download-sample-data \
     normalise-events \
     build-orderbooks \
     build-features-labels \
     build-datasets \
     train-baselines \
     train-deep-model \
     evaluate \
     backtest \
     report

## ── Monthly robustness extension ─────────────────────────────────────────────

# End-to-end: ingestion -> features/labels -> per-fold train/eval -> taker +
# market-making backtests -> reports/monthly_results.md. Add QUICK=1 for a fast
# smoke run (fewer TCN epochs / bootstrap samples).
monthly:
	python scripts/run_monthly_robustness.py --config $(MONTHLY_CONFIG) $(if $(QUICK),--quick,)

monthly-market-making:
	python scripts/run_market_making.py --config $(MONTHLY_CONFIG)

## ── Tests ────────────────────────────────────────────────────────────────────

test:
	pytest tests/

## ── Utilities ────────────────────────────────────────────────────────────────

clean:
	rm -rf data/interim data/processed data/features data/datasets artefacts reports/figures/* reports/tables/*
