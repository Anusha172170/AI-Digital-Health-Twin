# Reproducible clinical benchmark pipeline

Completed: source downloads, schema validation, independent preprocessing, train/validation/test exports, baseline and XGBoost training, saved inference bundles, TreeSHAP, evaluation plots and artifact tests.

## Run from repository root (Python 3.11)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
pip install -r ml_engine/requirements.txt
python -m ml_engine.src.load_datasets
python -m ml_engine.src.preprocess
python -m ml_engine.src.train
python -m pytest tests/ml_tests -q
python -m ml_engine.src.predict diabetes ml_engine/examples/diabetes.json
```

All paths are module-relative. Existing downloaded data lets preprocessing and training run offline. Downloader uses HTTPS with a GitHub CLI API fallback for restricted networks; it fails rather than silently generating synthetic data. Exact tested environment is pinned in requirements.txt (Linux CPU training; GPU is not required).

## Tracked outputs

- `datasets/raw/*_source.csv`: exact downloaded bytes; PIMA source has no header.
- `datasets/raw/diabetes.csv`, `framingham.csv`: same records with standardized CSV serialization; PIMA given descriptive headers.
- `datasets/raw/manifest.json`: immutable source URLs, commit hashes, counts, download timestamps and SHA-256 checksums.
- `datasets/processed/{task}/`: train, validation and test CSVs, their `*_scaled.csv` counterparts, and metadata. Source-row IDs preserve auditability; never used as predictors.
- `saved_models/{task}_preprocessor.pkl`: training-fitted median imputer plus StandardScaler.
- `saved_models/{task}_scaler.pkl`: scaler only, for inspection. **Use the full preprocessor for inference.**
- `saved_models/{task}_model.pkl`: selected model bundle, feature order, target, meaning, threshold and preprocessor.
- `saved_models/{task}_{dummy,logistic,xgboost}.pkl`: all trained candidates.
- `saved_models/{task}_xgboost.json`: portable XGBoost model (still requires matching preprocessing).
- `reports/{task}/`: metrics JSON, held-out predictions, confusion matrix/ROC/PR/calibration plots, actual XGBoost SHAP CSV and plot.
- `saved_models/sha256.json`: artifact integrity hashes.

Do not scale twice: `predict` takes **raw** values. Missing keys are rejected; explicit nulls are median-imputed and identified in the response. Only load trusted pickle/joblib files; they can execute code. Pin dependencies when loading them.

## Data sources and restrictions

See [dataset provenance](datasets/DATA_SOURCES.md). These are clinical benchmark mirrors, not live wearable data and not direct original-institution downloads. They do not contain a common cohort. We do not join unrelated patients or add random sleep, activity or oxygen values. There is intentionally no `unified_health_data.csv` or universal `scaler.pkl`.

## Statistical design

Exact duplicate source rows are removed before a fixed stratified 60/20/20 split (seed 42). Zero-coded missing measurements are converted to NaN. Medians and scaling are fitted on training only. All features use task-specific definitions; XGBoost does not require scaling, but scaled exports support the logistic baseline consistently. Model selection uses validation average precision; threshold uses validation F1. Neither uses test results. All candidates use fixed hyperparameters; no claimed exhaustive optimization or calibration.

The selected baseline may outperform XGBoost on validation; the pipeline reports that instead of claiming an advanced algorithm is always superior. SHAP outputs are specifically for the **XGBoost candidate**, in log-odds units, not percentage-point disease contributions.

## Not completed / not clinically supported

- Prospective hypertension prediction: this dataset only supports an exploratory `prevalentHyp` target. Direct BP and medication proxies and future CHD are excluded from its predictors.
- Validated biological age, causal sleep/step/diet effects, PPG-to-BP models, wearable integration, backend deployment or mobile inference wiring.
- External population validation (especially users in India), clinical calibration, regulatory approval, emergency triage or clinical performance guarantees.

The diabetes target is recorded status, not a future-onset horizon. CVD means ten-year **coronary heart disease**, not all cardiovascular outcomes. PIMA glucose is the two-hour oral glucose-tolerance measurement, not interchangeable with fasting glucose or CGM. Do not feed a watch's resting HR into exercise-HR fields or invent missing clinical tests. PIMA's demographic scope limits generalization. Each input must match training units and measurement context.

## Wearable extension

The separate [WISDM watch activity pipeline](wearable/README.md) now includes acquired source window features, participant-held-out splits, scaled CSVs, trained candidates and SHAP/evaluation artifacts. It is not merged with clinical patients and is not a live-device connector.

## Model improvement experiments

[V2 experiments](experiments/v2/README.md) compare tuned logistic/tree ensembles and train-CV sigmoid calibration. Results are mixed; no consistent improvement is established. Baseline artifacts are unchanged. Experimental bundles require explicit `predict_experiment` usage.

## Validation readiness and operating points

Run `python -m ml_engine.src.threshold_analysis` for research-only validation-constrained operating points. [Readiness documentation](../docs/validation/README.md) includes reviewed dataset candidates, software input contracts and a frozen external evaluator; actual independent clinical validation is still pending compatible data. These tools do not alter baseline models or connect external applications.

## End-to-end training completion

The wearable and V2 pipelines have been rerun and verified in this checkout. See the [completion report and execution evidence](reports/completion/README.md).

```bash
python -m ml_engine.src.complete_training
# Optional pinned wearable source reacquisition:
python -m ml_engine.src.complete_training --download
# Evaluate persisted artifacts only (no training):
python -m ml_engine.src.verify_saved_models
```

The clinical baselines are unchanged; V2 remains experimental. Wearable training/evaluation now uses the identical raw-input preprocessing path as inference, with full-test prediction parity verification.
