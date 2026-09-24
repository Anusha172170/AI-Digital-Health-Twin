# Wearable activity-recognition pipeline

## Source and attribution

Weiss, G. (2019). *WISDM Smartphone and Smartwatch Activity and Biometrics Dataset*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5HK59.

Official documentation/license: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset (CC BY 4.0). License: https://creativecommons.org/licenses/by/4.0/ . Attribution retained; conversion, scaling, modeling and derived outputs are modifications by this project, not endorsements by the dataset creators.

Direct UCI archive access failed in this environment. Acquired via GitHub API from `kseniacold/aai_530_dataset_static`, pinned commit `e70750583775ab997853a9aca57c4a45d934346c`. Exact source bytes, paths and SHA-256 checksums are in `raw/manifest.json`. Mirror equivalence to the entire official archive has not been independently verified.

**Coverage:** all 50 watch-accelerometer ARFF files at that commit; participant 1614 is absent. These are the dataset's published window features, NOT raw XYZ sample streams. Phone streams, gyroscope streams, the missing participant and raw motion streams were not downloaded. Source README and activity labels are retained.

## What is trained

18-class activity recognition from 91 published smartwatch-accelerometer features. The source `class` field contains participant identity: it is explicitly removed from inputs. The target is `ACTIVITY`. No subject/window IDs are included as predictors.

18,211 feature windows; 30 participants for training (10,788 windows), 10 for validation (4,177), 10 for test (3,246). Group splitting prevents one person's windows from appearing in multiple splits. Actual subject IDs are recorded in `processed/metadata.json`.

Median imputation and StandardScaler fit only on training. Dummy, logistic regression and XGBoost candidates use fixed settings. Selection uses validation macro F1; test scores never drive selection. No disease labels or synthetic physiology are added. Nonfinite features are converted to missing before imputation; no physiological targets are inferred.

## Reproduce from repository root

Use the existing Python 3.11 virtual environment and `ml_engine/requirements.txt`. Downloads additionally require authenticated GitHub CLI access; existing tracked data supports offline preprocessing/training.

```bash
python -m ml_engine.src.wearable_pipeline download
python -m ml_engine.src.wearable_pipeline preprocess
python -m ml_engine.src.wearable_pipeline train
python -m ml_engine.src.wearable_pipeline predict --input ml_engine/wearable/example_features.json
python -m pytest tests/ml_tests tests/wearable_tests -q
```

## Tracked artifacts

- `raw/`: 50 original ARFF feature files, source README/activity key, provenance manifest.
- `wearable_dataset.csv`: converted unscaled window-feature dataset, including labels and audit IDs.
- `processed/`: train/validation/test CSVs and scaled equivalents, plus feature/split metadata. CSV exports retain floating-point precision. Training and inference both transform raw-unit inputs through the same saved preprocessor; scaled exports are for inspection, not a second inference input.
- `saved_models/`: selected `wearable_activity_model.pkl` inference bundle, individual candidates, scaler, preprocessor, XGBoost JSON and checksums.
- `reports/`: validation/test metrics, per-class report, per-participant metrics, probability predictions, confusion matrix and actual TreeSHAP outputs.
- `example_features.json`: one source-derived held-out input example, not fabricated sensor data.

Only load trusted pickle files. Inference requires the exact raw-unit published feature vector; preprocessing is applied once. SHAP explains the XGBoost candidate's **raw class margins**, not direct probabilities or causal effects. Its sample of 20 test windows is illustrative, not a population-level explanation audit.

## Limits / connection to the health twin

This can support an experimental activity-context module. It cannot predict diabetes, CHD, hypertension, sleep quality, blood oxygen or biological age. It has not been validated on the user's watch, ring or free-living population. HealthKit/Health Connect daily summaries are not valid inputs to this model. Raw-to-feature extraction and live SDK/OAuth integration remain separate tasks requiring device specifications and parity testing.

Results: [measured metrics](reports/model_metrics.md). Activity-code meanings: [source activity key](raw/activity_key.txt).
