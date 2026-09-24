# Model-training milestone: completed and verified

## Scope

The **offline wearable activity-recognition pipeline** and **V2 clinical model experiments** have been executed in this checkout. This is completion of model training/evaluation, not completion of health-app integration or clinical validation.

The earlier branch history was recovered from GitHub and merged with the local clinical baseline commit. Then source acquisition, preprocessing, training, evaluation and artifact verification were run again. No baseline clinical artifact was replaced.

## Wearable data and models

- Pinned public WISDM mirror: all 50 watch-accelerometer ARFF feature files; original bytes and attribution retained. Subject 1614 is absent in the mirror; not the complete raw WISDM release.
- 18,211 windows, 91 published features, 18 activity classes.
- Participant-separated split: 30 training / 10 validation / 10 test participants. Identity and target columns excluded from inputs.
- Original ARFF, raw-unit CSV, split CSVs, scaled CSVs, fitted preprocessor and scaler, three trained candidate models and selected bundle are tracked.
- XGBoost selected by validation macro F1. Held-out accuracy **0.657**, macro F1 **0.655**.
- Per-class/per-participant metrics, probability predictions, confusion matrix and actual XGBoost TreeSHAP outputs included.

### Correctness fix during this run

Full-test inference verification exposed that training/evaluating rounded scaled CSVs could differ from raw-input inference at tree thresholds. The corrected pipeline preserves export precision and trains/evaluates through the same saved preprocessing transform used for raw-input inference. Verification now checks probabilities and labels across **all 3,246 wearable test windows**, not merely the example row. Inspecting scaled CSVs does not mean they should be passed to raw-input inference.

## V2 clinical experiments

Three tasks × five algorithm families × eight configurations × three training folds = **360 CV fits**, plus final refits/calibration. All best-per-family candidate pipelines are saved, alongside each selected calibrated bundle. Search tables, parameters, source feature order, validation-based selection, test predictions, class reports, PR/calibration plots, permutation importance and bootstrap intervals are included.

| Task | Selected V2 model | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Diabetes status | Extra Trees + sigmoid | 0.791 | 0.537 | 0.811 | 0.647 |
| Ten-year CHD | Histogram gradient boosting + sigmoid | 0.678 | 0.216 | 0.682 | 0.328 |
| Existing hypertension | Extra Trees + sigmoid | 0.732 | 0.441 | 0.726 | 0.549 |

**Decision: retain baseline clinical models.** V2 F1 changes are small, probability errors worsen slightly on the historical test set, and paired bootstrap intervals do not establish reliable improvement. No test-driven promotion, clinical-risk claims, future hypertension labels or causal lifestyle claims.

## Verification evidence

- `run_manifest.json`: successful final execution; stage commands, exit codes, durations, Python/package versions, training-source hashes, dataset/output hashes and baseline-unchanged verification.
- `tests.log`: **43 passed**, three third-party SHAP/matplotlib pending-deprecation warnings.
- `saved_model_verification.log`: metrics recomputed from loaded wearable/V2 bundles without fitting; compared against persisted predictions and reports.
- `download.log`: successful pinned-mirror reacquisition from the preceding execution. The final complete run reused and checksum-verified those source files.
- `wearable_preprocess.log`, `wearable_train.log`, `clinical_v2_train.log`: execution logs.

Timing and timestamps in the manifest come from the execution environment clock. It records the starting Git revision and source hashes; it is not a claim that the subsequently generated outputs were present in that starting commit.

## Reproduce from repository root

Python 3.11 environment:

```bash
python -m venv .venv
# Activate the environment for your OS, then:
pip install -r ml_engine/requirements.txt
python -m ml_engine.src.complete_training
```

All required datasets and original clinical baseline bundles are tracked, so this command runs offline after dependencies are installed. To reacquire WISDM first (requires GitHub CLI access):

```bash
python -m ml_engine.src.complete_training --download
```

The runner caps CPU threads in subprocesses, writes failure status if a stage fails, and marks completion only after saved-model verification and tests succeed. It intentionally overwrites wearable/V2 experimental outputs, not the clinical baseline models. It is not a concurrent-run or transactional deployment system.

### Verify existing artifacts without retraining

```bash
python -m ml_engine.src.verify_saved_models
python -m pytest tests/ml_tests tests/wearable_tests -q
```

### Run example predictions

```bash
python -m ml_engine.src.wearable_pipeline predict --input ml_engine/wearable/example_features.json
python -m ml_engine.src.predict_experiment diabetes ml_engine/examples/diabetes.json
```

Use trusted PKL files only. The wearable example requires published motion-window features, NOT health-app daily summaries. V2 examples require the task's clinical features, units and contexts.

## Still outside this completed milestone

Native/API health-app integration, production backend/UI wiring, compatible independent clinical-cohort validation, clinical alert approval, prospective hypertension prediction, biological age and validated what-if simulations.
