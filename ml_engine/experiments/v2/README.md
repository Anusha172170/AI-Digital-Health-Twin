# V2 model improvement experiment

## Outcome

**No consistent generalization improvement established. Baseline artifacts and default inference remain unchanged.** See [comparison](comparison.md). All three candidates modestly increased F1 at their chosen thresholds, but ranking, recall/precision tradeoffs and probability errors were mixed. Brier error worsened slightly for all three on the historical test set despite improving on validation. Paired bootstrap intervals for changes in ROC-AUC, AP and Brier all cross zero.

## Methods

- Same original train/validation/test rows and target definitions; no new data or synthetic clinical labels.
- Five algorithm families: regularized logistic regression, Random Forest, Extra Trees, histogram gradient boosting and XGBoost.
- Eight hyperparameter configurations per family, three stratified training CV folds: 72 CV fits per family across three tasks, 360 CV fits total, plus refits/calibration.
- Imputation, missing-value indicators and scaling refit inside each fold. Class weighting explored where supported; no oversampling outside folds.
- CV optimizes average precision (AP); one best configuration from each family compared on validation with the original selected baseline.
- Family selected by validation AP. Train-only cross-validated sigmoid calibration accepted only if validation Brier decreases and AP drops no more than 0.01.
- Threshold chosen by maximum validation F1, not clinical safety criteria. Final model remains fit only on training data; validation is not included in refitting.
- Historical test accessed only after selection decisions saved. Because it was already inspected in earlier work, it is NOT a fresh confirmatory benchmark.
- 1,000 paired row-bootstrap samples estimate test metric-change intervals. Not clinical uncertainty intervals, not repeated training, and not an external validation study.

## Reproduce

From repository root with the existing pinned environment:

```bash
# Thread caps keep CPU experiments predictable on small machines.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m ml_engine.src.improve_models
python -m ml_engine.src.predict_experiment diabetes ml_engine/examples/diabetes.json
python -m pytest tests/ml_tests tests/wearable_tests -q
```

On Windows, set the environment variables separately or omit the caps. Running the experiment overwrites only v2 experiment outputs, not baseline models.

Each task folder contains CV search tables, selected parameters, validation diagnostics, `model_bundle.pkl`, test probability CSV, paired uncertainty estimates, PR/calibration plots and validation permutation importance. The best training-CV pipeline from each of the five families is saved as `{family}_candidate.pkl`, alongside the final selected/calibrated `model_bundle.pkl`. All candidate artifacts have checksums; remaining searched configurations can be reproduced from the search tables. Dependencies are pinned in `ml_engine/requirements.txt`. Use trusted PKL files only. Opt-in inference accepts raw task-specific values, performs the original zero-to-missing cleaning, then uses its own stored preprocessing exactly once.

Permutation importance explains global validation performance sensitivity of the chosen pipeline. It is not per-user SHAP, not causal evidence and may be unstable for correlated features. Existing baseline SHAP artifacts must not be used to explain these new predictions.

## Next useful improvements

1. Acquire independently sourced, permissioned data with labels and features matching health-app inputs and intended population.
2. Collect clinical measurements rather than substituting steps/HR for glucose, cholesterol or blood pressure.
3. Use repeated/nested CV for more stable development estimates, then a genuinely untouched external cohort.
4. Specify time horizon and clinical utility before selecting alert thresholds; evaluate precision/recall, calibration and subgroup performance.
5. For prospective hypertension and lifestyle what-if effects, acquire appropriate longitudinal data; algorithm tuning cannot create absent outcomes or establish causal effects.

The wearable activity model and app integration are not modified by these experiments.
