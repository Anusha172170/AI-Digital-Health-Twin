# Model evaluation

Research benchmarks only; no clinical validation or guaranteed accuracy.

60% train / 20% validation / 20% test, stratified, seed 42. Imputer and scaler fit on training only. Model selection uses validation average precision; thresholds maximize validation F1, not a clinical safety objective. Test set is evaluated only after selection. Models are not probability-calibrated; Brier and reliability plots are diagnostic.

| Task | Selected | ROC-AUC | AP | Recall | Precision | F1 | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| diabetes | logistic | 0.788 | 0.676 | 0.830 | 0.506 | 0.629 | 0.174 |
| cvd | xgboost | 0.686 | 0.310 | 0.752 | 0.207 | 0.325 | 0.120 |
| hypertension | logistic | 0.740 | 0.541 | 0.673 | 0.454 | 0.542 | 0.183 |

## Interpretation and limitations

- Diabetes means the PIMA recorded status label, not future disease development. Population-specific benchmark; not representative of all users.
- CVD artifact predicts the TenYearCHD label, specifically coronary heart disease.
- Hypertension is exploratory prevalent-status classification. No prospective hypertension dataset was acquired. BP and medication proxies are excluded.
- Separate datasets and feature schemas; no fabricated merging, sleep/steps, biological ages or sensor values.
- SHAP files describe the XGBoost candidate, in log-odds units, even if a baseline is selected. They do not establish causal effects.
- A low score is not an all-clear. No emergency triage, wearable validation, demographic external validation or causal lifestyle simulator is established.
- IDs are source row numbers, not known longitudinal subject IDs. External subject-level validation remains necessary.
- Full baseline comparisons, confusion matrices, prediction CSVs and calibration plots are in each task subdirectory.
