# Wearable activity benchmark

Selected on validation macro F1: **xgboost**.

| Candidate | Test accuracy | Test macro F1 | Test weighted F1 |
|---|---:|---:|---:|
| dummy | 0.055 | 0.006 | 0.006 |
| logistic | 0.614 | 0.607 | 0.603 |
| xgboost | 0.657 | 0.655 | 0.652 |

Test participants are absent from training and validation. Full metrics, per-class and per-participant results are included.
No clinical targets, raw-to-feature deployment adapter, probability calibration or external device validation. SHAP describes the XGBoost candidate in raw class-margin units, not probability contributions.
