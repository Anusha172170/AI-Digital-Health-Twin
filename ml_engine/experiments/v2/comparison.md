# V2 clinical model experiments

Baseline models are unchanged. Selection is based on training CV and validation only. The historical test set has been seen before; all comparisons remain exploratory.

| Task | V2 selection | ROC-AUC old → v2 | AP old → v2 | Precision old → v2 | Recall old → v2 | Brier old → v2 |
|---|---|---|---|---|---|---|
| diabetes | extra_trees + sigmoid | 0.788 → 0.791 | 0.676 → 0.662 | 0.506 → 0.537 | 0.830 → 0.811 | 0.174 → 0.175 |
| cvd | hist_gradient + sigmoid | 0.686 → 0.678 | 0.310 → 0.311 | 0.207 → 0.216 | 0.752 → 0.682 | 0.120 → 0.121 |
| hypertension | extra_trees + sigmoid | 0.740 → 0.732 | 0.541 → 0.529 | 0.454 → 0.441 | 0.673 → 0.726 | 0.183 → 0.186 |

AP = average precision; higher is better. Brier = probability squared error; lower is better. Threshold metrics are not directly comparable without their precision/recall tradeoff.
Each metrics.json includes 1,000 paired row-bootstrap 95% intervals for v2-minus-baseline changes. These do not include training/model-selection uncertainty or prove clinical benefit.
Preprocessing is refitted inside each training CV fold. No SMOTE before splitting, invented clinical features, new label proxies or test-driven candidate selection.
The selected pipeline is explained using validation permutation importance. Baseline XGBoost SHAP files do NOT explain these new ensembles/calibrated models.
Prospective hypertension, biological age and causal what-if outcomes remain unsupported. Wearable activity models are unchanged.
