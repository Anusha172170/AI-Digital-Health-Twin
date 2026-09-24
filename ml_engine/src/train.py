"""Train baselines and XGBoost, select model/threshold ONLY on validation data."""
import hashlib
import importlib.metadata
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, f1_score
from sklearn.calibration import CalibrationDisplay
from xgboost import XGBClassifier
from ml_engine.src.config import MODELS, PROCESSED, REPORTS, TASKS, SEED
from ml_engine.src.evaluate_models import evaluate
from ml_engine.src.explainability import explain


def train(task):
    spec = TASKS[task]
    dest = PROCESSED / task
    parts = {s: pd.read_csv(dest / f'{s}_scaled.csv') for s in ['train', 'validation', 'test']}
    X = {s: df[spec['features']] for s, df in parts.items()}
    y = {s: df[spec['target']] for s, df in parts.items()}
    candidates = {'dummy': DummyClassifier(strategy='prior'),
                  'logistic': LogisticRegression(max_iter=2000, random_state=SEED),
                  'xgboost': XGBClassifier(n_estimators=150, max_depth=3, learning_rate=.04,
                       subsample=.9, colsample_bytree=.9, reg_lambda=5, eval_metric='logloss',
                       random_state=SEED, n_jobs=2, tree_method='hist')}
    validations = {}
    for name, model in candidates.items():
        model.fit(X['train'], y['train'])
        p = model.predict_proba(X['validation'])[:, 1]
        threshold = max(np.arange(.05, .951, .01), key=lambda t: f1_score(y['validation'], p >= t))
        validations[name] = evaluate(y['validation'], p, threshold)
        joblib.dump(model, MODELS / f'{task}_{name}.pkl')
    # Select by validation average precision, never by test performance.
    selected = max(validations, key=lambda name: validations[name]['average_precision'])
    threshold = validations[selected]['threshold']
    model = candidates[selected]
    bundle = {'model': model, 'preprocessor': joblib.load(MODELS / f'{task}_preprocessor.pkl'),
              'features': spec['features'], 'task': task, 'target': spec['target'],
              'meaning': spec['meaning'], 'threshold': threshold, 'selected_algorithm': selected,
              'warning': 'Research benchmark only. Uncalibrated, not a clinical diagnosis or validated wearable risk score.'}
    joblib.dump(bundle, MODELS / f'{task}_model.pkl', compress=3)
    candidates['xgboost'].save_model(MODELS / f'{task}_xgboost.json')
    metrics = {name: evaluate(y['test'], m.predict_proba(X['test'])[:, 1], validations[name]['threshold'])
               for name, m in candidates.items()}
    folder = REPORTS / task
    folder.mkdir(parents=True, exist_ok=True)
    p = model.predict_proba(X['test'])[:, 1]
    pd.DataFrame({'source_row_id': parts['test']['source_row_id'], 'actual': y['test'],
                  'probability': p, 'predicted': (p >= threshold).astype(int)}).to_csv(folder / 'test_predictions.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(y['test'], p >= threshold, ax=axes[0, 0], colorbar=False)
    RocCurveDisplay.from_predictions(y['test'], p, ax=axes[0, 1])
    PrecisionRecallDisplay.from_predictions(y['test'], p, ax=axes[1, 0])
    CalibrationDisplay.from_predictions(y['test'], p, n_bins=8, ax=axes[1, 1])
    fig.suptitle(f'{task}: {selected} — held-out test set')
    fig.tight_layout()
    fig.savefig(folder / 'evaluation.png', dpi=140)
    plt.close(fig)
    # Explain the XGBoost candidate even when validation selects a different model.
    sample = X['test'].head(50)
    ex = explain(candidates['xgboost'], sample)
    sv = pd.DataFrame(ex.values, columns=spec['features'])
    sv.insert(0, 'source_row_id', parts['test']['source_row_id'].head(50).to_numpy())
    sv['base_log_odds'] = ex.base_values
    sv.to_csv(folder / 'xgboost_shap_log_odds.csv', index=False)
    importance = pd.Series(np.abs(ex.values).mean(axis=0), index=spec['features']).sort_values()
    ax = importance.plot.barh(figsize=(8, 5), title='XGBoost mean |SHAP| — first 50 test rows (log-odds)')
    ax.figure.tight_layout()
    ax.figure.savefig(folder / 'xgboost_shap.png', dpi=140)
    plt.close(ax.figure)
    report = {'task': task, 'meaning': spec['meaning'], 'selected_algorithm': selected,
              'selection': 'Highest validation average precision; validation-max-F1 threshold',
              'validation': validations, 'test': metrics,
              'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'shap', 'joblib']}}
    (folder / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(task, selected, metrics[selected])
    return report


def run_all():
    reports = [train(task) for task in TASKS]
    lines = ['# Model evaluation\n', 'Research benchmarks only; no clinical validation or guaranteed accuracy.\n',
             '60% train / 20% validation / 20% test, stratified, seed 42. Imputer and scaler fit on training only. '
             'Model selection uses validation average precision; thresholds maximize validation F1, not a clinical safety objective. '
             'Test set is evaluated only after selection. Models are not probability-calibrated; Brier and reliability plots are diagnostic.\n',
             '| Task | Selected | ROC-AUC | AP | Recall | Precision | F1 | Brier |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        m = r['test'][r['selected_algorithm']]
        lines.append(f"| {r['task']} | {r['selected_algorithm']} | " + ' | '.join(f'{m[k]:.3f}' for k in ['roc_auc', 'average_precision', 'recall', 'precision', 'f1', 'brier_score']) + ' |')
    lines += ['\n## Interpretation and limitations\n',
        '- Diabetes means the PIMA recorded status label, not future disease development. Population-specific benchmark; not representative of all users.',
        '- CVD artifact predicts the TenYearCHD label, specifically coronary heart disease.',
        '- Hypertension is exploratory prevalent-status classification. No prospective hypertension dataset was acquired. BP and medication proxies are excluded.',
        '- Separate datasets and feature schemas; no fabricated merging, sleep/steps, biological ages or sensor values.',
        '- SHAP files describe the XGBoost candidate, in log-odds units, even if a baseline is selected. They do not establish causal effects.',
        '- A low score is not an all-clear. No emergency triage, wearable validation, demographic external validation or causal lifestyle simulator is established.',
        '- IDs are source row numbers, not known longitudinal subject IDs. External subject-level validation remains necessary.',
        '- Full baseline comparisons, confusion matrices, prediction CSVs and calibration plots are in each task subdirectory.']
    (REPORTS / 'model_metrics.md').write_text('\n'.join(lines) + '\n')
    hashes = {str(p.relative_to(MODELS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in MODELS.iterdir() if p.is_file() and p.name != 'sha256.json'}
    (MODELS / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')

if __name__ == '__main__':
    run_all()
