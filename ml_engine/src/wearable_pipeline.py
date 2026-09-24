"""WISDM watch-accelerometer activity recognition, held out by participant.

Uses published window features, NOT raw sensor samples or clinical outcomes.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, log_loss
from xgboost import XGBClassifier
from ml_engine.src.config import BASE, SEED

ROOT = BASE / 'wearable'
RAW = ROOT / 'raw'
DATA = ROOT / 'processed'
MODELS = ROOT / 'saved_models'
REPORTS = ROOT / 'reports'
REPO = 'kseniacold/aai_530_dataset_static'
COMMIT = 'e70750583775ab997853a9aca57c4a45d934346c'
PREFIX = 'wisdm-dataset'
ARFF_DIR = f'{PREFIX}/arff_files/watch/accel'


def api(path):
    r = subprocess.run(['gh', 'api', f'repos/{REPO}/contents/{path}?ref={COMMIT}'],
                       capture_output=True, text=True, check=True, timeout=90)
    return json.loads(r.stdout)


def acquire():
    RAW.mkdir(parents=True, exist_ok=True)
    entries = [e for e in api(ARFF_DIR) if e['name'].endswith('.arff')]
    paths = [e['path'] for e in entries] + [f'{PREFIX}/activity_key.txt', f'{PREFIX}/README.txt']
    def fetch(path):
        meta = api(path)
        content = base64.b64decode(meta['content'])
        target = RAW / Path(path).name
        target.write_bytes(content)
        return {'file': target.name, 'source_path': path, 'bytes': len(content),
                'sha256': hashlib.sha256(content).hexdigest(), 'github_blob_sha': meta['sha']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, paths))
    manifest = {'dataset': 'WISDM Smartphone and Smartwatch Activity and Biometrics',
        'doi': '10.24432/C5HK59', 'license': 'CC BY 4.0 (UCI dataset page)',
        'author': 'Gary Weiss (2019)', 'upstream': 'https://archive.ics.uci.edu/dataset/507/',
        'mirror': REPO, 'commit': COMMIT, 'downloaded_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'All 50 watch accelerometer ARFF files in this pinned mirror; subject 1614 absent. Not the full 51-participant/four-stream raw release.',
        'files': files}
    (RAW / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Downloaded', len(entries), 'source ARFF files')


def prepare():
    manifest = json.loads((RAW / 'manifest.json').read_text())
    for entry in manifest['files']:
        if hashlib.sha256((RAW / entry['file']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"Source checksum mismatch: {entry['file']}")
    frames = []
    for path in sorted(RAW.glob('*.arff')):
        array, _ = arff.loadarff(io.StringIO(path.read_text()))
        df = pd.DataFrame(array)
        df.columns = df.columns.str.strip(chr(34))
        subject = int(path.name.split('_')[1])
        if {int(v.decode()) for v in df['class']} != {subject}:
            raise ValueError(f'Participant ID mismatch in {path.name}')
        df['activity'] = df.pop('ACTIVITY').str.decode('utf-8')
        df = df.drop(columns='class')  # Identity is NOT an input feature.
        df['subject_id'] = subject
        df['window_id'] = [f'{subject}:{i}' for i in range(len(df))]
        frames.append(df)
    if len(frames) != 50:
        raise ValueError('Expected all 50 mirror subjects; run download first')
    df = pd.concat(frames, ignore_index=True)
    features = [c for c in df if c not in ['activity', 'subject_id', 'window_id']]
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / 'wearable_dataset.csv', index=False)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.4, random_state=SEED)
    train, rest = next(splitter.split(df, groups=df.subject_id))
    v, t = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=SEED).split(df.iloc[rest], groups=df.iloc[rest].subject_id))
    splits = {'train': train, 'validation': rest[v], 'test': rest[t]}
    pre = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    pre.fit(df.iloc[train][features])
    meta = {'features': features, 'seed': SEED, 'rows': len(df), 'subjects': 50,
            'target': 'activity', 'split_policy': '60/20/20 subjects, no participant overlap', 'splits': {}}
    for name, ids in splits.items():
        frame = df.iloc[ids].copy()
        if set(frame.activity) != set(df.activity):
            raise ValueError('Split lacks a target activity')
        frame.to_csv(DATA / f'{name}.csv', index=False)
        scaled = pd.DataFrame(pre.transform(frame[features]), columns=features)
        for c in ['activity', 'subject_id', 'window_id']:
            scaled[c] = frame[c].to_numpy()
        scaled.to_csv(DATA / f'{name}_scaled.csv', index=False)
        meta['splits'][name] = {'rows': len(frame), 'subjects': sorted(frame.subject_id.unique().tolist())}
    joblib.dump(pre, MODELS / 'activity_preprocessor.pkl')
    joblib.dump(pre.named_steps['scaler'], MODELS / 'activity_scaler.pkl')
    (DATA / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('Prepared', len(df), 'windows;', len(features), 'features;', meta['splits'])


def score(y, p):
    pred = p.argmax(axis=1)
    return {'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, average='macro'),
            'weighted_f1': f1_score(y, pred, average='weighted'), 'log_loss': log_loss(y, p)}


def train():
    meta = json.loads((DATA / 'metadata.json').read_text())
    features = meta['features']
    parts = {s: pd.read_csv(DATA / f'{s}.csv') for s in ['train', 'validation', 'test']}
    preprocessor = joblib.load(MODELS / 'activity_preprocessor.pkl')
    encoder = LabelEncoder().fit(parts['train'].activity)
    # Use exactly the same raw-input path as predict(), not rounded CSV features.
    X = {s: pd.DataFrame(preprocessor.transform(f[features]), columns=features) for s, f in parts.items()}
    y = {s: encoder.transform(f.activity) for s, f in parts.items()}
    models = {'dummy': DummyClassifier(strategy='prior'),
        'logistic': LogisticRegression(max_iter=3000, C=1, random_state=SEED),
        'xgboost': XGBClassifier(n_estimators=140, max_depth=4, learning_rate=.06,
            subsample=.85, colsample_bytree=.85, reg_lambda=5, n_jobs=2,
            random_state=SEED, tree_method='hist', eval_metric='mlogloss')}
    validation = {}
    for name, model in models.items():
        model.fit(X['train'], y['train'])
        validation[name] = score(y['validation'], model.predict_proba(X['validation']))
        joblib.dump(model, MODELS / f'activity_{name}.pkl', compress=3)
        print(name, validation[name], flush=True)
    selected = max(validation, key=lambda k: validation[k]['macro_f1'])
    bundle = {'model': models[selected], 'preprocessor': joblib.load(MODELS / 'activity_preprocessor.pkl'),
        'features': features, 'encoder': encoder, 'selected_algorithm': selected,
        'input_contract': 'Published WISDM watch accelerometer window features in source units; NOT raw XYZ or HealthKit summaries',
        'warning': 'Research activity classifier; no disease, sleep, biological-age or clinical risk claims.'}
    joblib.dump(bundle, MODELS / 'wearable_activity_model.pkl', compress=3)
    models['xgboost'].save_model(MODELS / 'activity_xgboost.json')
    REPORTS.mkdir(parents=True, exist_ok=True)
    results = {name: score(y['test'], m.predict_proba(X['test'])) for name, m in models.items()}
    probs = models[selected].predict_proba(X['test'])
    pred = probs.argmax(axis=1)
    report = {'selected': selected, 'selection': 'Validation macro F1; fixed hyperparameters, no test tuning',
        'validation': validation, 'test': results, 'classes': encoder.classes_.tolist(),
        'classification_report': classification_report(y['test'], pred, target_names=encoder.classes_, output_dict=True, zero_division=0)}
    (REPORTS / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    output = parts['test'][['subject_id', 'window_id', 'activity']].copy()
    output['predicted_activity'] = encoder.inverse_transform(pred)
    for i, label in enumerate(encoder.classes_):
        output[f'p_{label}'] = probs[:, i]
    output.to_csv(REPORTS / 'test_predictions.csv', index=False)
    per_subject = []
    for sid, group in output.groupby('subject_id'):
        per_subject.append({'subject_id': int(sid), 'windows': len(group),
            'accuracy': accuracy_score(group.activity, group.predicted_activity),
            'macro_f1': f1_score(group.activity, group.predicted_activity, average='macro')})
    pd.DataFrame(per_subject).to_csv(REPORTS / 'per_subject_metrics.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(confusion_matrix(y['test'], pred), display_labels=encoder.classes_).plot(ax=ax, colorbar=False, values_format='d')
    ax.set_title(f'WISDM watch activity: {selected}; unseen participants')
    fig.tight_layout()
    fig.savefig(REPORTS / 'confusion_matrix.png', dpi=140)
    plt.close(fig)
    import shap
    sample = X['test'].head(20)
    explanation = shap.TreeExplainer(models['xgboost'])(sample)
    margins = models['xgboost'].predict(sample, output_margin=True)
    np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1), margins, atol=1e-4)
    rows = []
    for i in range(len(sample)):
        for k, label in enumerate(encoder.classes_):
            for j, feature in enumerate(features):
                rows.append({'window_id': parts['test'].window_id.iloc[i], 'class': label, 'feature': feature,
                             'contribution_raw_margin': float(explanation.values[i, j, k]),
                             'base_raw_margin': float(explanation.base_values[i, k])})
    pd.DataFrame(rows).to_csv(REPORTS / 'xgboost_shap.csv', index=False)
    pd.Series(np.abs(explanation.values).mean(axis=(0, 2)), index=features).nlargest(15).sort_values().plot.barh(figsize=(9, 6))
    plt.title('XGBoost mean absolute SHAP: 20 test windows, class margins')
    plt.tight_layout()
    plt.savefig(REPORTS / 'xgboost_shap.png', dpi=140)
    plt.close()
    lines = ['# Wearable activity benchmark', '', 'Selected on validation macro F1: **' + selected + '**.', '',
        '| Candidate | Test accuracy | Test macro F1 | Test weighted F1 |', '|---|---:|---:|---:|']
    for name, m in results.items():
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} |")
    lines += ['', 'Test participants are absent from training and validation. Full metrics, per-class and per-participant results are included.',
        'No clinical targets, raw-to-feature deployment adapter, probability calibration or external device validation. SHAP describes the XGBoost candidate in raw class-margin units, not probability contributions.']
    (REPORTS / 'model_metrics.md').write_text('\n'.join(lines) + '\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in MODELS.iterdir() if p.name != 'sha256.json'}
    (MODELS / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    raw_example = pd.read_csv(DATA / 'test.csv').iloc[0][features]
    (ROOT / 'example_features.json').write_text(json.dumps(
        {k: None if pd.isna(v) else float(v) for k, v in raw_example.items()}, indent=2) + '\n')
    print('Selected:', selected, 'Test:', results[selected])


def predict(values):
    b = joblib.load(MODELS / 'wearable_activity_model.pkl')
    if set(values) != set(b['features']):
        raise ValueError('Supply exactly the published window features, not subject ID or raw sensor samples')
    frame = pd.DataFrame([values])[b['features']].apply(pd.to_numeric, errors='raise')
    if np.isinf(frame.to_numpy()).any() or frame.isna().all(axis=1).any():
        raise ValueError('Invalid feature values')
    transformed = pd.DataFrame(b['preprocessor'].transform(frame), columns=b['features'])
    p = b['model'].predict_proba(transformed)[0]
    return {'activity_code': b['encoder'].classes_[p.argmax()],
        'probabilities': dict(zip(b['encoder'].classes_.tolist(), p.tolist())), 'warning': b['warning']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['download', 'preprocess', 'train', 'predict'])
    parser.add_argument('--input')
    args = parser.parse_args()
    if args.stage == 'predict':
        if not args.input:
            parser.error('--input is required')
        print(json.dumps(predict(json.loads(Path(args.input).read_text())), indent=2))
    else:
        {'download': acquire, 'preprocess': prepare, 'train': train}[args.stage]()
