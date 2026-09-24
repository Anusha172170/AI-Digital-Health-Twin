"""Evaluate saved models without fitting; raise on artifact/prediction inconsistencies."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from ml_engine.src.config import BASE, PROCESSED, TASKS
from ml_engine.src.evaluate_models import evaluate


def verify_hashes(root):
    checksums = json.loads((root / 'sha256.json').read_text())
    for name, expected in checksums.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Artifact checksum mismatch: {root / name}')


def verify():
    v2 = BASE / 'experiments/v2'
    wearable = BASE / 'wearable'
    verify_hashes(v2)
    verify_hashes(wearable / 'saved_models')
    verify_hashes(BASE / 'saved_models')
    results = {}
    for task, spec in TASKS.items():
        folder = v2 / task
        b = joblib.load(folder / 'model_bundle.pkl')
        test = pd.read_csv(PROCESSED / task / 'test.csv')
        p = b['pipeline'].predict_proba(test[spec['features']])[:, 1]
        stored = pd.read_csv(folder / 'test_predictions.csv')
        np.testing.assert_array_equal(test.source_row_id, stored.source_row_id)
        np.testing.assert_allclose(p, stored.candidate_probability, atol=1e-12)
        measured = evaluate(test[spec['target']], p, b['threshold'])
        expected = json.loads((folder / 'metrics.json').read_text())['candidate_v2']
        for name, value in measured.items():
            np.testing.assert_allclose(value, expected[name], atol=1e-10)
        results[task] = measured
    b = joblib.load(wearable / 'saved_models/wearable_activity_model.pkl')
    test = pd.read_csv(wearable / 'processed/test.csv')
    X = pd.DataFrame(b['preprocessor'].transform(test[b['features']]), columns=b['features'])
    p = b['model'].predict_proba(X)
    predictions = pd.read_csv(wearable / 'reports/test_predictions.csv')
    np.testing.assert_array_equal(test.window_id, predictions.window_id)
    np.testing.assert_allclose(p, predictions[[f'p_{c}' for c in b['encoder'].classes_]], atol=1e-5)
    labels = b['encoder'].inverse_transform(p.argmax(axis=1))
    np.testing.assert_array_equal(labels, predictions.predicted_activity)
    from ml_engine.src.wearable_pipeline import score
    measured = score(b['encoder'].transform(test.activity), p)
    report = json.loads((wearable / 'reports/metrics.json').read_text())
    for name, value in measured.items():
        np.testing.assert_allclose(value, report['test'][report['selected']][name], atol=1e-5)
    results['wearable'] = measured
    return results

if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
