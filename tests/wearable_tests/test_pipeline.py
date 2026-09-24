import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from ml_engine.src.wearable_pipeline import RAW, DATA, MODELS, REPORTS, ROOT, predict


def test_source_checksums_and_coverage():
    manifest = json.loads((RAW / 'manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((RAW / item['file']).read_bytes()).hexdigest() == item['sha256']
    assert len(list(RAW.glob('*.arff'))) == 50


def test_participant_isolation_and_scaling():
    meta = json.loads((DATA / 'metadata.json').read_text())
    features = meta['features']
    assert len(features) == 91
    assert not {'class', 'ACTIVITY', 'activity', 'subject_id', 'window_id'} & set(features)
    subjects, windows = set(), set()
    pre = joblib.load(MODELS / 'activity_preprocessor.pkl')
    for split in ['train', 'validation', 'test']:
        raw = pd.read_csv(DATA / f'{split}.csv')
        scaled = pd.read_csv(DATA / f'{split}_scaled.csv')
        assert not subjects & set(raw.subject_id)
        assert not windows & set(raw.window_id)
        subjects.update(raw.subject_id)
        windows.update(raw.window_id)
        assert raw.window_id.tolist() == scaled.window_id.tolist()
        assert raw.activity.nunique() == 18
        np.testing.assert_allclose(scaled[features], pre.transform(raw[features]), rtol=1e-4, atol=1e-5)
        assert np.isfinite(scaled[features].to_numpy()).all()
        if split == 'train':
            assert pre.named_steps['scaler'].n_samples_seen_ == len(raw)
            np.testing.assert_allclose(pre.named_steps['imputer'].statistics_, raw[features].median(), rtol=1e-6)
    assert len(subjects) == 50
    assert len(windows) == meta['rows'] == 18211


def test_inference_and_saved_metrics():
    value = json.loads((ROOT / 'example_features.json').read_text())
    response = predict(value)
    assert len(response['probabilities']) == 18
    assert sum(response['probabilities'].values()) == pytest.approx(1, abs=1e-6)
    result = pd.read_csv(REPORTS / 'test_predictions.csv')
    assert response['activity_code'] == result.predicted_activity.iloc[0]
    metrics = json.loads((REPORTS / 'metrics.json').read_text())
    from sklearn.metrics import f1_score
    assert f1_score(result.activity, result.predicted_activity, average='macro') == pytest.approx(metrics['test'][metrics['selected']]['macro_f1'])
    with pytest.raises(ValueError):
        predict({'heart_rate': 72})
    with pytest.raises(ValueError):
        predict({**value, 'subject_id': 1600})
    for name, sha in json.loads((MODELS / 'sha256.json').read_text()).items():
        assert hashlib.sha256((MODELS / name).read_bytes()).hexdigest() == sha
