import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from ml_engine.src.config import RAW, PROCESSED, MODELS, TASKS
from ml_engine.src.preprocess import clean_features

@pytest.mark.parametrize('task', TASKS)
def test_splits_and_train_only_preprocessing(task):
    spec = TASKS[task]
    parts = {s: pd.read_csv(PROCESSED / task / f'{s}.csv') for s in ['train', 'validation', 'test']}
    sets = [set(df.source_row_id) for df in parts.values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    source = pd.read_csv(RAW / spec['file'])
    pre = joblib.load(MODELS / f'{task}_preprocessor.pkl')
    train = parts['train'][spec['features']]
    np.testing.assert_allclose(pre.named_steps['imputer'].statistics_, train.median().to_numpy())
    assert pre.named_steps['scaler'].n_samples_seen_ == len(train)
    for split, df in parts.items():
        np.testing.assert_array_equal(df[spec['target']], source.loc[df.source_row_id, spec['target']])
        expected = clean_features(source.loc[df.source_row_id], task).to_numpy(dtype=float)
        np.testing.assert_allclose(df[spec['features']], expected, equal_nan=True)
        scaled = pd.read_csv(PROCESSED / task / f'{split}_scaled.csv')
        np.testing.assert_allclose(scaled[spec['features']], pre.transform(df[spec['features']]), atol=1e-12)
        assert np.isfinite(scaled[spec['features']].to_numpy()).all()

def test_original_source_hashes():
    for record in json.loads((RAW / 'manifest.json').read_text()).values():
        assert hashlib.sha256((RAW / record['source_file']).read_bytes()).hexdigest() == record['sha256']

def test_hypertension_has_no_direct_label_proxies():
    assert not {'prevalentHyp', 'sysBP', 'diaBP', 'BPMeds', 'TenYearCHD'} & set(TASKS['hypertension']['features'])
