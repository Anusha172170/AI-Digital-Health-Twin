import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from ml_engine.src.config import TASKS, MODELS, PROCESSED, REPORTS
from ml_engine.src.predict import predict
from ml_engine.src.explainability import explain
from ml_engine.src.evaluate_models import evaluate

@pytest.mark.parametrize('task', TASKS)
def test_saved_prediction_and_metrics(task):
    spec = TASKS[task]
    frame = pd.read_csv(PROCESSED / task / 'test.csv')
    result = predict(task, frame[spec['features']].iloc[0].to_dict())
    saved = pd.read_csv(REPORTS / task / 'test_predictions.csv')
    assert result['probability'] == pytest.approx(saved.probability.iloc[0], abs=1e-7)
    assert 0 <= result['probability'] <= 1
    assert result['warning']
    report = json.loads((REPORTS / task / 'metrics.json').read_text())
    metrics = report['test'][report['selected_algorithm']]
    actual = evaluate(saved.actual, saved.probability, metrics['threshold'])
    assert actual['roc_auc'] == pytest.approx(metrics['roc_auc'])
    assert actual['confusion_matrix'] == metrics['confusion_matrix']
    with pytest.raises(ValueError):
        predict(task, {'age': 40})
    with pytest.raises(ValueError):
        predict(task, {f: None for f in spec['features']})

@pytest.mark.parametrize('task', TASKS)
def test_shap_additivity(task):
    model = joblib.load(MODELS / f'{task}_xgboost.pkl')
    X = pd.read_csv(PROCESSED / task / 'test_scaled.csv')[TASKS[task]['features']].head(3)
    result = explain(model, X)
    assert np.isfinite(result.values).all()

def test_model_checksums():
    for name, checksum in json.loads((MODELS / 'sha256.json').read_text()).items():
        assert hashlib.sha256((MODELS / name).read_bytes()).hexdigest() == checksum
