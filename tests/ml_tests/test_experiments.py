import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from ml_engine.src.config import TASKS, PROCESSED
from ml_engine.src.improve_models import OUT, threshold_for
from ml_engine.src.predict_experiment import predict
from ml_engine.src.evaluate_models import evaluate

@pytest.mark.parametrize('task',TASKS)
def test_v2_artifact_and_selection(task):
    dest = OUT / task
    spec = TASKS[task]
    bundle = joblib.load(dest/'model_bundle.pkl')
    decision = json.loads((dest/'selection.json').read_text())
    assert decision['selected_family'] == max(decision['validation'],key=lambda n: decision['validation'][n]['average_precision'])
    raw_metrics = decision['validation'][decision['selected_family']]
    calibrated = decision['calibrated_validation']
    assert decision['sigmoid_calibration_selected'] == (calibrated['brier_score'] < raw_metrics['brier_score'] and calibrated['average_precision'] >= raw_metrics['average_precision']-.01)
    va = pd.read_csv(PROCESSED/task/'validation.csv')
    vp = bundle['pipeline'].predict_proba(va[spec['features']])[:,1]
    assert bundle['threshold'] == pytest.approx(threshold_for(va[spec['target']],vp))
    test = pd.read_csv(PROCESSED/task/'test.csv')
    p = bundle['pipeline'].predict_proba(test[spec['features']])[:,1]
    saved = pd.read_csv(dest/'test_predictions.csv')
    np.testing.assert_allclose(saved.candidate_probability,p)
    metrics = evaluate(test[spec['target']],p,bundle['threshold'])
    stored = json.loads((dest/'metrics.json').read_text())['candidate_v2']
    assert metrics['confusion_matrix'] == stored['confusion_matrix']
    assert metrics['average_precision'] == pytest.approx(stored['average_precision'])
    result = predict(task,test[spec['features']].iloc[0].to_dict())
    assert result['probability'] == pytest.approx(p[0])
    with pytest.raises(ValueError):
        predict(task,{'age':30})
    with pytest.raises(ValueError):
        predict(task,{f:None for f in spec['features']})


def test_experiment_checksums():
    for path, sha in json.loads((OUT/'sha256.json').read_text()).items():
        assert hashlib.sha256((OUT/path).read_bytes()).hexdigest() == sha
