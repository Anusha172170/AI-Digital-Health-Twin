import json
import joblib
import numpy as np
import pandas as pd
import pytest
from ml_engine.src.config import BASE, PROCESSED, TASKS
from ml_engine.src.verify_saved_models import verify


def test_all_saved_models_reproduce_reported_results():
    result = verify()
    assert set(result) == {*TASKS, 'wearable'}


@pytest.mark.parametrize('task', TASKS)
def test_all_tuned_candidates_load_and_predict(task):
    folder = BASE / 'experiments/v2' / task
    selection = json.loads((folder / 'selection.json').read_text())
    frame = pd.read_csv(PROCESSED / task / 'validation.csv')
    for family in ['logistic','random_forest','extra_trees','hist_gradient','xgboost']:
        bundle = joblib.load(folder / f'{family}_candidate.pkl')
        p = bundle['pipeline'].predict_proba(frame[bundle['features']])[:, 1]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        from sklearn.metrics import average_precision_score
        assert average_precision_score(frame[TASKS[task]['target']],p) == pytest.approx(
            selection['validation'][family]['average_precision'])
