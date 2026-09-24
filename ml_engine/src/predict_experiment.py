"""Opt-in inference for v2 candidates; does not change production/baseline routing."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from ml_engine.src.config import TASKS
from ml_engine.src.improve_models import OUT
from ml_engine.src.preprocess import clean_features


def predict(task, values):
    if task not in TASKS:
        raise ValueError('Unknown task')
    bundle = joblib.load(OUT / task / 'model_bundle.pkl')
    if set(values) != set(bundle['features']):
        raise ValueError('Supply exact task-specific raw features; use null for missing values')
    frame = clean_features(pd.DataFrame([values]),task)
    if frame.isna().all(axis=1).any():
        raise ValueError('All features missing')
    p = float(bundle['pipeline'].predict_proba(frame)[0,1])
    return {'task':task,'version':'experimental_v2','probability':p,
        'positive':p>=bundle['threshold'],'threshold':bundle['threshold'],
        'meaning':bundle['meaning'],'warning':bundle['warning']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task',choices=TASKS)
    parser.add_argument('input')
    args = parser.parse_args()
    print(json.dumps(predict(args.task,json.loads(Path(args.input).read_text())),indent=2))
