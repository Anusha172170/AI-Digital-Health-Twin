"""Inference using task-specific raw features and the persisted preprocessing bundle."""
import argparse
import json
import joblib
import pandas as pd
from ml_engine.src.config import MODELS, TASKS
from ml_engine.src.preprocess import clean_features


def predict(task, values):
    bundle = joblib.load(MODELS / f'{task}_model.pkl')  # Load only trusted local artifacts.
    missing = set(bundle['features']) - values.keys()
    extra = values.keys() - set(bundle['features'])
    if missing or extra:
        raise ValueError(f'Feature mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    frame = clean_features(pd.DataFrame([values]), task)
    if frame.isna().all(axis=1).any():
        raise ValueError('All features missing')
    X = bundle['preprocessor'].transform(frame)
    X = pd.DataFrame(X, columns=bundle['features'])
    p = float(bundle['model'].predict_proba(X)[0, 1])
    return {'task': task, 'probability': p, 'positive': p >= bundle['threshold'],
            'threshold': bundle['threshold'], 'meaning': bundle['meaning'],
            'imputed_features': frame.columns[frame.isna().iloc[0]].tolist(), 'warning': bundle['warning']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=TASKS)
    parser.add_argument('json_file')
    args = parser.parse_args()
    print(json.dumps(predict(args.task, json.loads(open(args.json_file).read())), indent=2))
