"""Independent patient-preserving 60/20/20 splits; training-only imputation/scaling."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml_engine.src.config import RAW, PROCESSED, MODELS, TASKS, SEED


def clean_features(frame, task):
    spec = TASKS[task]
    result = frame[spec['features']].apply(pd.to_numeric, errors='raise').copy()
    result[spec['zero_missing']] = result[spec['zero_missing']].replace(0, np.nan)
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature value')
    return result


def prepare(task):
    spec = TASKS[task]
    df = pd.read_csv(RAW / spec['file'])
    # Remove identical rows before splitting to prevent exact duplicate leakage.
    original_count = len(df)
    df = df.drop_duplicates().dropna(subset=[spec['target']])
    y = df[spec['target']].astype(int)
    if not set(y.unique()) == {0, 1}:
        raise ValueError('Expected binary labels')
    X = clean_features(df, task)
    train, other = train_test_split(df.index, test_size=.4, stratify=y, random_state=SEED)
    val, test = train_test_split(other, test_size=.5, stratify=y.loc[other], random_state=SEED)
    pre = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    pre.fit(X.loc[train])
    dest = PROCESSED / task
    dest.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    for split, indices in [('train', train), ('validation', val), ('test', test)]:
        raw = X.loc[indices].copy()
        raw.insert(0, 'source_row_id', indices)
        raw[spec['target']] = y.loc[indices]
        raw.to_csv(dest / f'{split}.csv', index=False)
        scaled = pd.DataFrame(pre.transform(X.loc[indices]), columns=spec['features'])
        scaled.insert(0, 'source_row_id', indices)
        scaled[spec['target']] = y.loc[indices].to_numpy()
        scaled.to_csv(dest / f'{split}_scaled.csv', index=False)
    joblib.dump(pre, MODELS / f'{task}_preprocessor.pkl')
    joblib.dump(pre.named_steps['scaler'], MODELS / f'{task}_scaler.pkl')
    metadata = {**spec, 'seed': SEED, 'original_rows': original_count,
                'retained_rows': len(df), 'split_sizes': {'train': len(train), 'validation': len(val), 'test': len(test)}}
    (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(task, metadata['split_sizes'])

if __name__ == '__main__':
    for task in TASKS:
        prepare(task)
