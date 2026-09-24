"""Frozen baseline evaluation on a reviewed external CSV. No fitting or tuning.

Automated checks cannot establish source independence or target equivalence;
those must be reviewed before a manifest is signed off.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ml_engine.src.config import MODELS, PROCESSED, TASKS, BASE
from ml_engine.src.preprocess import clean_features
from ml_engine.src.evaluate_models import evaluate


def fingerprint(frame):
    # Canonical hash detects exact/rounded feature copies, not all near duplicates.
    normalized = frame.astype(float).round(6).fillna(-99999999.)
    return pd.util.hash_pandas_object(normalized,index=False).to_numpy()


def validate_external(task, path, manifest):
    spec = TASKS[task]
    if manifest.get('task') != task or manifest.get('target_definition') != spec['meaning']:
        raise ValueError('Task/target definition not approved')
    for field in ['source_url','license','reviewer','population','measurement_protocol','independence_evidence']:
        if not isinstance(manifest.get(field),str) or not manifest[field].strip():
            raise ValueError(f'Missing reviewed provenance: {field}')
    if manifest.get('independence_reviewed') is not True or manifest.get('measurement_compatibility_reviewed') is not True:
        raise ValueError('Independent-cohort and measurement compatibility reviews required')
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get('csv_sha256'):
        raise ValueError('CSV checksum mismatch')
    model_path = MODELS / f'{task}_model.pkl'
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest.get('frozen_model_sha256'):
        raise ValueError('Frozen model mismatch')
    contracts = json.loads((BASE.parent/'docs/validation/model_input_contracts.json').read_text())
    if manifest.get('feature_units_and_context') != contracts[task]['features']:
        raise ValueError('Feature units/context not matched to model contract')
    frame = pd.read_csv(path)
    expected = set(spec['features']) | {spec['target'],'participant_id'}
    if set(frame) != expected or frame.empty:
        raise ValueError('CSV must contain exact features, target and participant_id')
    if frame.participant_id.isna().any() or frame.participant_id.duplicated().any():
        raise ValueError('One unique nonmissing participant per clinical row required')
    y = pd.to_numeric(frame[spec['target']],errors='raise')
    if set(y.unique()) != {0,1}:
        raise ValueError('Both binary outcomes required; missing/unknown labels forbidden')
    X = clean_features(frame,task)
    if X.isna().all(axis=1).any():
        raise ValueError('Entirely missing participant features')
    hashes = fingerprint(X)
    if len(np.unique(hashes)) != len(hashes):
        raise ValueError('Duplicate feature rows in external cohort')
    for split in ['train','validation','test']:
        known = pd.read_csv(PROCESSED/task/f'{split}.csv')
        if np.intersect1d(hashes,fingerprint(clean_features(known,task))).size:
            raise ValueError(f'External feature overlap with existing {split} set')
    return frame, X, y


def run(task, csv_path, manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    frame,X,y = validate_external(task,csv_path,manifest)
    bundle = joblib.load(MODELS/f'{task}_model.pkl')
    transformed = pd.DataFrame(bundle['preprocessor'].transform(X),columns=bundle['features'])
    p = bundle['model'].predict_proba(transformed)[:,1]
    result = {'task':task,'cases':len(frame),'prevalence':float(y.mean()),
        'model_sha256':manifest['frozen_model_sha256'],'data_sha256':manifest['csv_sha256'],
        'metrics':evaluate(y,p,bundle['threshold']), 'threshold_source':'Frozen baseline artifact; no external tuning',
        'missingness':X.isna().mean().to_dict(),
        'warning':'Source independence is reviewer-attested, not proven by row-overlap checks. External performance does not certify clinical safety.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    # Only aggregate output, never external participant-level health records.
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task',choices=TASKS)
    parser.add_argument('csv',type=Path)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    run(args.task,args.csv,args.manifest,args.output)
