"""Single-command wearable + clinical V2 training, artifact verification and test runner.

python -m ml_engine.src.complete_training [--download]
Existing clinical raw/splits and baseline bundles are prerequisites (never overwritten).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ML = REPO / 'ml_engine'
OUT = ML / 'reports/completion'


def hashes(paths):
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Reacquire pinned WISDM feature files; requires GitHub CLI')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_paths = list((ML / 'saved_models').glob('*'))
    baseline = hashes(baseline_paths)
    if not baseline:
        raise FileNotFoundError('Clinical baseline artifacts required; see ml_engine/README.md')
    manifest = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_revision_at_start': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','shap','joblib']},
        'training_source_sha256': hashes((ML / 'src').glob('*.py')), 'stages': []}
    mpath = OUT / 'run_manifest.json'
    mpath.write_text(json.dumps(manifest, indent=2)+'\n')
    environment = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    stages = []
    if args.download:
        stages.append(('download', ['-m', 'ml_engine.src.wearable_pipeline', 'download']))
    stages += [('wearable_preprocess', ['-m', 'ml_engine.src.wearable_pipeline', 'preprocess']),
               ('wearable_train', ['-m', 'ml_engine.src.wearable_pipeline', 'train']),
               ('clinical_v2_train', ['-m', 'ml_engine.src.improve_models']),
               ('saved_model_verification', ['-m', 'ml_engine.src.verify_saved_models']),
               ('tests', ['-m', 'pytest', 'tests/ml_tests', 'tests/wearable_tests', '-q'])]
    try:
        for name, command in stages:
            print(f'Starting {name}', flush=True)
            started = time.monotonic()
            with (OUT / f'{name}.log').open('w') as log:
                result = subprocess.run([sys.executable, *command], cwd=REPO, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            manifest['stages'].append({'name': name, 'command': [sys.executable,*command],
                'returncode': result.returncode, 'duration_seconds': round(time.monotonic()-started,2)})
            mpath.write_text(json.dumps(manifest,indent=2)+'\n')
            if result.returncode:
                raise RuntimeError(f'{name} failed; see {OUT / (name+".log")}')
            print(f'Completed {name}', flush=True)
        if hashes(baseline_paths) != baseline:
            raise RuntimeError('Baseline models changed unexpectedly')
        manifest['baseline_unchanged'] = True
        manifest['baseline_sha256'] = baseline
        tracked_outputs = [*(ML/'wearable').rglob('*'), *(ML/'experiments/v2').rglob('*')]
        manifest['output_sha256'] = hashes(tracked_outputs)
        manifest['input_sha256'] = hashes((ML/'datasets/processed').rglob('*.csv'))
        manifest['status'] = 'passed'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        mpath.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'All stages passed. Manifest: {mpath}', flush=True)

if __name__ == '__main__':
    main()
