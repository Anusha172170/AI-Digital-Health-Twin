import copy
from datetime import datetime,timezone
import hashlib
import json
import pandas as pd
import pytest
from ml_engine.src.config import TASKS, MODELS, PROCESSED
from ml_engine.src.external_validation import validate_external, run
from ml_engine.src.threshold_analysis import choose_operating_point
from wearable_integration.common.model_readiness import assess_inputs, CONTRACT_PATH


def test_threshold_infeasible_and_feasible():
    assert choose_operating_point([0,0,0,1],[.5,.5,.5,.5],.8,.5) is None
    good = choose_operating_point([0,0,1,1],[.1,.2,.8,.9],.9,.5)
    assert good['precision'] == good['recall'] == 1
    assert good['threshold'] == .8


def test_input_contract():
    contracts = json.loads(CONTRACT_PATH.read_text())
    records = {name:{'value':1,'unit':spec['unit'],'context':spec['context'],
        'source':'test_source','is_demo':False,'observed_at':'2026-09-19T00:00:00Z'}
        for name,spec in contracts['diabetes']['features'].items()}
    now = datetime(2026,9,19,1,tzinfo=timezone.utc)
    ready = assess_inputs('diabetes',records,max_age_days=1,now=now)
    assert ready['input_contract_ready'] and not ready['clinical_deployment_allowed']
    records['glucose']['context'] = 'interstitial_cgm'
    assert not assess_inputs('diabetes',records,max_age_days=1,now=now)['input_contract_ready']
    del records['age']
    assert any(i['reason']=='missing' for i in assess_inputs('diabetes',records,max_age_days=1,now=now)['issues'])
    records['glucose']['observed_at'] = '2025-01-01T00:00:00Z'
    assert any(i['reason']=='future_or_stale' for i in assess_inputs('diabetes',records,max_age_days=1,now=now)['issues'])

@pytest.fixture
def external_fixture(tmp_path):
    # Artificial fixture to exercise code, NOT an independent clinical dataset.
    task='diabetes'; spec=TASKS[task]
    f=pd.read_csv(PROCESSED/task/'test.csv').groupby(spec['target']).head(2)
    f=f[spec['features']+[spec['target']]].copy()
    f['participant_id']=[f'fixture-{i}' for i in range(len(f))]
    path=tmp_path/'fixture.csv'
    f.to_csv(path,index=False)
    manifest={'task':task,'target_definition':spec['meaning'],
        'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'frozen_model_sha256':hashlib.sha256((MODELS/f'{task}_model.pkl').read_bytes()).hexdigest(),
        'independence_reviewed':True,'measurement_compatibility_reviewed':True,
        'feature_units_and_context':json.loads(CONTRACT_PATH.read_text())[task]['features']}
    for key in ['source_url','license','reviewer','population','measurement_protocol','independence_evidence']:
        manifest[key]='SYNTHETIC CODE TEST ONLY'
    return task,path,manifest,f


def test_reject_reused_test_cohort(external_fixture):
    task,path,m,_=external_fixture
    with pytest.raises(ValueError,match='overlap'):
        validate_external(task,path,m)


def test_reject_bad_provenance(external_fixture):
    task,path,m,_=external_fixture
    m['independence_reviewed']=False
    with pytest.raises(ValueError,match='reviews required'):
        validate_external(task,path,m)
    m['independence_reviewed']=True
    m['target_definition']='future diabetes'
    with pytest.raises(ValueError,match='target definition'):
        validate_external(task,path,m)


def test_frozen_evaluator_plumbing(external_fixture,tmp_path):
    task,path,m,f=external_fixture
    # Perturb only for testing an executable path; never used as empirical evidence.
    f['age']+=.012345
    f.to_csv(path,index=False)
    m['csv_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    mp=tmp_path/'manifest.json'; mp.write_text(json.dumps(m))
    output=tmp_path/'result.json'
    result=run(task,path,mp,output)
    assert result['cases']==len(f)
    assert result['threshold_source'].startswith('Frozen')
    assert 'participant_id' not in output.read_text()
    m['frozen_model_sha256']='bad'
    with pytest.raises(ValueError,match='model mismatch'):
        validate_external(task,path,m)
