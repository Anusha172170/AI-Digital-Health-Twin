"""Strict input audit; never an authorization to deploy clinical predictions."""
from datetime import datetime, timezone
from pathlib import Path
import json
import math

CONTRACT_PATH = Path(__file__).resolve().parents[2] / 'docs/validation/model_input_contracts.json'


def assess_inputs(task, records, *, max_age_days, now=None):
    if not isinstance(max_age_days,(int,float)) or isinstance(max_age_days,bool) or not math.isfinite(max_age_days) or max_age_days <= 0:
        raise ValueError('Provide an explicit positive freshness policy')
    contract = json.loads(CONTRACT_PATH.read_text())[task]
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('now must be timezone aware')
    issues = []
    for feature, spec in contract['features'].items():
        record = records.get(feature)
        if not isinstance(record,dict):
            issues.append({'feature':feature,'reason':'missing'})
            continue
        value = record.get('value')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            issues.append({'feature':feature,'reason':'invalid_value'})
        if record.get('unit') != spec['unit'] or record.get('context') != spec['context']:
            issues.append({'feature':feature,'reason':'incompatible_unit_or_context'})
        if not isinstance(record.get('source'),str) or not record['source'].strip() or record.get('is_demo') is not False:
            issues.append({'feature':feature,'reason':'missing_provenance_or_demo'})
        try:
            at = datetime.fromisoformat(record['observed_at'].replace('Z','+00:00'))
            if at.tzinfo is None:
                raise ValueError('Timezone missing')
            age = (now-at).total_seconds()/86400
            if age < 0 or age > max_age_days:
                issues.append({'feature':feature,'reason':'future_or_stale'})
        except (KeyError,TypeError,ValueError,AttributeError):
            issues.append({'feature':feature,'reason':'invalid_timestamp'})
    for extra in set(records)-set(contract['features']):
        issues.append({'feature':extra,'reason':'unexpected_field'})
    return {'task':task,'input_contract_ready':not issues,'clinical_deployment_allowed':False,
        'issues':issues,'warning':'Schema/context audit only. No clinical validity, numeric physiological range check, or provider authenticity established.'}
