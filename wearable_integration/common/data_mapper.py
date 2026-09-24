"""Provider-neutral ingestion contract for future native/OAuth adapters.

This module validates normalized observations. It does NOT connect to a device,
request permissions, read proprietary exports, or infer clinical measurements.
"""
from datetime import datetime, timezone
import hashlib
import json
import math

PROVIDERS = {'health_connect', 'healthkit', 'fitbit', 'oura'}
UNITS = {'heart_rate': 'bpm', 'spo2': '%', 'steps': 'count', 'sleep_duration': 'hours'}


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError('ISO timestamp must be a string')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return result.astimezone(timezone.utc)


def normalize_observation(record):
    """Normalize one explicit observation; missing values remain missing.

    A native adapter must supply actual source, record ID, units and timestamps.
    No user identity or access token belongs in this portable sample contract.
    """
    required = {'provider', 'source_record_id', 'metric', 'value', 'unit', 'start_time', 'end_time', 'is_demo'}
    if set(record) != required:
        raise ValueError('Invalid observation keys')
    provider, metric = record['provider'], record['metric']
    if provider not in PROVIDERS or metric not in UNITS:
        raise ValueError('Unsupported provider or metric')
    if not isinstance(record['source_record_id'], str) or not record['source_record_id'].strip():
        raise ValueError('Source record ID required')
    if not isinstance(record['is_demo'], bool):
        raise ValueError('is_demo must be explicit boolean')
    start, end = _timestamp(record['start_time']), _timestamp(record['end_time'])
    if end < start:
        raise ValueError('End precedes start')
    unit = record['unit']
    supported = {
        'heart_rate': {'bpm': 1}, 'spo2': {'%': 1, 'fraction': 100},
        'steps': {'count': 1}, 'sleep_duration': {'hours': 1, 'minutes': 1/60, 'seconds': 1/3600}}
    if unit not in supported[metric]:
        raise ValueError('Incompatible measurement unit')
    value = record['value']
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Value must be finite numeric or null')
        value *= supported[metric][unit]
        if value < 0 or (metric == 'spo2' and value > 100):
            raise ValueError('Measurement outside structural bounds')
        if metric == 'heart_rate' and value == 0:
            raise ValueError('Zero pulse is not a valid missing-data marker; use null')
        if metric == 'steps':
            if int(value) != value:
                raise ValueError('Step count must be integral')
            value = int(value)
        if metric in {'steps', 'sleep_duration'} and end == start:
            raise ValueError('Aggregate measurement requires a nonempty interval')
        if metric == 'sleep_duration' and value > (end - start).total_seconds() / 3600:
            raise ValueError('Sleep duration exceeds observation interval')
    canonical_id = json.dumps([provider, record['source_record_id'], metric], separators=(',', ':'))
    return {**record, 'value': value, 'unit': UNITS[metric],
            'start_time': start.isoformat(), 'end_time': end.isoformat(),
            'quality': 'missing' if value is None else 'unverified',
            'observation_key': hashlib.sha256(canonical_id.encode()).hexdigest()}


def normalize_batch(records):
    """Deduplicate source IDs within one batch; conflicting updates must be explicit."""
    mapped = {}
    for record in records:
        result = normalize_observation(record)
        key = result['observation_key']
        if key in mapped and mapped[key] != result:
            raise ValueError('Conflicting duplicate source record')
        mapped[key] = result
    return list(mapped.values())
