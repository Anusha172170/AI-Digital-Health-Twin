import json
from pathlib import Path
import pytest
from wearable_integration.common.data_mapper import normalize_observation, normalize_batch

@pytest.fixture
def observation():
    return {'provider': 'healthkit', 'source_record_id': 'test-1', 'metric': 'heart_rate',
            'value': 72, 'unit': 'bpm', 'start_time': '2026-09-19T08:00:00+05:30',
            'end_time': '2026-09-19T09:00:00+05:30', 'is_demo': True}


def test_normalization_missing_and_duplicates(observation):
    result = normalize_observation(observation)
    assert result['start_time'] == '2026-09-19T02:30:00+00:00'
    assert result['quality'] == 'unverified'
    assert len(normalize_batch([observation, observation])) == 1
    assert normalize_observation({**observation, 'value': None})['quality'] == 'missing'
    with pytest.raises(ValueError):
        normalize_batch([observation, {**observation, 'value': 90}])


def test_units(observation):
    assert normalize_observation({**observation, 'metric': 'spo2', 'unit': 'fraction', 'value': .97})['value'] == 97
    assert normalize_observation({**observation, 'metric': 'sleep_duration', 'unit': 'minutes', 'value': 45})['value'] == .75

@pytest.mark.parametrize('change', [
    {'value': float('nan')}, {'value': True}, {'value': -1}, {'value': 0},
    {'provider': 'unknown'}, {'unit': 'mmHg'}, {'is_demo': 'false'},
    {'start_time': '2026-09-19T08:00:00'}, {'end_time': '2026-09-18T08:00:00Z'},
    {'metric': 'steps', 'unit': 'count', 'value': 1.5},
    {'metric': 'spo2', 'unit': '%', 'value': 101},
    {'metric': 'sleep_duration', 'unit': 'hours', 'value': 8},
])
def test_reject_invalid(observation, change):
    with pytest.raises(ValueError):
        normalize_observation({**observation, **change})


def test_samples_explicitly_demo():
    for path in Path('wearable_integration/sample_wearable_data').glob('*.json'):
        result = normalize_observation(json.loads(path.read_text()))
        assert result['is_demo'] is True
