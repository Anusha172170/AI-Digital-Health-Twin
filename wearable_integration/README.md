# Wearable integration: scope and current status

## Completed

1. Real WISDM **smartwatch accelerometer window-feature dataset** acquired and modeled. See [wearable ML](../ml_engine/wearable/README.md).
2. Tested provider-neutral observation validator in `common/data_mapper.py`: explicit source record IDs, timezone-aware intervals, unit conversion, missingness, demo flags and batch deduplication.

## Not connected yet

No actual watch or ring has been connected. No HealthKit/Health Connect native SDK, Fitbit/Oura OAuth flow, background sync, production token storage, backend authentication, consent/deletion flow or physical-device test is implemented. The common mapper is a contract for those adapters, not a provider SDK.

**Next required information:** watch/ring brand and exact model, phone OS (Android/iOS), and the manufacturer's companion app. We will select and implement one real connector instead of pretending every device is supported.

## Intended data flow

```text
Device → manufacturer/phone health store → permissioned native/API adapter
       → normalized observations → authenticated backend → history/dashboard
```

Health-store metrics (HR, steps, sleep summaries) do NOT match the 91 WISDM activity-model features. That classifier requires the same source feature extraction semantics and device validation. It cannot consume four health summaries, infer raw motion from them, or manufacture missing clinical tests.

## Data handling contract

- Import only user-authorized metrics; keep original timestamps, units and source IDs.
- Missing heart rate or sleep remains `null`, never defaulted to a healthy reading.
- Observe interval semantics; overlapping sleep sessions must not be summed blindly.
- `observation_key` identifies a provider record within an authenticated user's storage namespace. The backend must namespace keys by its verified user identity; the hash is not an authentication mechanism.
- Deduplication here handles exact repeated source records within a batch, not overlaps across providers or durable sync cursors.
- Physiological plausibility/clinical alerts are not established by structural validation.
- No secrets in Git; real health records are private and must not be committed.
- FCM delivery and delayed health-store sync cannot guarantee emergency monitoring.

Files under `sample_wearable_data` are **fabricated contract examples**, not genuine provider responses. They are marked `is_demo: true` and are never used for model training.
