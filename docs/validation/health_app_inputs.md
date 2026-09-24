# Health-application input contract (design, not a connected integration)

Scope: software imports authorized records from existing health applications/platforms; no Bluetooth pairing, sensor firmware or custom physical devices.

## Sources and availability

- Android Health Connect: per-type permissions; background and history reads need additional permissions and feature availability checks. [4](https://developer.android.com/health-and-fitness/health-connect/data-types)
- Apple HealthKit: per-type authorization and native HealthKit capability. An empty read must not be interpreted as confirmed read denial or proof that the user has no measurements. [Apple official authorization documentation](https://developer.apple.com/documentation/healthkit/authorizing-access-to-health-data).
- Fitbit/Oura/provider cloud adapters: planned, not implemented or verified here. Verify current scopes, terms, data semantics and latency before development.
- Browser clients access our authenticated backend; they do not query HealthKit or Health Connect directly.

| Canonical field | Unit/semantics | Proposed source | Model use now |
|---|---|---|---|
| heart_rate | bpm, timestamped instantaneous observation | Authorized app record | Dashboard only; not automatically equivalent to clinical resting HR |
| resting_heart_rate | bpm, source-defined summary and interval | Authorized app record if present | Clinical context must be reviewed before mapping to Framingham heartRate |
| steps | Count within explicit non-overlapping interval | Authorized app aggregate | Dashboard; none of current disease models uses steps |
| sleep_duration | Hours asleep, NOT time in bed | Authorized sleep sessions, preserving stage/source | Dashboard; no validated sleep-risk model yet |
| spo2 | Percent; original observation context retained | Only if source publishes a measurement | Dashboard; not a current clinical model input |
| glucose | mg/dL AND specimen/protocol metadata | Clinical/CGM app if authorized | PIMA requires two-hour OGTT plasma glucose; CGM/interstitial/fasting values are not interchangeable |
| systolic/diastolic BP | mmHg, measurement context | Existing app containing actual measurement | Not derived from HR; candidate clinical input after review |
| weight, height, BMI | kg, m, kg/m² and measurement date | Authorized records or explicit user entry | BMI clinical provenance required |
| age, history, smoking, medications | Profile/history with documented definitions | Consent-based profile/clinical entry | Needed for applicable models; never infer from activity |
| insulin, skin thickness, pedigree, cholesterol | Task-specific clinical units/definitions | Clinical data entry/import | Usually unavailable from basic watch-app summaries |

Only `heart_rate`, `steps`, `sleep_duration`, `spo2` are implemented by the common observation mapper. Other fields in this table are requirements, NOT implemented adapters. Existing provider samples are demo contract records, not real API responses.

## Import/sync requirements

Each observation needs a verified user's storage namespace, source app/provider, original record ID, original unit, metric context, interval, observed timestamp, ingestion timestamp, demo flag and quality status. These wider backend fields are planned; the current small mapper is not the full production envelope. Preserve changes/deletions, pagination cursors and revocation. Deduplicate exact records and resolve overlapping sources before aggregating steps or sleep. Display last source observation AND last synchronization; never label old data live.

Only request needed read permissions; no writes by default. Consent to read a platform record is not blanket consent to export/share it. Encrypt transport/storage and establish retention/deletion/access controls before real-user use.

## Model eligibility

`wearable_integration/common/model_readiness.py` audits explicit model-feature records and reports missing, invalid or incompatible inputs. It requires canonical units/context and provenance; it does not fabricate fields from health summaries. Temporal validity must be supplied as an explicit engineering policy (`max_age_days`), not inferred as a medical standard. Requiring complete inputs is intentionally stricter than the offline median-imputing benchmark scripts.

Passing this audit means only **input-contract ready for research**. Every result still has `clinical_deployment_allowed=false` and requires independent clinical validation. The helper is not yet wired into any backend route; existing offline predict scripts remain benchmark tools.
