# Research threshold analysis

No alert thresholds deployed. Baseline models and inference remain unchanged.
Experimental constraints: recall >= 80% or 90%, AND precision >= 50%, selected on validation only. These are research examples, not agreed clinical requirements.
Select maximum specificity among feasible validation thresholds; report infeasible instead of silently relaxing constraints.
Historical test performance is descriptive; constraints satisfied on validation need not hold on test or deployment.

| Task | Policy | Status | Test precision | Test recall | Test false positives | Test missed positives |
|---|---|---|---:|---:|---:|---:|
| diabetes | existing_baseline | Research only | 0.506 | 0.830 | 43 | 9 |
| diabetes | recall_80 | Research only | 0.520 | 0.736 | 36 | 14 |
| diabetes | recall_90 | Research only | 0.494 | 0.774 | 42 | 12 |
| cvd | existing_baseline | Research only | 0.207 | 0.752 | 371 | 32 |
| cvd | recall_80 | Infeasible on validation | — | — | — | — |
| cvd | recall_90 | Infeasible on validation | — | — | — | — |
| hypertension | existing_baseline | Research only | 0.454 | 0.673 | 213 | 86 |
| hypertension | recall_80 | Infeasible on validation | — | — | — | — |
| hypertension | recall_90 | Infeasible on validation | — | — | — | — |

Flags per 1,000 are per historical test CASES, not per day or per patient stream. No alert-frequency or emergency-detection claim can be made from these tabular data.
Repeated real-time evaluation of long-horizon disease scores must not create repeated emergency alerts. Event deduplication, persistence checks, notification caps, symptoms, clinical review and explicit consent require separate design/validation.
