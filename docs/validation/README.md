# Health-app model readiness: four workstreams

1. **Input definition completed:** [health-app fields and contract](health_app_inputs.md), machine-readable `model_input_contracts.json`, tested `model_readiness.py` audit. No live connector implemented.
2. **Dataset suitability review completed:** [three sourced candidates](longitudinal_dataset_review.md), with longitudinal coverage, outcomes, licensing/access and deployment mismatch. No new dataset downloaded or model trained in this step.
3. **Independent evaluation workflow implemented; actual validation blocked:** [protocol](independent_validation_protocol.md) and frozen evaluator. Needs a reviewed compatible independent cohort; no external performance claim is made.
4. **Experimental threshold analysis executed:** [measured operating points](../../ml_engine/reports/thresholds/README.md). Example constraints, not clinician-approved alert requirements; no production thresholds changed.

## Findings

Neither CHD nor hypertension baseline satisfies recall >=80% AND precision >=50% on validation. Even feasible diabetes validation policies fail to maintain the requested recall on historical test. A threshold change cannot make these models ready for real-time clinical alerts.

Next decision: implement health-app record ingestion/dashboard first; choose a feasible, explicitly defined additional model target only after validating input availability and data access. Do not substitute a motion-classification dataset for clinical risk validation.
