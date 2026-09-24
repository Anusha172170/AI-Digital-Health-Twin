# Independent validation protocol — execution pending compatible data

## Current status

**No external clinical validation has been completed.** The existing test sets have been inspected during development and cannot be represented as fresh external evidence. A different mirror of PIMA or Framingham is not an independent cohort. WISDM, glycemic sensor time series and sleep stage datasets have different labels/input definitions and cannot validate our current three clinical classifiers as-is.

## Freeze before access

1. Choose baseline artifact and record SHA-256, feature units/context, target definition, threshold, intended population, study eligibility and primary metrics.
2. Register data source/license, sampling frame and independence evidence; a reviewer confirms no known subject/source overlap and clinically equivalent measurements/outcomes. Automated checks alone cannot do this.
3. For current models use one baseline record per person. For future longitudinal models split by person, reserve entire cohorts/sites, and ensure no future observations enter features. For within-person forecasting additionally use chronological cutoffs and a gap covering feature/label windows.
4. Do not fit imputation, scaling, calibration, feature selection or thresholds on external data. Any adaptation requires a separate adaptation cohort and another untouched test cohort.
5. Report AP, ROC-AUC, calibration/Brier, precision, recall, specificity, prevalence, missingness, subgroup results where adequately supported, confidence intervals and exact denominators. Prespecify cohort size/event-count justification. Avoid promises of a fixed percentage accuracy.
6. Clinical release needs review of intended use, costs of missed events/false alarms, monitoring and applicable legal/regulatory/privacy obligations.

## Implemented evaluator

`python -m ml_engine.src.external_validation TASK PRIVATE_CSV REVIEWED_MANIFEST OUTPUT_JSON`

It checks the frozen model/data checksums, declared units/context and target, provenance/review attestations, both binary outcomes, unique participant rows and exact/rounded feature overlap against all existing splits. It then runs the frozen baseline without fitting anything and exports aggregate metrics/missingness. It does not yet implement subgroup analysis or confidence intervals; these remain required study reporting work. Near duplicates, edited copies and unrecorded patient overlap cannot be automatically ruled out.

The manifest needs: `task`, `target_definition` (exact TASKS meaning), `csv_sha256`, `frozen_model_sha256`, `source_url`, `license`, `reviewer`, `population`, `measurement_protocol`, `independence_evidence`, `independence_reviewed: true`, `measurement_compatibility_reviewed: true`, and `feature_units_and_context` copied from the reviewed input contract. Do not set review flags to true without actual review.

CSV: exactly all model feature columns + model target + `participant_id`; no names/emails. Missing feature measurements may be handled by the already-fitted benchmark imputer, but are explicitly reported; missing labels are forbidden. Keep private files under ignored `private_validation/` or outside the repository. Tests use synthetic fixtures only to test code, never to claim real-world validation.

## Required next resource

A licensed independent clinical cohort with compatible PIMA/Framingham measurements and label definitions, or a new model trained for a well-defined health-app-supported outcome with a separate external cohort. None of the reviewed wearable datasets meets all requirements for current disease-model validation.
