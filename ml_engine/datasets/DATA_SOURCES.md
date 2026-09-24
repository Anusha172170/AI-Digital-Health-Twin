# Dataset provenance and model suitability

## PIMA diabetes benchmark

- Acquired mirror: https://github.com/jbrownlee/Datasets/blob/d20fcb6402ae34e653d4513b00f39257bb37ed7f/pima-indians-diabetes.data.csv
- 768 rows, eight clinical predictors and one recorded diabetes outcome.
- Historical Pima Indian women aged at least 21 benchmark; not representative of all populations or of young male app users.
- Columns: pregnancies, two-hour plasma glucose during oral glucose tolerance test (mg/dL), diastolic BP (mmHg), skin thickness (mm), two-hour insulin (mu U/ml), BMI (kg/m²), pedigree function, age, outcome.
- Zero-coded glucose/BP/skin/insulin/BMI are treated as missing. Pregnancies=0 remains valid.
- Purpose: academic diabetes-status classification. Not proof of future disease prediction or causal lifestyle effects.

## Framingham-derived CHD benchmark

- Acquired mirror: https://github.com/OpenMined/TenSEAL/blob/c962074f0aa82cf30eeaddf17af887a8ea0e66fe/tutorials/data/framingham.csv
- 4,238 rows and 16 columns in this downloaded version. Do not assume the 4,240-row count quoted in other mirrors.
- Clinical measurements include age, sex coding, smoking, medication/history indicators, cholesterol, systolic/diastolic BP, BMI, heart rate and glucose; target is `TenYearCHD`.
- Purpose: ten-year coronary heart disease benchmark. `prevalentHyp` additionally supports exploratory prevalent-hypertension classification, not prospective onset prediction.
- This mirror has no stable patient identifiers or detailed measurement protocol. Institutional origin does not imply the mirror is an institution-endorsed or audited release.

## Provenance and rights

Exact source bytes and pinned commit URLs/checksums are recorded in `raw/manifest.json`. Named CSVs preserve source row order. Downloads are real public benchmark records, not generated data. Neither source was downloaded directly from its original clinical institution. Dataset-specific redistribution rights are not established by a hosting repository's software license. Keep attribution, review upstream terms and obtain permission where needed before broader publication or commercial use; no new dataset license is asserted here.

## Why no PPG-BP, WESAD or PMData download?

They are not necessary to train these tabular benchmark models. The pasted conversation's dataset URLs and claimed targets cannot be assumed correct. No PPG, sleep, biological-age labels or wearable measurements have been fabricated. A separate sourced, licensed dataset and evaluation plan are required for each additional task.
