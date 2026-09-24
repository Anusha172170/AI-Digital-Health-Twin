# Longitudinal dataset suitability review

Reviewed official source documentation on 2026-09-19. **Documentation reviewed only: these new datasets have not been downloaded or used for training.** Version pins below identify the reviewed releases; some have newer releases whose changes must be reviewed before acquisition.

## 1. BIG IDEAs Lab Glycemic Variability and Wearable Device Data

[Official version 1.1.2](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/) · DOI: 10.13026/zthx-5212 · Cho, Kim, Bent and Dunn (2023).

- 16 participants, 8–10 days, Empatica E4 observations with Dexcom G6 interstitial glucose every five minutes; HR, motion, BVP, EDA, IBI, temperature and food logs. Documentation provides HbA1c/gender metadata.
- Open access under Open Data Commons Attribution License v1.0; cite dataset and original papers. Source dates are shifted.
- Useful candidate for a NEW short-horizon glycemic-forecasting research model. Define a 30-minute forecast target, past-only windows and a persistence baseline before training; compare a CGM-history model with a separately labeled wearable-only experiment. Do not imply CGM history exists for ordinary smartwatch users.
- Small, selected population; thousands of windows do not equal thousands of independent patients. Reserve participants and a separate cohort for external evaluation; temporal gaps within a person must cover lookback and forecast windows.
- **Not external validation for PIMA diabetes status or ten-year CHD.** CGM interstitial glucose is not PIMA's two-hour plasma glucose; source usage notes say OGTT files were corrupted/unavailable. No long-term disease-onset target is established.
- Raw EDA/PPG/motion may not be exposed through ordinary health-app APIs. Requires input-availability audit before choosing features. Not a basis for non-invasive glucose monitoring claims.

## 2. BIDSleep multi-night Apple Watch / EEG dataset

[Official version 1.0.0](https://physionet.org/content/bidsleep-dataset/1.0.0/) · DOI: 10.13026/a0sy-7t69 · Song (2026).

- 47 healthy adults, 253 nights, up to seven nights each. Instantaneous HR and three-axis motion from a custom watchOS app, aligned to Dreem-2 EEG-derived sleep stages; automated and expert-reviewed labels are distinguished in the documentation.
- Open access under Open Data Commons Attribution License v1.0.
- Useful for a NEW sleep-staging study, not diagnosis of sleep disorders or chronic disease. Split by participant across all nights, not random epochs. Use expert labels, exclude unknown labels and account for device time alignment/timezones.
- Important deployment mismatch: the source used custom watch recording; routine health-app sleep/HR summaries do not guarantee equivalent cadence or motion access. This project's software-app integration scope must not silently expand to direct sensor collection.
- No demonstrated compatibility with current diabetes/CHD/hypertension models. Not acquired here.

## 3. DREAMT

[Official reviewed version 1.0.0](https://physionet.org/content/dreamt/1.0.0/) · DOI: 10.13026/62an-cb28 · Wang, Yang, Shetty and Dunn (2024).

- 100 participants recruited through a sleep-disorder laboratory, one overnight study per participant, Empatica E4 physiological signals aligned to technician-annotated PSG stages and clinical metadata.
- Restricted access; the reviewed version requires signing the project's data-use agreement. Do not bypass access terms with an unofficial mirror. Newer 2.x releases exist and must be reviewed separately.
- Potential separate-cohort test for a future sleep model, only after harmonizing shared inputs and labels. Device, health status and laboratory/home differences are substantial and should be reported, not concealed.
- **Not a longitudinal disease-incidence cohort**, and not an external test set for our current clinical models. Not acquired here.

## Decision

None of these sources justifies claiming independent validation of the existing clinical risk models. A future sleep module has a possible development/external-cohort pairing (BIDSleep and DREAMT), conditional on lawful access and API-feature compatibility. A future CGM-enabled glycemic forecasting module has a small development candidate (BIG IDEAs), but still needs an independent cohort and explicit CGM availability.

For the immediate application, prioritize authorized HR/steps/sleep history and provenance-aware dashboards. Do not turn imported sleep summaries into validated disease-risk probabilities. Keep clinical models explicitly research-only until matching independent clinical data and a reviewed intended-use protocol are available.
