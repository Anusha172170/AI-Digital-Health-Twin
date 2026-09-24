from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / 'datasets' / 'raw'
PROCESSED = BASE / 'datasets' / 'processed'
MODELS = BASE / 'saved_models'
REPORTS = BASE / 'reports'
SEED = 42
PIMA_COLUMNS = ['pregnancies', 'glucose', 'diastolic_bp', 'skin_thickness',
                'insulin', 'bmi', 'pedigree', 'age', 'Outcome']
TASKS = {
    'diabetes': {
        'file': 'diabetes.csv', 'target': 'Outcome',
        'features': PIMA_COLUMNS[:-1],
        'zero_missing': ['glucose', 'diastolic_bp', 'skin_thickness', 'insulin', 'bmi'],
        'meaning': 'PIMA recorded diabetes status; not prospective diabetes incidence'},
    'cvd': {
        'file': 'framingham.csv', 'target': 'TenYearCHD',
        'features': ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
                     'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
                     'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'],
        'zero_missing': ['totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'],
        'meaning': 'Recorded ten-year coronary heart disease outcome, not all CVD'},
    'hypertension': {
        'file': 'framingham.csv', 'target': 'prevalentHyp',
        'features': ['male', 'age', 'currentSmoker', 'cigsPerDay', 'totChol', 'BMI', 'heartRate', 'glucose'],
        'zero_missing': ['totChol', 'BMI', 'heartRate', 'glucose'],
        'meaning': 'Exploratory prevalent hypertension classification; NOT future risk. BP, BP medication and future CHD excluded'},
}
