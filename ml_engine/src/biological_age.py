"""No biological-age model is supported by the acquired datasets."""

def predict_biological_age(*args, **kwargs):
    raise NotImplementedError('Requires a defined, validated biological-age target and appropriate longitudinal biomarkers.')
