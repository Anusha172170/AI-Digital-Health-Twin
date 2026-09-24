"""Actual TreeSHAP contributions in raw model margin (log-odds), not percentages."""
import numpy as np
import shap


def explain(model, X):
    result = shap.TreeExplainer(model)(X)
    margins = model.predict(X, output_margin=True)
    np.testing.assert_allclose(result.base_values + result.values.sum(axis=1), margins, atol=1e-5)
    return result
