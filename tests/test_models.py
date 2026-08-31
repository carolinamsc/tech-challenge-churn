import numpy as np
import pandas as pd

from src.models.evaluate import evaluate_predictions
from src.models.model_selection import build_models


def test_all_required_models_are_available():
    models = build_models()
    assert set(models) == {"logistic_regression", "random_forest", "mlp_classifier"}


def test_evaluate_predictions_returns_expected_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.8, 0.9])

    result = evaluate_predictions(y_true, probabilities, threshold=0.5)

    assert result["roc_auc"] == 1.0
    assert result["f1"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]
