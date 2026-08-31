"""Candidate churn classification models."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessor
from src.utils.config import RANDOM_STATE


def build_models() -> dict[str, Pipeline]:
    """Return the three required candidate models as complete pipelines."""
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "mlp_classifier": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
                random_state=RANDOM_STATE
            )),
        ]),
    }
