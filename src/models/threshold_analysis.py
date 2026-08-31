"""Threshold analysis for churn intervention decisions."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data, split_features_target
from src.models.model_selection import build_models
from src.utils.config import RANDOM_STATE, TEST_SIZE
from src.utils.reporting import write_report

OUTPUT = Path("reports/threshold_analysis.csv")
THRESHOLDS = [round(x / 100, 2) for x in range(20, 71, 5)]


def analyze_thresholds(model_name: str = "logistic_regression") -> pd.DataFrame:
    """Evaluate precision, recall, F1 and intervention volume across thresholds."""
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    models = build_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    model = models[model_name]
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    rows = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "intervention_rate": predictions.mean(),
                "intervention_count": int(predictions.sum()),
            }
        )

    result = pd.DataFrame(rows)
    return write_report(result, OUTPUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyze_thresholds()
