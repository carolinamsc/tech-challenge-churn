"""Probability-threshold analysis for churn intervention decisions."""

import numpy as np
import pandas as pd

from src.models.evaluate import evaluate_predictions


def threshold_analysis(
    y_true: pd.Series, probabilities, thresholds=None
) -> pd.DataFrame:
    """Evaluate classification metrics across candidate thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.20, 0.81, 0.05)

    rows = []
    for threshold in thresholds:
        metrics = evaluate_predictions(y_true, probabilities, float(threshold))
        rows.append(
            {
                "threshold": metrics["threshold"],
                "roc_auc": metrics["roc_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
            }
        )
    return pd.DataFrame(rows)
