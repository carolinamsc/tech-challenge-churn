"""Cross-validation experiment for the required churn candidate models."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.loader import load_raw_data, split_features_target
from src.models.model_selection import build_models
from src.utils.config import CV_FOLDS, RANDOM_STATE
from src.utils.reporting import write_report

OUTPUT = Path("reports/cross_validation.csv")
SCORING = {"roc_auc": "roc_auc", "f1": "f1", "precision": "precision", "recall": "recall"}


def run_cross_validation() -> pd.DataFrame:
    """Evaluate all candidate models using stratified cross-validation."""
    df = load_raw_data()
    X, y = split_features_target(df)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, pipeline in build_models().items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    return write_report(result, OUTPUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cross_validation()
