"""Baseline Logistic Regression training and evaluation."""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data.loader import load_raw_data, split_features_target
from src.features.preprocessing import build_preprocessor

RANDOM_STATE = 42


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics for a binary classifier."""

    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    confusion_matrix: list[list[int]]


def build_logistic_pipeline() -> Pipeline:
    """Build the baseline model as one reproducible sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    """Evaluate a fitted binary classifier at the default 0.5 threshold."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return EvaluationResult(
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        f1=float(f1_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions)),
        recall=float(recall_score(y_test, predictions)),
        accuracy=float(accuracy_score(y_test, predictions)),
        confusion_matrix=confusion_matrix(y_test, predictions).tolist(),
    )


def cross_validate_baseline(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Run stratified cross-validation without leaking preprocessing across folds."""
    model = build_logistic_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
        },
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "metric": ["roc_auc", "f1", "precision", "recall"],
            "mean": [
                scores["test_roc_auc"].mean(),
                scores["test_f1"].mean(),
                scores["test_precision"].mean(),
                scores["test_recall"].mean(),
            ],
            "std": [
                scores["test_roc_auc"].std(),
                scores["test_f1"].std(),
                scores["test_precision"].std(),
                scores["test_recall"].std(),
            ],
        }
    )


def train_baseline() -> tuple[Pipeline, EvaluationResult]:
    """Train the baseline and return its untouched-test evaluation."""
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = build_logistic_pipeline()
    model.fit(X_train, y_train)
    return model, evaluate_classifier(model, X_test, y_test)


if __name__ == "__main__":
    baseline, result = train_baseline()
    print("Baseline Logistic Regression")
    print(result)
