"""Train and compare candidate churn models on a reproducible holdout split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data, split_features_target
from src.models.evaluate import evaluate_predictions
from src.models.model_selection import build_models
from src.utils.config import RANDOM_STATE, TEST_SIZE

OUTPUT = Path("reports/model_results.csv")


def compare_models() -> pd.DataFrame:
    """Fit all candidate models and persist holdout metrics."""
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    rows = []
    for name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, probabilities)
        rows.append({"model": name, **metrics})

    result = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    compare_models()
