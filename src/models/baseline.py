"""Baseline model training and evaluation helpers."""

from src.data.loader import load_raw_data, split_features_target
from src.models.evaluate import evaluate_predictions
from src.models.model_selection import build_models
from src.utils.config import RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split


def train_baseline():
    """Train the Logistic Regression baseline and evaluate it on the holdout set."""
    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = build_models()["logistic_regression"]
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    return model, evaluate_predictions(y_test, probabilities, threshold=0.50)


if __name__ == "__main__":
    baseline, result = train_baseline()
    print("Baseline Logistic Regression")
    print(result)
