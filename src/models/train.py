"""Train and persist a churn classification pipeline."""

from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data, split_features_target
from src.models.model_selection import build_models
from src.utils.config import MODEL_ARTIFACT, RANDOM_STATE, TEST_SIZE


def train_and_save(model_name: str = "logistic_regression") -> Path:
    """Fit a named model on the training split and persist the full pipeline."""
    models = build_models()
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {sorted(models)}")

    df = load_raw_data()
    X, y = split_features_target(df)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = models[model_name]
    pipeline.fit(X_train, y_train)

    artifact = Path(MODEL_ARTIFACT)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact)
    return artifact


if __name__ == "__main__":
    print(f"Saved model to {train_and_save()}")
