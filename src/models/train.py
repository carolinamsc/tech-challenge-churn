"""Train and persist the selected churn classification pipeline."""

import logging
from pathlib import Path

import joblib

from src.data.loader import load_raw_data, split_features_target
from src.models.model_selection import build_models
from src.utils.config import MODEL_ARTIFACT

logger = logging.getLogger(__name__)


def train_and_save(model_name: str = "logistic_regression") -> Path:
    """Fit the selected model on all available labeled data and persist the full pipeline."""
    models = build_models()
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {sorted(models)}")

    df = load_raw_data()
    X, y = split_features_target(df)
    pipeline = models[model_name]
    pipeline.fit(X, y)

    artifact = Path(MODEL_ARTIFACT)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact)
    return artifact


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("saved model to %s", train_and_save())
