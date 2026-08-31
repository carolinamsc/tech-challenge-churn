"""Inference utilities for the persisted churn pipeline."""

from pathlib import Path

import joblib
import pandas as pd

from src.utils.config import DEFAULT_THRESHOLD, MODEL_ARTIFACT


def load_model(path: str | Path = MODEL_ARTIFACT):
    """Load the persisted sklearn pipeline."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def predict_churn(model, customer: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Return churn probability, class and a simple risk label."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")

    frame = pd.DataFrame([customer])
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= threshold)
    risk = "high" if probability >= 0.70 else "medium" if probability >= 0.40 else "low"

    return {
        "churn_probability": probability,
        "churn_prediction": prediction,
        "risk_level": risk,
        "threshold": threshold,
    }
