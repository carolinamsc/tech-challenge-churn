"""Tests for the PSI drift monitoring helpers."""

import numpy as np
import pandas as pd

from src.monitoring.drift import classify, compute_drift


def sample_frame(seed: int, shift: float = 0.0, churn_heavy: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = 2000
    contracts = ["Month-to-month", "One year", "Two year"]
    weights = [0.9, 0.05, 0.05] if churn_heavy else [0.55, 0.21, 0.24]
    return pd.DataFrame(
        {
            "MonthlyCharges": rng.normal(65 + shift, 30, size),
            "Contract": rng.choice(contracts, size=size, p=weights),
        }
    )


def test_no_drift_between_samples_of_the_same_population():
    result = compute_drift(sample_frame(1), sample_frame(2))
    assert (result["status"] == "stable").all()


def test_drift_is_detected_on_shifted_features():
    result = compute_drift(sample_frame(1), sample_frame(2, shift=40, churn_heavy=True))
    status = dict(zip(result["feature"], result["status"], strict=True))
    assert status["MonthlyCharges"] == "alert"
    assert status["Contract"] != "stable"


def test_classify_thresholds():
    assert classify(0.05) == "stable"
    assert classify(0.15) == "warning"
    assert classify(0.30) == "alert"
