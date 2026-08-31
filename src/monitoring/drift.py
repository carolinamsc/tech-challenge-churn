"""Population Stability Index (PSI) checks for input drift monitoring."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data, split_features_target
from src.utils.config import RANDOM_STATE, TEST_SIZE
from src.utils.reporting import write_report

OUTPUT = Path("reports/drift_baseline.csv")
BINS = 10
WARNING_THRESHOLD = 0.10
ALERT_THRESHOLD = 0.25

logger = logging.getLogger(__name__)


def classify(psi: float) -> str:
    """Translate a PSI value into an operational status."""
    if psi >= ALERT_THRESHOLD:
        return "alert"
    if psi >= WARNING_THRESHOLD:
        return "warning"
    return "stable"


def psi_from_shares(reference_share: np.ndarray, current_share: np.ndarray) -> float:
    """Compute PSI between two discrete distributions, guarding empty buckets."""
    epsilon = 1e-6
    reference_share = np.clip(reference_share, epsilon, None)
    current_share = np.clip(current_share, epsilon, None)
    delta = current_share - reference_share
    return float(np.sum(delta * np.log(current_share / reference_share)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = BINS) -> float:
    """PSI for a numeric feature using quantile bins from the reference sample."""
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = np.histogram(reference.dropna(), bins=edges)[0] / len(reference.dropna())
    current_share = np.histogram(current.dropna(), bins=edges)[0] / len(current.dropna())
    return psi_from_shares(reference_share, current_share)


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature using the categories observed in reference."""
    categories = reference.dropna().unique()
    reference_share = np.array([(reference == category).mean() for category in categories])
    current_share = np.array([(current == category).mean() for category in categories])
    return psi_from_shares(reference_share, current_share)


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Compute PSI per feature and classify the drift status."""
    rows = []
    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            psi = numeric_psi(reference[column], current[column])
        else:
            psi = categorical_psi(reference[column], current[column])
        rows.append({"feature": column, "psi": psi, "status": classify(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def run_baseline_drift() -> pd.DataFrame:
    """Compare train and holdout partitions from the same population."""
    features, target = split_features_target(load_raw_data())
    reference, current = train_test_split(
        features,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    result = compute_drift(reference, current)
    return write_report(result, OUTPUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_baseline_drift()
