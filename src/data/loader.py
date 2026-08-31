"""Data loading and target/feature separation utilities."""

from pathlib import Path

import pandas as pd

RAW_DATA = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_raw_data(path: str | Path = RAW_DATA) -> pd.DataFrame:
    """Load the Telco Customer Churn CSV and normalize TotalCharges."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate Churn from features and remove the customer identifier."""
    if "Churn" not in df.columns:
        raise ValueError("DataFrame must contain the 'Churn' target column")

    y = df["Churn"].copy()
    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Churn contains unsupported or missing target values")
    y = y.astype(int)

    X = df.drop(columns=["Churn", "customerID"], errors="ignore").copy()
    return X, y
