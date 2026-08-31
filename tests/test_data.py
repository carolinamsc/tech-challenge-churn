import pandas as pd
import pytest

from src.data.loader import split_features_target


def test_split_features_target_removes_id_and_target():
    df = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "tenure": [1, 12],
            "MonthlyCharges": [20.0, 50.0],
            "TotalCharges": [20.0, 600.0],
            "Churn": [0, 1],
        }
    )

    X, y = split_features_target(df)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert y.tolist() == [0, 1]


def test_split_features_target_requires_target():
    with pytest.raises(ValueError):
        split_features_target(pd.DataFrame({"tenure": [1]}))
