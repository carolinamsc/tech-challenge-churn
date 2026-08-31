import pandas as pd

from src.features.preprocessing import build_preprocessor


def test_preprocessor_handles_missing_total_charges():
    X = pd.DataFrame(
        {
            "tenure": [1, 12],
            "MonthlyCharges": [20.0, 50.0],
            "TotalCharges": [None, 600.0],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
        }
    )

    transformed = build_preprocessor().fit_transform(X)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 3
