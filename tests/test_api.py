import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0,
    "TotalCharges": 960.0,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_invalid_input():
    response = client.post("/predict", json={"tenure": -1})
    assert response.status_code == 422


def test_predict_returns_prediction(monkeypatch):
    class DummyModel:
        def predict_proba(self, frame):
            assert len(frame) == 1
            return np.array([[0.30, 0.70]])

    monkeypatch.setattr("src.api.main.get_model", lambda: DummyModel())
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["churn_probability"] == 0.70
    assert body["churn_prediction"] == 1
    assert body["threshold"] == 0.55
    assert body["risk_level"] == "high"
