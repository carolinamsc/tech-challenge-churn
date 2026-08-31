"""FastAPI application for churn prediction."""

from fastapi import FastAPI, HTTPException

from src.api.schemas import CustomerInput, PredictionResponse
from src.models.predict import load_model, predict_churn

app = FastAPI(
    title="Churn Prediction API",
    version="0.1.0",
    description="API for customer churn probability and classification.",
)

_model = None


def get_model():
    """Load the persisted model lazily when prediction is requested."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health without requiring a model artifact."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> PredictionResponse:
    """Predict churn for one customer."""
    try:
        result = predict_churn(get_model(), customer.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact is not available") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(**result)
