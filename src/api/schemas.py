"""Pydantic request/response schemas for the churn API."""

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Customer attributes expected by the trained pipeline."""

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: float = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Churn prediction response."""

    churn_probability: float
    churn_prediction: int
    risk_level: str
    threshold: float
