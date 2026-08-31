"""Streamlit front-end for the churn prediction API or local model."""

import os

import requests
import streamlit as st

from src.data.download import download_dataset
from src.data.loader import load_raw_data, split_features_target
from src.models.model_selection import build_models
from src.models.predict import predict_churn
from src.utils.config import DEFAULT_THRESHOLD

API_URL = os.getenv("API_URL")

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource
def get_local_model():
    """Train and cache the selected model for standalone Streamlit hosting."""
    data_path = download_dataset()
    df = load_raw_data(data_path)
    X, y = split_features_target(df)
    model = build_models()["logistic_regression"]
    model.fit(X, y)
    return model


def local_predict(customer: dict) -> dict:
    """Predict directly when no remote API URL is configured."""
    return predict_churn(get_local_model(), customer, threshold=DEFAULT_THRESHOLD)


st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(49,51,63,.55), rgba(49,51,63,.25));
    }
    .hero h1 {margin-bottom: .2rem;}
    .muted {color: #9aa0ad; font-size: .95rem;}
    .result-card {
        padding: 1.35rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        margin-top: 1rem;
    }
    .probability {font-size: 2.4rem; font-weight: 700; margin: .15rem 0 .5rem 0;}
    .small-note {color: #9aa0ad; font-size: .85rem; margin-top: .6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📉 Churn Prediction</h1>
        <div class="muted">Avaliação individual do risco de cancelamento de clientes</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("customer_form"):
    st.subheader("Dados do cliente")
    st.caption("Preencha o perfil e calcule a probabilidade estimada de churn.")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0.0, value=12.0)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total = st.number_input("Total Charges", min_value=0.0, value=840.0)

    submitted = st.form_submit_button(
        "Calcular risco",
        type="primary",
        use_container_width=True,
    )

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    try:
        if API_URL:
            response = requests.post(f"{API_URL}/predict", json=customer, timeout=10)
            response.raise_for_status()
            result = response.json()
        else:
            result = local_predict(customer)

        probability = result["churn_probability"]
        risk_level = result["risk_level"]
        prediction = result["churn_prediction"]
        threshold = result["threshold"]

        st.subheader("Resultado")
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        metric_col, status_col = st.columns([1, 2])
        with metric_col:
            st.caption("Probabilidade estimada")
            st.markdown(
                f'<div class="probability">{probability:.1%}</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(probability, 0.0), 1.0))
        with status_col:
            if risk_level == "high":
                st.error("🔴 Alto risco — cliente prioritário para retenção")
            elif risk_level == "medium":
                st.warning("🟠 Risco médio — acompanhar")
            else:
                st.success("🟢 Baixo risco")

            st.write(
                f"**Classificação:** {'Churn' if prediction == 1 else 'Não churn'}  "
                f"\n\n**Threshold de decisão:** {threshold:.2f}"
            )

        st.markdown(
            '<div class="small-note">A probabilidade é uma estimativa do modelo, '
            'não uma certeza sobre o comportamento do cliente.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Como interpretar este resultado"):
            st.write(
                "O modelo foi treinado para priorizar clientes com maior probabilidade "
                "de churn. O threshold de 0,55 foi definido após análise de trade-off "
                "entre precisão e recall; acima dele, o cliente é sinalizado como churn."
            )

    except (requests.RequestException, FileNotFoundError, ValueError) as exc:
        st.error(f"Não foi possível calcular o risco: {exc}")
