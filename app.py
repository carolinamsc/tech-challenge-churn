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

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.6rem 1.7rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(67,72,90,.58), rgba(36,39,49,.35));
    }
    .hero-title {font-size: 2.2rem; font-weight: 750; line-height: 1.1; margin: 0;}
    .hero-subtitle {color: #aeb4c0; margin-top: .45rem; font-size: 1rem;}
    .section-note {color: #9aa0ad; font-size: .9rem; margin-bottom: .7rem;}
    .result-card {
        padding: 1.5rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-top: 1.2rem;
        background: rgba(38, 40, 50, .35);
    }
    .probability {font-size: 2.8rem; font-weight: 800; margin: .05rem 0 .45rem 0;}
    .decision-pill {
        display: inline-block;
        padding: .35rem .7rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.3);
        font-size: .85rem;
        margin-bottom: .55rem;
    }
    .small-note {color: #9aa0ad; font-size: .84rem; margin-top: .7rem;}
    .footer-note {color: #7f8694; text-align: center; margin-top: 2rem; font-size: .82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📉 Churn Prediction</div>
        <div class="hero-subtitle">Avalie o risco de cancelamento e priorize clientes para retenção.</div>
    </div>
    """,
    unsafe_allow_html=True,
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

with st.form("customer_form"):
    st.subheader("Dados do cliente")
    st.caption("Preencha os campos abaixo. Eles representam as principais características usadas pelo modelo.")

    st.markdown("**👤 Perfil**")
    profile_1, profile_2, profile_3 = st.columns(3)
    with profile_1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
    with profile_2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    with profile_3:
        tenure = st.number_input("Tenure (months)", min_value=0.0, value=12.0)
        phone = st.selectbox("Phone Service", ["Yes", "No"])

    st.divider()
    st.markdown("**🌐 Serviços**")
    services_1, services_2, services_3 = st.columns(3)
    with services_1:
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    with services_2:
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    with services_3:
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.divider()
    st.markdown("**💳 Contrato e cobrança**")
    billing_1, billing_2, billing_3 = st.columns(3)
    with billing_1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with billing_2:
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
    with billing_3:
        monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total = st.number_input("Total Charges", min_value=0.0, value=840.0)

    st.markdown('<div class="section-note">O threshold atual é 0,55: acima dele, o cliente é sinalizado para retenção.</div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("Calcular risco", type="primary", use_container_width=True)

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
        decision = "Churn" if prediction == 1 else "Não churn"

        st.subheader("Resultado")
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        metric_col, decision_col = st.columns([1, 2])

        with metric_col:
            st.caption("Probabilidade estimada")
            st.markdown(f'<div class="probability">{probability:.1%}</div>', unsafe_allow_html=True)
            st.progress(min(max(probability, 0.0), 1.0))

        with decision_col:
            if risk_level == "high":
                st.error("🔴 Alto risco — cliente prioritário para retenção")
            elif risk_level == "medium":
                st.warning("🟠 Risco médio — acompanhar")
            else:
                st.success("🟢 Baixo risco — menor prioridade de intervenção")

            st.markdown(f'<div class="decision-pill">Decisão: <b>{decision}</b></div>', unsafe_allow_html=True)
            st.write(f"**Threshold:** {threshold:.2f}")

        st.markdown(
            '<div class="small-note">A probabilidade é uma estimativa do modelo, não uma certeza sobre o comportamento do cliente.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("ℹ️ Como interpretar"):
            st.write(
                "A probabilidade estima o risco de churn para o perfil informado. "
                "O threshold de 0,55 foi escolhido após analisar o trade-off entre precisão e recall. "
                "O objetivo é apoiar a priorização de ações de retenção, não substituir a decisão da equipe."
            )

    except (requests.RequestException, FileNotFoundError, ValueError) as exc:
        st.error(f"Não foi possível calcular o risco: {exc}")

st.markdown(
    "<div class='footer-note'>Modelo: Logistic Regression · Validação cruzada 5-fold · Threshold: 0,55</div>",
    unsafe_allow_html=True,
)