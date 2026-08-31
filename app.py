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
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.35rem 1.5rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        margin-bottom: 1.25rem;
        background: linear-gradient(135deg, rgba(49,51,63,.58), rgba(49,51,63,.22));
    }
    .hero-title {display:flex; align-items:center; gap:.7rem; margin:0; font-size:2.1rem; font-weight:750;}
    .hero-subtitle {margin:.35rem 0 0 2.55rem; color:#a8adb9; font-size:1rem;}
    .icon-box {
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 9px;
        border: 1px solid rgba(128,128,128,.28);
        background: rgba(255,255,255,.04);
        color: #d6d9e2;
    }
    .section-title {display:flex; align-items:center; gap:.6rem; margin:.25rem 0 1rem; font-size:1.15rem; font-weight:700;}
    .section-rule {height:1px; background:rgba(128,128,128,.22); margin:1.25rem 0 1.4rem;}
    .result-label {color:#9ea4b2; font-size:.92rem;}
    .probability {font-size:2.7rem; font-weight:760; line-height:1.05; margin:.2rem 0 .6rem;}
    .status-chip {display:inline-block; padding:.5rem .75rem; border-radius:999px; border:1px solid rgba(128,128,128,.28); font-weight:650;}
    .status-high {background:rgba(210,55,55,.14); color:#ff8c8c;}
    .status-medium {background:rgba(190,160,25,.16); color:#eadb75;}
    .status-low {background:rgba(35,155,95,.14); color:#79d6a5;}
    .small-note {color:#969cab; font-size:.84rem; margin-top:.65rem;}
    .footer-note {text-align:center; color:#7f8592; font-size:.82rem; margin-top:1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">
        <span class="icon-box">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 17 9 11 13 15 21 6"/><polyline points="14 6 21 6 21 13"/>
          </svg>
        </span>
        <span>Churn Prediction</span>
      </div>
      <div class="hero-subtitle">Avalie o risco de cancelamento e priorize clientes para retenção.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("customer_form"):
    st.subheader("Dados do cliente")
    st.caption("Preencha os campos abaixo para estimar o risco de churn.")

    st.markdown(
        """
        <div class="section-title">
          <span class="icon-box"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg></span>
          <span>Perfil</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
    with col2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    with col3:
        tenure = st.number_input("Tenure (months)", min_value=0.0, value=12.0)
        phone = st.selectbox("Phone Service", ["Yes", "No"])

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
          <span class="icon-box"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a14 14 0 0 1 0 18"/><path d="M12 3a14 14 0 0 0 0 18"/></svg></span>
          <span>Serviços</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    with col2:
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    with col3:
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
          <span class="icon-box"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18"/><path d="M7 15h2"/></svg></span>
          <span>Contrato e cobrança</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
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
    with col3:
        monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total = st.number_input("Total Charges", min_value=0.0, value=840.0)

    st.caption("Threshold atual: 0,55. Acima dele, o cliente é sinalizado para retenção.")
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
            data_path = download_dataset()
            df = load_raw_data(data_path)
            X, y = split_features_target(df)
            model = build_models()["logistic_regression"]
            model.fit(X, y)
            result = predict_churn(model, customer, threshold=DEFAULT_THRESHOLD)

        probability = result["churn_probability"]
        risk_level = result["risk_level"]
        prediction = result["churn_prediction"]
        threshold = result["threshold"]

        st.markdown('<div class="section-title" style="margin-top:1.6rem;"><span class="icon-box"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg></span><span>Resultado</span></div>', unsafe_allow_html=True)
        with st.container(border=True):
            metric_col, status_col = st.columns([1, 2])
            with metric_col:
                st.markdown('<div class="result-label">Probabilidade estimada</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="probability">{probability:.1%}</div>', unsafe_allow_html=True)
                st.progress(min(max(probability, 0.0), 1.0))
            with status_col:
                if risk_level == "high":
                    st.markdown('<span class="status-chip status-high">Alto risco · priorizar retenção</span>', unsafe_allow_html=True)
                elif risk_level == "medium":
                    st.markdown('<span class="status-chip status-medium">Risco médio · acompanhar</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-chip status-low">Baixo risco</span>', unsafe_allow_html=True)

                st.write(
                    f"**Decisão:** {'Churn' if prediction == 1 else 'Não churn'}  \n\n"
                    f"**Threshold:** {threshold:.2f}"
                )
                st.markdown('<div class="small-note">A classificação transforma a probabilidade em uma recomendação de priorização. Não é uma certeza sobre o comportamento do cliente.</div>', unsafe_allow_html=True)

        with st.expander("Como interpretar"):
            st.write(
                "O modelo foi selecionado após comparar Regressão Logística, Random Forest e MLP. "
                "A validação cruzada estratificada confirmou a Regressão Logística como melhor candidata "
                "em ROC-AUC e F1. O threshold de 0,55 foi definido após analisar o trade-off entre precisão "
                "e recall, com foco em priorização de retenção."
            )

        st.markdown(
            '<div class="footer-note">Modelo: Logistic Regression · Validação cruzada: 5-fold · Threshold: 0,55</div>',
            unsafe_allow_html=True,
        )

    except (requests.RequestException, FileNotFoundError, ValueError) as exc:
        st.error(f"Não foi possível calcular o risco: {exc}")
