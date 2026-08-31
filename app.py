"""Streamlit front-end for the churn prediction API."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")
st.title("📉 Churn Prediction")
st.caption("Avaliação individual do risco de cancelamento de clientes")

with st.form("customer_form"):
    st.subheader("Dados do cliente")
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
        response = requests.post(f"{API_URL}/predict", json=customer, timeout=10)
        response.raise_for_status()
        result = response.json()
        probability = result["churn_probability"]
        st.subheader("Resultado")
        st.metric("Probabilidade de churn", f"{probability:.1%}")
        if result["risk_level"] == "high":
            st.error("🔴 Alto risco — cliente prioritário para retenção")
        elif result["risk_level"] == "medium":
            st.warning("🟠 Risco médio — acompanhar")
        else:
            st.success("🟢 Baixo risco")
        st.write(f"Classificação: **{result['churn_prediction']}** · Threshold: **{result['threshold']:.2f}**")
    except requests.RequestException as exc:
        st.error(f"Não foi possível consultar a API: {exc}")
