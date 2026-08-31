"""Streamlit page showing the versioned input drift report."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.monitoring.drift import ALERT_THRESHOLD, WARNING_THRESHOLD

REPORT = Path("reports/drift_baseline.csv")
STATUS_LABEL = {
    "stable": "estável",
    "warning": "atenção",
    "alert": "alerta",
}

st.set_page_config(page_title="Monitoramento", layout="wide")

st.title("Monitoramento de drift")
st.caption(
    "Population Stability Index (PSI) por variável de entrada, regenerado a cada "
    "execução do CI e versionado em reports/drift_baseline.csv."
)

if not REPORT.exists():
    st.warning("Relatório de drift não encontrado. Rode `python -m src.monitoring.drift`.")
    st.stop()

data = pd.read_csv(REPORT)
counts = data["status"].value_counts()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Variáveis monitoradas", len(data))
col2.metric("Estáveis", int(counts.get("stable", 0)))
col3.metric("Em atenção", int(counts.get("warning", 0)))
col4.metric("Em alerta", int(counts.get("alert", 0)))

st.subheader("PSI por variável")
st.bar_chart(data.set_index("feature")["psi"], horizontal=True)

st.subheader("Detalhamento")
st.dataframe(
    data.assign(status=data["status"].map(STATUS_LABEL)).rename(
        columns={"feature": "variável", "psi": "PSI"}
    ),
    hide_index=True,
    width="stretch",
)

st.markdown(
    f"""
**Como ler:** PSI abaixo de {WARNING_THRESHOLD:.2f} indica distribuição estável;
entre {WARNING_THRESHOLD:.2f} e {ALERT_THRESHOLD:.2f} exige investigação;
a partir de {ALERT_THRESHOLD:.2f} dispara revalidação do modelo.

O baseline compara as partições de treino e holdout do dataset público — é a
referência contra a qual lotes futuros devem ser medidos. Não há dados de
produção neste projeto, portanto não existe monitoramento de performance em
tempo real: isso dependeria do rótulo de churn observado meses depois.
"""
)
