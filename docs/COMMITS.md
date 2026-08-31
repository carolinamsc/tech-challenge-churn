# Trilha de desenvolvimento

O histórico deste repositório é incremental e reflete o desenvolvimento real, incluindo
correções e revisões. Os commits se organizam nas quatro etapas do desafio:

| Etapa | Entrega | Onde está no repositório |
|---|---|---|
| 1 — Entendimento e EDA | ML Canvas, EDA executada, achados e gráficos | `docs/ML_CANVAS.md`, `notebooks/01_eda_baseline.ipynb`, `docs/EDA_FINDINGS.md` |
| 2 — Modelagem e comparação | baseline, ensemble e MLPClassifier, CV 5-fold, holdout, threshold | `src/models/`, `reports/*.csv`, `docs/EXPERIMENTS.md` |
| 3 — API de inferência | FastAPI com `/health` e `/predict`, Docker, Streamlit | `src/api/`, `app.py`, `pages/`, `Dockerfile` |
| 4 — Documentação e operação | Model Card, métrica de negócio, monitoramento de drift, CI | `docs/MODEL_CARD.md`, `docs/BUSINESS_METRIC.md`, `docs/MONITORING.md`, `.github/workflows/ci.yml` |

As mensagens seguem Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `ci:`) a partir
da consolidação do projeto.
