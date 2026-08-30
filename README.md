# Churn Prediction — Tech Challenge FIAP Fase 1

Projeto de Machine Learning para previsão de churn, desenvolvido com foco nos requisitos da Fase 1 do Tech Challenge e em boas práticas de engenharia de ML.

## Objetivo

Construir, avaliar e disponibilizar um modelo capaz de identificar clientes com maior probabilidade de churn, apoiando estratégias de retenção.

## Abordagem

1. Análise exploratória e entendimento do problema.
2. Pré-processamento reproduzível com `scikit-learn Pipeline`.
3. Comparação de **Regressão Logística**, **Random Forest** e **MLPClassifier**.
4. Validação cruzada estratificada e avaliação em conjunto de teste.
5. Seleção do modelo campeão considerando métricas técnicas e impacto de negócio.
6. Persistência do pipeline completo.
7. API REST com FastAPI (`/health` e `/predict`).
8. Testes automatizados, documentação e Model Card.

## Estrutura

```text
├── data/
│   ├── raw/              # dataset original (não versionado)
│   └── processed/        # dados derivados (não versionado)
├── notebooks/
│   └── 01_eda_baseline.ipynb
├── src/
│   ├── api/              # API FastAPI e schemas
│   ├── data/             # carregamento e validação dos dados
│   ├── features/         # preprocessing e feature engineering
│   ├── models/           # treino, avaliação e inferência
│   └── utils/             # configuração e reprodutibilidade
├── models/               # artefatos treinados (não versionados)
├── tests/                # testes automatizados
├── docs/                 # Model Card e documentação técnica
├── pyproject.toml
├── Makefile
└── Dockerfile
```

## Status

🚧 Em desenvolvimento.

## Reprodutibilidade

As decisões de modelagem, seeds, divisão dos dados, métricas e versões das dependências serão registradas no projeto para permitir a reprodução dos resultados.
