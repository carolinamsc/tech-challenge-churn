# Churn Prediction — Tech Challenge FIAP Fase 1

Projeto de Machine Learning para previsão de churn, desenvolvido para a Fase 1 do Tech Challenge FIAP. A solução combina análise de dados, comparação de modelos, otimização de threshold, API REST e interface web para apoiar decisões de retenção.

## Objetivo

Identificar clientes com maior probabilidade de cancelamento e transformar a previsão em uma regra de priorização para ações de retenção.

## Solução

```text
Dados
  ↓
EDA + validação
  ↓
Preprocessing reproduzível
  ↓
Logistic Regression | Random Forest | MLP
  ↓
Avaliação + threshold analysis
  ↓
Modelo final
  ↓
FastAPI ─────────── Streamlit
```

### Modelos avaliados

- Regressão Logística
- Random Forest
- MLPClassifier

Na avaliação inicial, a Regressão Logística apresentou ROC-AUC de **0,8413**, F1 de **0,6136** e recall de **0,7834**. O threshold foi posteriormente ajustado de 0,50 para **0,55**, pois esse ponto apresentou o melhor F1 na análise realizada.

O threshold não é tratado como uma constante arbitrária: ele representa a política de priorização de clientes para retenção. A análise também registra precision, recall, F1 e taxa de intervenção para diferentes pontos de corte.

## Aplicação

### Streamlit

Interface para simular um cliente e consultar:

- probabilidade de churn;
- classificação de churn;
- nível de risco;
- threshold utilizado.

Execute:

```bash
streamlit run app.py
```

### API

A API FastAPI disponibiliza:

- `GET /health` — health check;
- `POST /predict` — previsão de churn.

Execute:

```bash
uvicorn src.api.main:app --reload
```

A documentação interativa fica disponível em `/docs`.

## Docker

Para construir e executar a aplicação:

```bash
docker build -t churn-prediction .
docker run --rm -p 8501:8501 churn-prediction
```

A interface estará disponível em `http://localhost:8501`.

## Estrutura

```text
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_eda_baseline.ipynb
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── models/
├── reports/
├── tests/
├── docs/
├── app.py
├── Dockerfile
├── Makefile
└── pyproject.toml
```

## Qualidade e reprodutibilidade

O projeto possui CI no GitHub Actions executando lint, testes automatizados, validação dos modelos, análise de threshold e geração dos artefatos. Seeds e parâmetros de divisão/validação são centralizados para favorecer a reprodutibilidade.

## Próximos passos

- adicionar explicabilidade individual das previsões;
- publicar a aplicação em um ambiente cloud;
- incorporar monitoramento de drift e performance do modelo em produção.
