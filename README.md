# Churn Prediction — Tech Challenge FIAP Fase 1

Projeto de Machine Learning para previsão de churn, desenvolvido para a Fase 1 do Tech Challenge FIAP. A solução combina análise de dados, comparação de modelos, validação cruzada, otimização de threshold, API REST e interface web para apoiar decisões de retenção.

## Demo

**Aplicação Streamlit:** https://tech-challenge-churn-5qyxdnkzfe9gnagbxcnpks.streamlit.app/

A demo permite informar o perfil de um cliente e consultar a probabilidade estimada de churn, a classificação e o nível de risco. O app publicado roda em modo standalone no Streamlit Cloud; no ambiente local, o projeto também suporta a arquitetura Streamlit → FastAPI.

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
Holdout + validação cruzada 5-fold
  ↓
Threshold analysis
  ↓
Modelo final
  ↓
FastAPI ─────────── Streamlit
```

### Modelos avaliados

- Regressão Logística
- Random Forest
- MLPClassifier

A Regressão Logística foi selecionada como modelo final por apresentar o melhor desempenho de validação cruzada entre os candidatos: ROC-AUC médio de **0,8449**, F1 médio de **0,6258** e recall médio de **0,8015**.

No holdout, o modelo obteve ROC-AUC de **0,8413**, F1 de **0,6136** e recall de **0,7834** com threshold 0,50. Após a análise de pontos de corte, o threshold adotado foi **0,55**, que elevou o F1 para **0,6176** e reduziu a taxa de intervenção para **38,0%**.

O threshold não é tratado como uma constante arbitrária: ele representa a política de priorização de clientes para retenção. A análise também registra precision, recall, F1 e taxa de intervenção para diferentes pontos de corte.

## Aplicação

### Streamlit

Interface para simular um cliente e consultar:

- probabilidade de churn;
- classificação de churn;
- nível de risco;
- threshold utilizado.

Execute localmente:

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

Para executar a aplicação completa com Streamlit + FastAPI:

```bash
docker compose up --build
```

A interface ficará disponível em `http://localhost:8501`.

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
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Qualidade e reprodutibilidade

O projeto possui CI no GitHub Actions executando lint, testes automatizados, validação dos modelos, validação cruzada, análise de threshold e geração dos artefatos. Seeds e parâmetros de divisão/validação são centralizados para favorecer a reprodutibilidade.

O modelo final é treinado em todo o dataset após a etapa de avaliação e pode ser reconstruído sem depender de um arquivo binário versionado previamente no repositório.

## Próximos passos

- adicionar explicabilidade individual das previsões;
- incorporar monitoramento de drift e performance do modelo em produção;
- avaliar calibração de probabilidades caso a aplicação evolua para uso operacional.
