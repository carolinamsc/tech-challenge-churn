# Arquitetura da solução

## Visão geral

A solução separa ingestão e preparação dos dados, modelagem, serviço de inferência e interface de demonstração. O pipeline de pré-processamento é compartilhado entre os candidatos para manter a comparação consistente.

```text
                         Dataset público
                               │
                               ▼
                        src/data/download
                               │
                               ▼
                         src/data/loader
                               │
                               ▼
                      features + target
                               │
                               ▼
                    Scikit-Learn Pipeline
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
      Logistic Regression  Random Forest   MLPClassifier
              │                │                │
              └────────────────┼────────────────┘
                               │
                     Holdout + 5-fold CV
                               │
                               ▼
                       Modelo selecionado
                               │
                     threshold = 0,55
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
              FastAPI                    Streamlit
           /health /predict              interface
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                         risco de churn
```

## Camadas

### Dados — `src/data`

`download.py` centraliza a origem do dataset e `loader.py` organiza a leitura e a separação entre features e target. A mesma rotina de download é usada no CI e no Docker para evitar múltiplas fontes de dados no projeto.

### Features — `src/features`

O pré-processamento é implementado como pipeline do Scikit-Learn. Variáveis numéricas e categóricas seguem transformações próprias dentro do pipeline, garantindo que as transformações aprendidas no treino também sejam aplicadas na inferência.

### Modelos — `src/models`

A camada contém:

- definição dos candidatos;
- comparação no holdout;
- validação cruzada estratificada 5-fold;
- avaliação das métricas;
- análise de threshold;
- treinamento do artefato final.

Os três candidatos usam o mesmo protocolo de avaliação. O modelo final é a Regressão Logística.

### API — `src/api`

A FastAPI expõe os endpoints de inferência:

- `GET /health` para health check;
- `POST /predict` para gerar probabilidade, classificação, nível de risco e threshold.

O Swagger/OpenAPI fica disponível em `/docs`.

### Interface — `app.py`

O Streamlit permite simular um cliente preenchendo as variáveis de entrada. Com `API_URL`, a interface consulta a FastAPI. Sem essa variável, a aplicação usa o caminho standalone, adequado ao deploy da demo no Streamlit Community Cloud.

No modo standalone, o modelo é mantido em cache para evitar novo treinamento a cada submissão.

## Modos de execução

### Completo, com Docker Compose

```bash
docker compose up --build
```

Esse modo demonstra a separação entre front e serviço de inferência.

### API e front separadamente

```bash
uvicorn src.api.main:app --reload
streamlit run app.py
```

### Demo pública

A demo publicada no Streamlit Community Cloud executa o front em modo standalone, sem depender de uma API externa pública.

## Reprodutibilidade

O ambiente de referência para os experimentos é:

- Python 3.11
- scikit-learn 1.9.0
- numpy 2.4.6
- scipy 1.17.1
- pandas 2.3.3
- `random_state = 42`

As versões críticas estão fixadas no `pyproject.toml`. Os relatórios em `reports/` são gerados pelos scripts de experimento e verificados pelo CI.

## Segurança e uso responsável

O modelo deve ser usado como apoio à priorização de retenção. A saída é probabilística e não estabelece causalidade. Decisões de alto impacto devem permanecer sob revisão humana.

## Trade-offs principais

**Por que Logistic Regression?** É o candidato com melhor ROC-AUC, F1 e recall nos protocolos avaliados. Para o problema de retenção, o recall tem relevância operacional porque mede quantos churners reais entram na priorização.

**Por que threshold 0,55?** No holdout, esse ponto maximiza o F1 entre os thresholds avaliados e implica taxa de intervenção de 38,0%, contra 41,2% em 0,50.

**Por que Streamlit standalone na cloud?** Simplifica a demonstração pública e reduz a infraestrutura necessária, preservando a arquitetura Streamlit → FastAPI para execução local.
