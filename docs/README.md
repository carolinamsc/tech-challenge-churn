# Documentação

Este diretório reúne a documentação técnica, metodológica e operacional da solução.

## Modelo e dados

- [Model Card](MODEL_CARD.md) — objetivo, uso pretendido, dados, métricas, threshold, limitações, vieses e cenários de falha.
- [EDA Findings](EDA_FINDINGS.md) — qualidade dos dados, tratamento de `TotalCharges`, hipóteses e decisões de modelagem.
- [Experimentos](EXPERIMENTS.md) — protocolos de holdout e 5-fold CV, comparação dos candidatos, escolha do modelo e análise de threshold.

## Negócio e operação

- [Métrica de negócio](BUSINESS_METRIC.md) — taxa de intervenção, recall e trade-off do threshold para priorização de retenção.
- [Plano de monitoramento](MONITORING.md) — drift, distribuição das previsões, performance retrospectiva, gatilhos de revalidação e playbook.

## Engenharia

- [Arquitetura](ARCHITECTURE.md) — camadas da aplicação, API, Streamlit, Docker, deploy e decisões técnicas.

## Resultados reproduzíveis

Os resultados dos experimentos ficam versionados em `../reports/`:

- `model_results.csv` — comparação dos candidatos no holdout;
- `cross_validation.csv` — métricas médias e desvios dos 5 folds;
- `threshold_analysis.csv` — avaliação dos pontos de corte de 0,20 a 0,70.

Os arquivos são produzidos pelos scripts em `src/models/` e validados pelo GitHub Actions. O ambiente oficial é definido no `pyproject.toml` para reduzir diferenças entre execuções.
