# Model Card — Churn Prediction

## 1. Visão geral

Modelo de classificação binária para estimar a propensão de churn de clientes de telecomunicações. O objetivo é apoiar a priorização de ações de retenção; a predição não deve ser interpretada como certeza de cancelamento.

## 2. Dados

Dataset público Telco Customer Churn, com variáveis demográficas, serviços contratados, contrato, cobrança e tempo de relacionamento. O alvo é `Churn`.

## 3. Modelos avaliados

- Regressão Logística — baseline e candidata final
- Random Forest
- MLPClassifier

Todos utilizam pipeline de pré-processamento do Scikit-Learn para reduzir risco de inconsistência entre treino e inferência.

## 4. Performance

Na avaliação inicial em conjunto de teste:

| Modelo | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8413 | 0.6136 | 0.5043 | 0.7834 | 0.7381 |
| MLPClassifier | 0.8391 | 0.5514 | 0.6604 | 0.4733 | 0.7956 |
| Random Forest | 0.8193 | 0.5845 | 0.5423 | 0.6337 | 0.7608 |

A Regressão Logística foi selecionada por apresentar o maior ROC-AUC e maior recall entre os candidatos, características relevantes para uma estratégia de retenção em que deixar um potencial churn passar pode ser mais custoso que abordar um falso positivo.

## 5. Threshold

A análise de thresholds indicou `0.55` como ponto de corte com melhor F1 na avaliação realizada. Esse valor é uma política de priorização e pode ser recalibrado conforme custos reais de retenção e capacidade operacional.

## 6. Limitações

- O dataset é histórico e representa um contexto específico de telecomunicações.
- As métricas podem mudar em dados de produção.
- Probabilidade estimada não equivale a causalidade.
- O modelo não mede se uma intervenção de retenção efetivamente evitará o churn.
- O threshold atual não incorpora custos financeiros reais, pois esses valores não fazem parte do dataset.

## 7. Possíveis vieses e riscos

Variáveis demográficas e de perfil podem produzir diferenças de desempenho entre grupos. Antes de uso em produção, devem ser avaliadas métricas de desempenho por subgrupo, calibração e impactos de decisões automatizadas. A saída deve apoiar, e não substituir, uma decisão de negócio responsável.

## 8. Uso pretendido

Priorização de clientes para análise ou campanhas de retenção. Não recomendado para decisões irreversíveis, exclusão de serviços ou outras decisões de alto impacto sem revisão humana.

## 9. Reprodutibilidade

Seed principal: `42`. O pipeline de CI executa lint, testes, validação dos modelos e análise de threshold.
