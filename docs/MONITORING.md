# Plano de monitoramento — Churn Prediction

## Objetivo

Verificar se a distribuição dos dados, as previsões e a capacidade operacional permanecem próximas do comportamento observado na validação.

## O que monitorar

Acompanhar especialmente `Contract`, `tenure`, `MonthlyCharges` e `PaymentMethod`, além da distribuição das probabilidades preditas, percentual acima do threshold `0,55`, taxa de intervenção esperada de aproximadamente **38%**, volume de chamadas ao `/predict`, latência e erros da API.

Quando novos rótulos de churn estiverem disponíveis, recalcular ROC-AUC, F1, precision e recall.

## Detecção de drift — implementação executável

O drift de entrada está implementado em `src/monitoring/drift.py` usando **Population Stability Index (PSI)**. O baseline pode ser gerado com `python -m src.monitoring.drift`, visualizado com `python -m src.viz.drift_chart` e consultado na página **Monitoramento** do Streamlit em `pages/1_Monitoramento.py`.

| PSI | Status | Leitura |
|---|---|---|
| < 0,10 | `stable` | distribuição equivalente à referência |
| 0,10 – 0,25 | `warning` | desvio moderado, acompanhar |
| ≥ 0,25 | `alert` | desvio relevante, investigar e considerar revalidação |

Variáveis numéricas usam bins por quantis da referência; categóricas usam as categorias observadas na referência. `python -m src.monitoring.drift` compara treino e holdout e grava `reports/drift_baseline.csv`. Como as duas partições vêm da mesma população, o baseline esperado é estável.

Para um lote futuro, o mesmo `compute_drift(referencia, lote_atual)` pode ser aplicado antes do scoring. O CI executa o baseline e verifica a consistência do relatório versionado.

## Gatilhos para revalidação ou retreino

Investigar quando ocorrer: ROC-AUC abaixo de **0,80** na base rotulada mais recente; taxa de intervenção fora de **30% a 45%** de forma persistente; drift relevante nas principais features; ou mudança no perfil da população/estratégia comercial.

## Limitação atual

O drift de entrada é medido por código. A performance depende de rótulos disponíveis apenas depois do período de observação. A aplicação publicada é uma demonstração, não um ambiente operacional com coleta de métricas.

## Playbook

1. Detectar desvio.
2. Verificar se houve mudança planejada.
3. Reexecutar a avaliação no conjunto mais recente.
4. Confirmar degradação e revisar features, threshold ou retreino.
5. Registrar decisão, métricas e versão do modelo.
