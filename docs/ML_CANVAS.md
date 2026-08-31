# ML Canvas — previsão de churn

## Proposta de valor

Antecipar o cancelamento de clientes de telecom para que a operação de retenção
atue antes da perda, priorizando quem tem maior risco em vez de tratar a base inteira.

## Decisão suportada

Quais clientes entram na fila de retenção do mês. O modelo não decide sozinho:
ele ordena e sinaliza; a ação (contato, oferta, desconto) é do time de retenção.

## Stakeholders

- **Retenção / CRM** — consome a lista priorizada e executa as ações.
- **Operações** — define quantos clientes cabem na capacidade mensal de contato.
- **Produto / Comercial** — usa os padrões de churn para rever contratos e planos.
- **Ciência de dados** — mantém o pipeline, o threshold e o monitoramento.

## Fonte de dados

IBM Telco Customer Churn (7.043 clientes, 21 colunas), com `Churn` como alvo.
`customerID` é descartado; `TotalCharges` é convertido para numérico e os 11 registros
vazios correspondem a clientes com `tenure = 0`. Nenhuma variável posterior ao momento
da decisão é utilizada.

## Predição

Probabilidade de cancelamento por cliente, transformada em decisão binária pelo
threshold de 0,55.

## Métricas

- **Modelo:** ROC-AUC e F1 (validação cruzada estratificada 5-fold e holdout de 20%).
- **Negócio:** recall (quantos churns são capturados) e taxa de intervenção
  (quantos clientes precisam ser acionados). No threshold de 0,55: recall de 75,1%
  acionando 38,0% da base.

## Restrições e riscos

- Base pública e estática: não há dados de produção nem rótulo em tempo real —
  o churn observado só existe meses depois do contato.
- Classe minoritária (26,5% de churn) tratada com `class_weight="balanced"`.
- Variáveis como `gender` e `SeniorCitizen` exigem cuidado: o Model Card registra
  a análise de viés e a recomendação de avaliação por subgrupo.
- Deriva de distribuição é acompanhada por PSI (`src/monitoring/drift.py`);
  o plano de resposta está em `docs/MONITORING.md`.

## Ciclo de vida

Treino offline reprodutível (seed 42, versões fixadas), artefato `.joblib` servido pela
API FastAPI, relatórios regenerados e conferidos pelo CI a cada push, e revalidação
disparada pelos gatilhos descritos no plano de monitoramento.
