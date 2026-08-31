# EDA — findings and decisions

## Dataset

O projeto usa o dataset público **IBM Telco Customer Churn**, com 7.043 registros e 21 colunas. O alvo é `Churn`.

A inspeção inicial encontrou 0 duplicatas e `Churn` em aproximadamente **26,5%** dos registros.

## Qualidade dos dados

- `customerID` é um identificador e não é usado como feature.
- `SeniorCitizen` é um indicador binário.
- `tenure` representa o tempo de relacionamento em meses.
- `MonthlyCharges` é numérica.
- `TotalCharges` é armazenada como texto no CSV original e possui **11 valores em branco**.
- Os 11 registros com `TotalCharges` ausente estão associados a clientes com `tenure = 0`, indicando que o branco não deve ser tratado automaticamente como uma cobrança monetária real igual a zero.

A decisão adotada foi converter `TotalCharges` para numérico e tratar os valores ausentes pela imputação dentro do pipeline de pré-processamento, mantendo o tratamento reproduzível entre treino e inferência.

## Padrões observados

A análise exploratória mostrou diferenças de churn relacionadas principalmente a:

- **Contract:** clientes com contrato `Month-to-month` apresentam churn proporcionalmente maior que clientes com contratos de maior duração.
- **PaymentMethod:** o método de pagamento apresenta diferenças relevantes nas taxas observadas de churn.
- **Tenure:** clientes com pouco tempo de relacionamento apresentam maior propensão observada de churn.

Esses padrões foram tratados como sinais de modelagem e não como relações causais.

## Leakage policy

As regras adotadas antes do treinamento foram:

- nenhum recurso derivado do target;
- `customerID` removido;
- pré-processamento ajustado apenas nos dados de treino por `Pipeline`/`ColumnTransformer`;
- conjunto de teste preservado para avaliação final;
- validação cruzada executada sobre a parte de treino.

## Decisões de modelagem

Foram comparados três candidatos com o mesmo pipeline de pré-processamento:

1. Regressão Logística como baseline;
2. Random Forest;
3. `MLPClassifier` do Scikit-Learn.

A comparação por holdout e validação cruzada mostrou a Regressão Logística como melhor candidata para o objetivo de retenção. Ela apresentou o maior ROC-AUC e F1 nos dois protocolos, além de recall substancialmente maior.

A escolha final não foi baseada em complexidade do modelo, mas em desempenho medido e aderência ao objetivo de capturar potenciais churners.

## Próximos usos

O EDA também orientou a análise de threshold e a definição da **taxa de intervenção** como métrica operacional. O threshold final de `0,55` foi escolhido após comparar precision, recall, F1 e o percentual da carteira que seria acionado.
