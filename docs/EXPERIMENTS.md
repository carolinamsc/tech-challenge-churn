# Experimentos e seleção do modelo

## 1. Pergunta de negócio

O objetivo não é apenas maximizar uma métrica estatística. A solução deve identificar clientes com maior propensão a churn para apoiar a priorização de ações de retenção.

Por isso, o experimento considera tanto desempenho preditivo quanto impacto operacional da decisão de classificação.

## 2. Dados

O projeto utiliza o dataset público IBM Telco Customer Churn, com 7.043 registros e 21 variáveis. `customerID` é tratado como identificador e não entra no modelo. `TotalCharges` é convertido para numérico e seus valores ausentes são tratados no pipeline.

## 3. Pipeline de modelagem

Os três candidatos compartilham o mesmo pipeline de pré-processamento:

```text
Dados brutos
   ↓
Limpeza e separação X/y
   ↓
ColumnTransformer / preprocessing
   ↓
Modelo candidato
   ↓
Probabilidade de churn
   ↓
Threshold
   ↓
Decisão de priorização
```

Candidatos:

- Logistic Regression
- Random Forest
- MLPClassifier (Scikit-Learn)

## 4. Protocolos de avaliação

### Holdout

É utilizado um split estratificado de 20%, com `random_state=42`. O teste fica separado para a avaliação final.

### Validação cruzada

Os três candidatos são avaliados pelo mesmo protocolo de validação cruzada estratificada 5-fold. O pré-processamento permanece dentro do pipeline para evitar vazamento entre folds.

As métricas principais são ROC-AUC, F1, precision e recall. Accuracy também é registrada no holdout.

## 5. Resultado

A Regressão Logística apresentou o melhor resultado nos dois protocolos.

### CV 5-fold

| Modelo | ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0,8449 ± 0,0135** | **0,6258** | 0,5134 | **0,8015** |
| MLPClassifier | 0,8390 ± 0,0145 | 0,5462 | **0,6718** | 0,4676 |
| Random Forest | 0,8227 ± 0,0106 | 0,6040 | 0,5658 | 0,6479 |

### Holdout

| Modelo | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0,8413** | **0,6136** | 0,5043 | **0,7834** | 0,7381 |
| MLPClassifier | 0,8391 | 0,5514 | **0,6604** | 0,4733 | **0,7956** |
| Random Forest | 0,8193 | 0,5845 | 0,5423 | 0,6337 | 0,7608 |

A escolha considera o contexto de retenção: recall mede a parcela dos churners reais que entram na lista de priorização. A Regressão Logística também apresenta o maior F1 e o maior ROC-AUC em ambos os protocolos.

## 6. Threshold

O modelo produz uma probabilidade; o threshold transforma essa probabilidade em uma decisão binária.

No holdout, o threshold `0,55` foi escolhido por maximizar o F1 entre os pontos avaliados. Nesse ponto:

- F1 = 0,6176
- recall = 75,1%
- precision = 52,4%
- intervenção = 38,0%
- 536 de 1.409 clientes seriam acionados

A taxa de intervenção traduz a decisão para capacidade operacional: aumentar a cobertura exige contatar uma parcela maior da carteira.

## 7. Reprodutibilidade

O ambiente oficial dos experimentos é:

- Python 3.11
- scikit-learn 1.9.0
- numpy 2.4.6
- scipy 1.17.1
- pandas 2.3.3
- seed 42

As versões críticas estão fixadas em `pyproject.toml`. Os scripts geram os CSVs em `reports/`, e o CI verifica que os arquivos versionados correspondem à execução no ambiente oficial.

## 8. Artefatos

- `reports/model_results.csv` — comparação no holdout
- `reports/cross_validation.csv` — comparação em 5 folds
- `reports/threshold_analysis.csv` — trade-off de thresholds
- `docs/MODEL_CARD.md` — uso, desempenho, riscos e limitações
- `docs/BUSINESS_METRIC.md` — interpretação operacional do threshold
