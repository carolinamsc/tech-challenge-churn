# Model Card — Churn Prediction

## 1. Visão geral

Modelo de classificação binária para estimar a propensão de churn de clientes de telecomunicações. O objetivo é apoiar a priorização de ações de retenção; a predição não deve ser interpretada como certeza de cancelamento.

Os candidatos avaliados foram Regressão Logística, Random Forest e `MLPClassifier` do Scikit-Learn. Todos compartilham o mesmo pipeline de pré-processamento e são comparados nos mesmos protocolos de holdout e validação cruzada estratificada.

## 2. Uso pretendido

O modelo estima o risco de churn para ajudar uma equipe de retenção a priorizar contatos. A decisão operacional é baseada na probabilidade estimada e no threshold definido para o experimento.

A saída não deve ser usada isoladamente para negar serviço, alterar condições contratuais ou tomar decisões de alto impacto sem revisão humana.

## 3. Dados

O projeto usa o dataset público IBM Telco Customer Churn, com 7.043 registros e 21 colunas. `Churn` é o alvo.

`customerID` é tratado como identificador e removido das features. `TotalCharges` é convertido para numérico e os valores ausentes são tratados dentro do pipeline de pré-processamento.

## 4. Treinamento e avaliação

- Seed principal: `42`
- Holdout: `20%`, estratificado
- Validação cruzada: `5-fold`, estratificada
- Métricas: ROC-AUC, F1, precision, recall e accuracy
- O conjunto de teste é mantido fora do processo de seleção por CV e usado para avaliação final.
- Ambiente de referência dos experimentos e do CI: Python 3.11, scikit-learn 1.9.0, numpy 2.4.6, scipy 1.17.1 e pandas 2.3.3.
- As versões das bibliotecas que influenciam os resultados estão fixadas em `pyproject.toml`.

### Validação cruzada 5-fold

| Modelo | ROC-AUC (média ± dp) | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.8449 ± 0.0135** | **0.6258** | 0.5134 | **0.8015** |
| MLPClassifier | 0.8390 ± 0.0145 | 0.5462 | **0.6718** | 0.4676 |
| Random Forest | 0.8227 ± 0.0106 | 0.6040 | 0.5658 | 0.6479 |

### Holdout 20%

| Modelo | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.8413** | **0.6136** | 0.5043 | **0.7834** | 0.7381 |
| MLPClassifier | 0.8391 | 0.5514 | **0.6604** | 0.4733 | **0.7956** |
| Random Forest | 0.8193 | 0.5845 | 0.5423 | 0.6337 | 0.7608 |

A Regressão Logística foi selecionada por apresentar o melhor ROC-AUC, F1 e recall nos protocolos utilizados. Para retenção, a capacidade de capturar clientes que realmente cancelariam é especialmente relevante.

### Sensibilidade do Random Forest ao ambiente

Durante a validação de reprodutibilidade, verificou-se que o Random Forest apresentou diferenças de métricas entre ambientes com versões diferentes de scikit-learn/numpy, enquanto Regressão Logística e MLPClassifier reproduziram os resultados observados. Por isso, os resultados oficiais do projeto são definidos pelo ambiente pinado no repositório e validado pelo CI. Essa observação reforça a importância de fixar dependências ao reproduzir os experimentos; não significa que exista falha no algoritmo Random Forest.

## 5. Threshold e decisão operacional

O threshold final adotado é `0.55`.

No holdout, esse ponto apresentou:

- F1: **0.6176**
- Recall: **75,1%**
- Precision: **52,4%**
- Taxa de intervenção: **38,0%**
- Clientes acionados: **536 de 1.409**

O threshold foi escolhido a partir do trade-off entre precisão e recall. Ele representa uma política de priorização, não uma propriedade fixa do modelo, e pode ser recalibrado conforme a capacidade operacional e os custos reais de retenção.

## 6. Limitações

- O dataset representa uma população histórica específica de telecomunicações e pode não generalizar para outros produtos ou mercados.
- A probabilidade prevista é uma associação estatística, não uma explicação causal do churn.
- Fatores externos, como mudança de preço, concorrência ou eventos pessoais, não são diretamente observáveis nas features.
- O modelo não distingue causas de churn voluntárias de eventos como inadimplência quando essas causas não estão representadas nas variáveis.
- O desempenho pode degradar com mudanças na população ou nas políticas comerciais.
- Métricas de modelos como Random Forest podem variar entre versões de bibliotecas; por isso, o ambiente de execução deve ser respeitado para reproduzir os números oficiais.

## 7. Vieses identificados e riscos

Variáveis de perfil podem reproduzir diferenças históricas de churn entre grupos. Em particular, a taxa observada de churn é maior entre clientes `SeniorCitizen` e entre contratos mensais; o modelo pode, portanto, sinalizar esses grupos com maior frequência.

Antes de uso em produção, recomenda-se avaliar performance, calibração e taxa de decisão por subgrupo. A saída deve apoiar uma decisão de negócio responsável, e não automatizá-la sem supervisão.

## 8. Cenários de falha

- Clientes com `tenure` muito baixo podem apresentar `TotalCharges` ausente no dataset original; a imputação por mediana pode subestimar o risco desse grupo.
- Mudanças relevantes de preço, concorrência ou produto podem gerar data drift e reduzir a validade das previsões.
- O modelo não possui informação suficiente para explicar churn provocado por fatores externos não observados.
- A degradação de distribuição, calibração ou performance pode tornar o threshold atual inadequado.
- A execução com dependências diferentes do ambiente de referência pode alterar os resultados de alguns candidatos, especialmente Random Forest.

## 9. Reprodutibilidade

A seed principal e os parâmetros de split são centralizados no projeto. O pipeline de pré-processamento é serializável e aplicado de forma consistente entre treino e inferência.

O ambiente oficial de avaliação é Python 3.11 com scikit-learn 1.9.0, numpy 2.4.6, scipy 1.17.1 e pandas 2.3.3. As versões críticas estão fixadas no `pyproject.toml`.

O GitHub Actions executa lint, testes, validação cruzada, comparação de modelos, análise de threshold e geração do artefato final. Os reports são regenerados no próprio CI e o job falha quando divergem dos arquivos versionados.
