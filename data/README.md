# Dataset

## Telco Customer Churn

O projeto usa o dataset **Telco Customer Churn**, originalmente disponibilizado pela IBM e distribuído publicamente via Kaggle. O arquivo esperado é `WA_Fn-UseC_-Telco-Customer-Churn.csv`, com 7.043 clientes e 21 colunas, incluindo `Churn` como variável-alvo.

Fonte original:
- Kaggle — Telco Customer Churn: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Para facilitar a reprodução sem exigir credenciais do Kaggle, o projeto disponibiliza um script que baixa uma cópia pública do mesmo arquivo:

```bash
python -m src.data.download
```

O CSV bruto não é versionado no Git. Isso mantém o repositório leve e evita redistribuir o dataset como parte da entrega.

### Observação de qualidade

O campo `TotalCharges` contém valores vazios representados como espaços em algumas linhas. Esse caso será tratado explicitamente na etapa de preparação dos dados, e a decisão será registrada na EDA em vez de simplesmente descartar linhas sem justificativa.
