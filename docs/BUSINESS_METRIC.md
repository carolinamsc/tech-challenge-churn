# Métrica de negócio — priorização de retenção

## Objetivo

A previsão de churn será usada para priorizar clientes para ações de retenção. Por isso, além das métricas estatísticas do modelo, acompanhamos uma métrica operacional: **taxa de intervenção**.

### Taxa de intervenção

> Percentual da carteira que seria encaminhada para uma ação de retenção no threshold escolhido.

\[
\text{Taxa de intervenção} = \frac{\text{clientes classificados como churn}}{\text{clientes avaliados}}
\]

Essa métrica traduz o threshold em capacidade operacional: um threshold menor aumenta a cobertura de potenciais churns, mas também aumenta o número de clientes que precisam ser contatados.

## Métrica combinada para decisão

A decisão é analisada em conjunto com **recall (churn capture)**:

- **Recall:** percentual dos clientes que realmente cancelaram e foram identificados pelo modelo.
- **Taxa de intervenção:** percentual da carteira que receberia uma ação.

Assim, a pergunta de negócio deixa de ser apenas "qual modelo tem maior AUC?" e passa a ser:

> **Quanto da perda potencial conseguimos capturar com a capacidade de intervenção disponível?**

No experimento de threshold da Regressão Logística, o ponto final adotado foi **0,55**, com **75,1% de recall**, **52,4% de precision**, **F1 de 0,6176** e **38,0% de taxa de intervenção** — **536 de 1.409 clientes** seriam acionados no holdout.

| Threshold | Precision | Recall | F1 | Taxa de intervenção | Clientes acionados |
|---|---:|---:|---:|---:|---:|
| 0,20 | 38,9% | 96,3% | 0,5538 | 65,7% | 926 |
| 0,30 | 42,9% | 92,8% | 0,5866 | 57,4% | 809 |
| 0,35 | 44,8% | 90,6% | 0,5995 | 53,7% | 757 |
| 0,40 | 46,6% | 86,6% | 0,6062 | 49,3% | 695 |
| 0,45 | 48,3% | 83,7% | 0,6125 | 46,0% | 648 |
| 0,50 | 50,4% | 78,3% | 0,6136 | 41,2% | 581 |
| **0,55** | **52,4%** | **75,1%** | **0,6176** | **38,0%** | **536** |
| 0,60 | 54,0% | 70,6% | 0,6118 | 34,7% | 489 |
| 0,65 | 56,7% | 66,6% | 0,6125 | 31,2% | 439 |
| 0,70 | 60,3% | 60,2% | 0,6024 | 26,5% | 373 |

O threshold de 0,55 maximiza o F1 entre os pontos avaliados, ao mesmo tempo em que reduz a taxa de intervenção em relação ao threshold padrão de 0,50.

## Evolução possível

Em um cenário real, os custos de retenção e o valor esperado de um cliente recuperado devem substituir essa análise operacional por uma função de custo/benefício parametrizada. O projeto não assume valores financeiros sem dados da operação.
