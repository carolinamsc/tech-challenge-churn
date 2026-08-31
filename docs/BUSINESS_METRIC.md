# Métrica de negócio — priorização de retenção

## Objetivo

A previsão de churn será usada para priorizar clientes para ações de retenção. Por isso, além das métricas estatísticas do modelo, acompanhamos uma métrica operacional: **taxa de intervenção**.

### Taxa de intervenção

> Percentual da carteira que seria encaminhada para uma ação de retenção no threshold escolhido.

\[
\text{Taxa de intervenção} = \frac{\text{clientes classificados como churn}}{\text{clientes avaliados}}
\]

Essa métrica permite traduzir o threshold em capacidade operacional: um threshold menor aumenta a cobertura de potenciais churns, mas também aumenta o número de clientes que precisam ser contatados.

## Métrica combinada para decisão

A decisão é analisada em conjunto com **recall (churn capture)**:

- **Recall:** percentual dos clientes que realmente cancelaram e foram identificados pelo modelo.
- **Taxa de intervenção:** percentual da carteira que receberia uma ação.

Assim, a pergunta de negócio deixa de ser apenas "qual modelo tem maior AUC?" e passa a ser:

> **Quanto da perda potencial conseguimos capturar com a capacidade de intervenção disponível?**

No experimento de threshold, a Regressão Logística com threshold 0,55 apresentou aproximadamente **75% de recall** com **53,6% de taxa de intervenção**, sendo o ponto de maior F1 entre os thresholds avaliados.

## Evolução possível

Em um cenário real, os custos de retenção e o valor esperado de um cliente recuperado devem substituir essa análise operacional por uma função de custo/benefício parametrizada. O projeto não assume valores financeiros sem dados da operação.
