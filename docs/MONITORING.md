# Plano de monitoramento — Churn Prediction

## Objetivo

O monitoramento deve verificar se a distribuição dos dados, as previsões e a capacidade operacional continuam próximas do comportamento observado durante a validação.

## O que monitorar

### Dados e drift

Acompanhar a distribuição das features com maior relevância para o modelo e para a decisão de negócio, especialmente:

- `Contract`
- `tenure`
- `MonthlyCharges`
- `PaymentMethod`

Comparar a distribuição atual com a referência usada no treinamento. Mudanças persistentes podem indicar data drift e necessidade de revalidação.

### Saída do modelo

Monitorar:

- distribuição das probabilidades preditas;
- percentual de clientes acima do threshold `0,55`;
- taxa de intervenção observada versus a taxa esperada de aproximadamente **38%** no cenário de referência;
- volume de chamadas ao endpoint `/predict`;
- latência e taxa de erro da API.

### Performance

O churn real só fica conhecido depois do período de observação. Quando novos rótulos estiverem disponíveis, recalcular ROC-AUC, F1, precision e recall e comparar com a validação de referência.

## Gatilhos para revalidação ou retreino

Considerar investigação e revalidação quando ocorrer qualquer uma das condições:

- ROC-AUC abaixo de **0,80** na base rotulada mais recente;
- taxa de intervenção persistindo fora da faixa de **30% a 45%** sem mudança planejada na política de retenção;
- drift relevante e persistente nas principais features;
- mudança operacional que altere o perfil dos clientes ou a estratégia comercial.

Esses valores são **critérios operacionais iniciais**, não evidência de que uma degradação já ocorreu em produção.

## Limitação atual

O projeto não possui rótulo de churn em tempo real. Portanto, a avaliação de performance é necessariamente **retrospectiva**: primeiro são geradas as previsões, depois os resultados reais ficam disponíveis e somente então as métricas podem ser calculadas.

## Playbook resumido

1. Detectar desvio em dados, previsões, taxa de intervenção ou performance.
2. Verificar se a mudança é causada por alteração planejada no produto, preço ou população.
3. Reexecutar a avaliação no conjunto de dados mais recente.
4. Se houver degradação confirmada, revisar features, threshold e necessidade de retreino.
5. Registrar a decisão, métricas e versão do modelo.
