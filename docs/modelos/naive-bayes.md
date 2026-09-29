# E03 — Multinomial Naive Bayes

## Estrutura

```text
Texto composto
  → TF-IDF de unigramas e bigramas, min_df=2
  → Multinomial Naive Bayes, alpha=0,1
  → P(fake)
```

O modelo estima evidência lexical condicionada à classe. Sua hipótese de
independência entre termos é simplificadora, mas produz um viés diferente dos
classificadores lineares discriminativos.

## Resultado

| Métrica | Valor |
|---|---:|
| Accuracy | 0,9522 |
| Precision | 0,9581 |
| Recall | 0,9457 |
| Macro-F1 | 0,9522 |
| ROC-AUC | 0,9890 |
| Brier Score | 0,0362 |

![Diagnóstico do E03](../assets/generated/E03_diagnostics.png)

## Papel no projeto

O NB não é candidato individual, mas seus erros são os menos sobrepostos aos do
SVM: Jaccard de 0,186. Essa complementaridade explica por que ele foi mantido na
melhor ablação do stacking.
