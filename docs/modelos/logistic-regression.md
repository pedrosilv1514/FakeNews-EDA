# E01 — Logistic Regression

## Estrutura

```text
Texto composto
  → TF-IDF de unigramas e bigramas
  → Logistic Regression, C=2,0
  → P(fake)
```

O melhor ajuste usa bigramas, `min_df=5` e regularização controlada por `C=2,0`.
A saída probabilística é nativa da regressão logística.

## Resultado

| Métrica | Valor |
|---|---:|
| Accuracy | 0,9752 |
| Precision | 0,9809 |
| Recall | 0,9693 |
| Macro-F1 | 0,9752 |
| ROC-AUC | 0,9968 |
| Brier Score | 0,0271 |

![Diagnóstico do E01](../assets/generated/E01_diagnostics.png)

## Leitura técnica

É uma baseline forte, barata e mais simples de inspecionar por coeficientes. No
entanto, ficou abaixo do SVM tanto no teste agrupado quanto no temporal. Seus
erros têm sobreposição parcial com os demais modelos, mas sua inclusão no
stacking não melhorou a validação.
