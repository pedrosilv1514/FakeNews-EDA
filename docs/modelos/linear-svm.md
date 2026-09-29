# E02 — Linear SVM calibrado

## Estrutura

```text
Texto composto
  → TF-IDF de unigramas e bigramas
  → Linear SVM, C=1,5
  → Calibração sigmoid group-aware
  → P(fake)
```

O SVM aprende um hiperplano de máxima margem no espaço esparso do TF-IDF. Como a
margem não é probabilidade, `CalibratedClassifierCV` aplica calibração sigmoid
em folds internos que também respeitam os grupos.

## Resultado

| Métrica | Valor |
|---|---:|
| Accuracy | **0,9876** |
| Precision | 0,9847 |
| Recall | **0,9906** |
| Macro-F1 | **0,9876** |
| ROC-AUC | **0,9982** |
| Brier Score | **0,0108** |
| Log Loss | **0,0460** |

![Diagnóstico do E02](../assets/generated/E02_diagnostics.png)

## Por que é o candidato?

- melhor Macro-F1 e qualidade probabilística;
- apenas 21 erros em 1.694 notícias de teste;
- melhor resultado temporal, com Macro-F1 0,9815;
- menor custo e arquitetura mais simples que o stacking;
- pipeline único e diretamente carregável para inferência.
