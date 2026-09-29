# Modelos avaliados

Os três modelos individuais usam TF-IDF, mas representam hipóteses de
aprendizado diferentes.

| Modelo | Mecanismo | Ponto forte | Limitação principal |
|---|---|---|---|
| Logistic Regression | log-loss linear | simplicidade e coeficientes | ligeiramente inferior ao SVM |
| Linear SVM | maximização de margem | textos esparsos de alta dimensão | requer calibração para probabilidades |
| Multinomial NB | frequências condicionais | rápido e complementar | independência condicional simplificadora |
| Stacking | meta-aprendizado sobre probabilidades | combina vieses distintos | custo e complexidade maiores |

## Pipelines persistidos

```text
E01: texto → TF-IDF → Logistic Regression → P(fake)
E02: texto → TF-IDF → Linear SVM → calibração sigmoid → P(fake)
E03: texto → TF-IDF → Multinomial NB → P(fake)
E04: texto → E02 + E03 → Logistic Regression → P(fake)
```

Os artefatos `.joblib` incluem vetorizador, classificador, calibração quando
aplicável e metadados de versão. Eles são gerados localmente e não são enviados
ao Git por estarem no `.gitignore`.

!!! info "Por que começar com modelos tradicionais?"
    Eles fornecem baselines rápidos, reproduzíveis e suficientemente fortes para
    medir se a complexidade de um Transformer realmente acrescenta valor.
