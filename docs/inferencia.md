# Testando a classificação

## Modelo recomendado

O artefato local recomendado é:

```text
models/E02_linear_svm.joblib
```

Ele contém TF-IDF, Linear SVM, calibradores sigmoid e metadados. Como a pasta
`models/` é ignorada pelo Git, os artefatos precisam ser treinados após um clone.

## Linha de comando

```bash
python -m src.predict \
  --title "Título da notícia" \
  --content "Conteúdo completo da notícia"
```

## Uso em Python

```python
from src.predict import NewsClassifier

classifier = NewsClassifier("models/E02_linear_svm.joblib")

result = classifier.predict(
    title="Título da notícia",
    content="Conteúdo completo da notícia",
)
print(result)
```

## Formato da resposta

```json
{
  "predicted_class": "real",
  "predicted_class_probability": 0.9683,
  "probability_fake": 0.0317,
  "probability_real": 0.9683,
  "threshold": 0.5,
  "model_id": "E02",
  "model_version": "1.0.0",
  "disclaimer": "Probabilidade de pertencimento a classe; nao comprova a veracidade factual."
}
```

## Recriando os modelos

```bash
python -m src.preprocessing.modeling
python -m src.train_baselines
python -m src.train_stacking  # opcional para E04
```

!!! note "GitHub Pages é apenas documentação"
    O Pages não executa Python nem carrega o `.joblib`. A inferência interativa
    exigirá um backend separado, como FastAPI. O frontend poderá consumir esse
    backend futuramente.
