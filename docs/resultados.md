# Comparação dos resultados

## Teste independente

Todos os experimentos tradicionais foram avaliados nos mesmos 1.694 registros,
com 847 exemplos por classe.

| Experimento | Modelo | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E01 | Logistic Regression | 0,9752 | 0,9809 | 0,9693 | 0,9751 | 0,9752 | 0,9968 | 0,9959 |
| E02 | Linear SVM calibrado | **0,9876** | 0,9847 | **0,9906** | **0,9876** | **0,9876** | **0,9982** | **0,9976** |
| E03 | Multinomial NB | 0,9522 | 0,9581 | 0,9457 | 0,9519 | 0,9522 | 0,9890 | 0,9890 |
| E04 | Stacking OOF | 0,9852 | 0,9824 | 0,9882 | 0,9853 | 0,9852 | 0,9979 | 0,9971 |

![Comparação dos modelos](assets/generated/model_comparison.png)

## Qualidade probabilística

| Experimento | Brier Score ↓ | Log Loss ↓ |
|---|---:|---:|
| E01 | 0,0271 | 0,1289 |
| E02 | **0,0108** | **0,0460** |
| E03 | 0,0362 | 0,1302 |
| E04 | 0,0112 | 0,0483 |

![Comparação de calibração](assets/generated/calibration_comparison.png)

Probabilidades calibradas são importantes para a futura apresentação do Hub,
mas continuam condicionadas ao dataset. Boa calibração não transforma o score
em probabilidade factual de uma notícia ser verdadeira.

## Custo computacional

| Experimento | Ajuste completo | Inferência por amostra | Tamanho local aproximado |
|---|---:|---:|---:|
| E01 | 20,31 s | 0,100 ms | 0,6 MB |
| E02 | 19,71 s | 0,498 ms | 2,6 MB |
| E03 | 16,68 s | 0,109 ms | 3,1 MB |
| E04 | 82,28 s | 0,288 ms | 5,7 MB |

O custo do E04 inclui geração OOF, seleção por ablação e ajuste final. Os tempos
são medições do ambiente experimental e variam conforme hardware e concorrência.

## Sobreposição dos erros

Jaccard menor significa que dois modelos erram conjuntos mais diferentes.

| Par | Jaccard dos erros |
|---|---:|
| E01 × E02 | 0,400 |
| E01 × E03 | 0,367 |
| E02 × E03 | **0,186** |
| E02 × E04 | 0,769 |

SVM e NB são complementares, mas essa diversidade não se converteu em ganho do
stacking no teste final.

## Experimentos opcionais

| Experimento | Situação | Motivo |
|---|---|---|
| E05 — BERTimbau Base | não executado | PyTorch/Transformers e acelerador não provisionados |
| E06 — Stacking + BERTimbau | não executado | exige cinco fine-tunings OOF mais modelo final |

Não há resultados simulados para E05 ou E06.
