# Experimentos de classificacao

## Protocolo

O corpus principal e o FakeRecogna original. Das 11.903 linhas, uma sem label foi
removida e 44 linhas em grupos com labels conflitantes foram excluidas. Restaram
11.858 registros e 11.615 grupos. Titulo, subtitulo e corpo compoem a entrada;
URL, dominio, autor, categoria, data e origem nao entram como features.

O split `StratifiedGroupKFold` fixo (seed 42) produziu 8.470/1.694/1.694 linhas
em treino/validacao/teste, cada parte exatamente balanceada. URL normalizada,
hashes de texto/corpo e 76 pares de quase duplicatas da EDA definem os grupos.
Hiperparametros usam cinco folds apenas no treino e Macro-F1 como criterio. A
validacao escolhe a ablation do stacking. Depois disso, os modelos finais usam
treino+validacao e o teste e aberto uma unica vez.

Nao foram removidas stopwords nem aplicados novos stemmers/lematizadores. Essas
operacoes podem apagar negacao, modalidade e nomes importantes. O corpo
distribuido pelo FakeRecogna ja aparenta pre-processamento linguistico; repetir
essa transformacao destruiria mais informacao.

## Resultados reais — teste isolado (n=1.694)

| Experimento | Modelo | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | PR-AUC | Brier | Log Loss | Custo de ajuste (s) | Inferencia (ms/amostra) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E01 | TF-IDF + Logistic Regression | 0,9752 | 0,9809 | 0,9693 | 0,9751 | 0,9752 | 0,9968 | 0,9959 | 0,0271 | 0,1289 | 20,31 | 0,100 |
| E02 | TF-IDF + Linear SVM calibrado | **0,9876** | 0,9847 | **0,9906** | **0,9876** | **0,9876** | **0,9982** | **0,9976** | **0,0108** | **0,0460** | 19,71 | 0,498 |
| E03 | TF-IDF + Multinomial NB | 0,9522 | 0,9581 | 0,9457 | 0,9519 | 0,9522 | 0,9890 | 0,9890 | 0,0362 | 0,1302 | 16,68 | 0,109 |
| E04 | OOF Stacking (SVM + NB) | 0,9852 | 0,9824 | 0,9882 | 0,9853 | 0,9852 | 0,9979 | 0,9971 | 0,0112 | 0,0483 | 82,28 | 0,288 |
| E05 | BERTimbau Base | — | — | — | — | — | — | — | — | — | — | — |
| E06 | Stacking + BERTimbau | — | — | — | — | — | — | — | — | — | — | — |

Para E01-E03, custo soma busca de hiperparametros e fit final; para E04, cobre
OOF, ablations e fit final. Tempos medidos no ambiente documentado em
`reports/metrics/environment.json`.
E05/E06 nao foram executados: `transformers` e PyTorch nao fazem parte do
ambiente e o fine-tuning/OOF de cinco BERTs nao deve comprometer o pipeline CPU.
Nao ha resultados simulados.

## Analise

E02 venceu em Macro-F1, ROC-AUC, PR-AUC, Brier e Log Loss. O stacking escolheu
SVM+NB na validacao (Macro-F1 0,9882), mas caiu para 0,9852 no teste e custou
aproximadamente 4,2 vezes o ajuste completo do SVM. A ablation com os
tres modelos nao melhorou a validacao. Logo, E02 e o candidato para a proxima
etapa; E04 continua disponivel como referencia experimental.

Os erros dos baselines nao sao identicos (Jaccard: E01/E02=0,40,
E02/E03=0,186), mas a diversidade nao se converteu em ganho final. Na avaliacao
temporal, E02 manteve Macro-F1 0,9815 (treino ate 2021-02-19, teste posterior),
contra 0,9674 de E01 e 0,9374 de E03.

As metricas sao altas, mas a EDA mostra forte separacao de fonte, autor, periodo,
comprimento e estilo entre classes. O sistema aprendeu o benchmark e nao deve ser
interpretado como verificador factual universal. Probabilidades sao estimativas
de classe condicionadas a esse corpus, nao percentuais de veracidade.

## BERTimbau

BERTimbau Base e um BERT cased de 12 camadas/~110M parametros pre-treinado em
portugues brasileiro. `src/train_bert.py` implementa tokenizacao, truncamento em
512 tokens, cabeca binaria, selecao por Macro-F1 na validacao e avaliacao final
no mesmo teste. O comando recomendado usa GPU/CUDA ou Apple MPS; CPU e possivel,
mas lento.

Para E06, cada feature de BERT usada pelo meta-learner teria de ser OOF: cinco
fine-tunings independentes no development, mais um modelo final. Inserir
predicoes in-sample do BERT seria leakage. Antes disso, devem ser medidos custo,
truncamento de textos longos e calibracao (temperature scaling em holdout/OOF).

Referencias: [paper/repositório BERTimbau](https://github.com/neuralmind-ai/portuguese-bert),
[modelo Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased) e
[guia de sequence classification](https://huggingface.co/docs/transformers/tasks/sequence_classification).
