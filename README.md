# Fake News Classification — PT-BR

[![Documentação](https://img.shields.io/badge/docs-GitHub%20Pages-4051b5)](https://pedrosilv1514.github.io/FakeNews-EDA/)

Pipeline reprodutivel de EDA, treinamento, calibracao, stacking e inferencia para
classificacao binaria de noticias em portugues brasileiro. O projeto nao usa
LLMs para classificar e nao implementa frontend nem verificacao factual.

> A probabilidade emitida e uma estimativa de pertencimento a classe aprendida
> no corpus. Ela nao comprova se uma noticia e factual.

## Dados e decisao metodologica

Os dados locais vieram de [FakeRecogna](https://huggingface.co/datasets/recogna-nlp/FakeRecogna)
e [FactChecks.br](https://github.com/fake-news-UFG/FactChecks.br). A EDA encontrou
42.591 linhas contadas entre seis subconjuntos, 11.877 assinaturas cruzadas e 132
conflitos semanticos. Dois subconjuntos nao têm texto, `central_de_fatos` e muito
desbalanceado e FactChecks/FakeRecogna se sobrepoe ao original.

Por isso, os experimentos usam o FakeRecogna original como benchmark canonico:
11.858 registros apos validacao/conflitos, balanceados entre fake/real. Grupos de
URL, texto exato e quase duplicatas nunca cruzam treino, validacao ou teste.
Metadados de fonte nao entram como features. Consulte [a EDA](reports/eda_summary.md),
[a pesquisa de stacking](reports/stacking_research.md) e
[o relatorio de experimentos](reports/ml_experiments.md).

A versão navegável da documentação está preparada para publicação em
[pedrosilv1514.github.io/FakeNews-EDA](https://pedrosilv1514.github.io/FakeNews-EDA/).

## Estrutura

```text
data/raw/                 fontes imutaveis (ignoradas pelo Git)
data/interim/             schemas padronizados da EDA
data/processed/           corpus e splits de modelagem
src/preprocessing/        limpeza conservadora, grupos e splits
src/models/               factories classicas e stacking carregavel
src/evaluation/           metricas, calibracao, erros e graficos
src/train_baselines.py    tuning e treino E01-E03
src/train_stacking.py     OOF manual, ablations e treino E04
src/train_bert.py         extensao opcional E05
src/evaluate_temporal.py  avaliacao passado -> futuro
src/predict.py            interface de inferencia/CLI
models/                   artefatos joblib locais (ignorados pelo Git)
reports/metrics/          metricas e predicoes auditaveis
reports/figures/          comparacoes, ROC/PR e calibracao
tests/                    testes automatizados
```

## Instalacao

Requer Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

O ambiente exato da execucao entregue esta em
`reports/metrics/environment.json`. Dados, modelos e caches continuam no
`.gitignore`; somente codigo e relatorios leves sao versionados.

## Reproducao

Com os dados brutos locais:

```bash
python -m src.data.make_dataset
python -m src.analysis.pipeline
python -m src.preprocessing.modeling
python -m src.train_baselines
python -m src.train_stacking
python -m src.evaluate_temporal
pytest -q
```

Ou, depois da EDA/interim:

```bash
python scripts/run_ml_pipeline.py
```

Cada TF-IDF e ajustado dentro do respectivo fold. O tuning usa cinco folds de
`StratifiedGroupKFold`, Macro-F1 e apenas o treino. A validacao escolhe a ablation
do stacking; o teste fica isolado ate a avaliacao final.

## Documentacao local

```bash
python -m pip install -r requirements-docs.txt
python scripts/prepare_docs_assets.py
mkdocs serve
```

O build usado pelo GitHub Pages pode ser validado com `mkdocs build --strict`.

## Resultados

| ID | Modelo | Macro-F1 | ROC-AUC | Brier | Status |
|---|---|---:|---:|---:|---|
| E01 | TF-IDF + Logistic Regression | 0,9752 | 0,9968 | 0,0271 | concluido |
| E02 | TF-IDF + Linear SVM calibrado | **0,9876** | **0,9982** | **0,0108** | concluido |
| E03 | TF-IDF + Multinomial NB | 0,9522 | 0,9890 | 0,0362 | concluido |
| E04 | OOF Stacking (SVM + NB) | 0,9852 | 0,9979 | 0,0112 | concluido |
| E05 | BERTimbau Base | — | — | — | opcional, nao executado |
| E06 | Stacking + BERTimbau | — | — | — | opcional, nao executado |

O candidato recomendado e E02: melhor qualidade discriminativa/probabilistica,
menor custo e Macro-F1 temporal de 0,9815. E04 nao superou o melhor individual.

## Inferencia

```bash
python -m src.predict \
  --model models/E02_linear_svm.joblib \
  --title "Titulo da noticia" \
  --content "Conteudo completo da noticia"
```

A resposta inclui classe, probabilidades, threshold, ID/versao e disclaimer. A
classe `NewsClassifier` pode ser instanciada diretamente por um futuro FastAPI.

## BERTimbau opcional

```bash
python -m pip install -e ".[bert]"
python -m src.train_bert --epochs 3 --max-length 512
```

O ambiente entregue nao inclui PyTorch/Transformers e E05/E06 nao foram rodados.
E06 exige previsoes BERT OOF; usar predicoes do mesmo ajuste no meta-learner
constituiria data leakage.
