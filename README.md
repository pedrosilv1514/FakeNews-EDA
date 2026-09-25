# Fake News Classification

Projeto de Engenharia de Machine Learning e Ciencia de Dados para estudar classificacao de noticias em portugues brasileiro. Esta primeira versao cobre aquisicao, perfil, qualidade, EDA textual, vieses e risco de data leakage. Ela **nao treina modelos** e **nao faz train/test split**.

## Objetivo

Construir uma base reprodutivel e auditavel para experimentos futuros de classificacao, probabilidade calibrada e explicabilidade. A prioridade atual e entender a unidade textual, as definicoes de label, a proveniencia, as duplicatas e os atalhos que poderiam inflar uma avaliacao.

## Datasets

### FactChecks.br

Fonte: [GitHub — FactChecks.br](https://github.com/fake-news-UFG/FactChecks.br/tree/main)

O release v0.1 contem cinco benchmarks com tarefas e schemas distintos:

- `fakebr`: texto de noticia/alegacao e metadados;
- `FakeRecogna`: texto de revisao/fonte no schema consolidado;
- `central_de_fatos`: textos de checagens;
- `fact_check_tweet_pt`: pares entre checagens e IDs de tweets, sem texto integral;
- `FakeNewsSet`: pares entre checagens e IDs, sem texto integral.

O release tambem traz `fake_br.tsv`, byte a byte identico a `fakebr.tsv`. O arquivo bruto e preservado, mas o alias nao e contado duas vezes nas analises.

### FakeRecogna

Fonte: [Hugging Face — FakeRecogna](https://huggingface.co/datasets/recogna-nlp/FakeRecogna)

O CSV original tem os campos `Titulo`, `Subtitulo`, `Noticia`, `Categoria`, `Data`, `Autor`, `URL` e `Classe`. O carregador usa preferencialmente `datasets` sobre a copia local; ha fallback para pandas. O raw nao e alterado.

As duas ocorrencias de FakeRecogna (fonte independente e subconjunto consolidado no FactChecks.br) sao mantidas separadas porque possuem contagens e schemas diferentes.

## Arquitetura

```text
data/raw/        fontes exatamente como obtidas (imutaveis e ignoradas pelo Git)
data/interim/    um Parquet padronizado por dataset, sem concatenacao
data/processed/  reservado para dados prontos para modelagem apos decisao metodologica
configs/         caminhos relativos e seed global
notebooks/       cinco analises narrativas que chamam funcoes de src/
src/data/        carregamento e materializacao intermediaria
src/preprocessing/ limpeza minima, datas, dominios e labels documentados
src/analysis/    perfil, texto, vocabulario, duplicatas e leakage
src/visualization/ graficos orientados a perguntas
reports/figures/ figuras geradas
reports/tables/  evidencias tabulares auditaveis
tests/           testes unitarios relevantes
```

## Instalacao

Requer Python 3.10 ou superior. A partir da raiz:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

As dependencias foram limitadas a pandas, NumPy, Matplotlib, scikit-learn, Hugging Face `datasets`, PyArrow, PyYAML, Jupyter e pytest. Nao ha Transformers, PyTorch, TensorFlow, LangChain ou RAG.

## Como executar

Para uma aquisicao nova, somente com destinos `data/raw` ausentes:

```bash
python scripts/fetch_data.py --acknowledge-network-download
```

O script se recusa a sobrescrever qualquer destino raw. Depois:

```bash
python -m src.data.make_dataset
python -m src.analysis.pipeline
pytest
jupyter lab
```

O pipeline cria Parquets separados em `data/interim`, tabelas em `reports/tables`, figuras em `reports/figures` e atualiza `reports/eda_summary.md`. Para uma verificacao mais rapida, `--skip-near-duplicates` pula somente a busca aproximada; duplicatas exatas continuam sendo calculadas.

### Relatorio LaTeX

Depois de executar a EDA, gere o relatorio academico com:

```bash
python scripts/generate_latex_report.py \
  --author "Nome do(a) autor(a)" \
  --institution "Nome da instituicao"
```

O resultado é um único fonte editável em `reports/eda_report.tex`. Ele reutiliza diretamente as figuras já existentes em `reports/figures`, sem criar cópias ou pacotes adicionais.

Para compilar localmente:

```bash
cd reports
latexmk -pdf -interaction=nonstopmode eda_report.tex
```

Para utilizar no Overleaf, envie `eda_report.tex` e a pasta `figures/`. No Overleaf, selecione `eda_report.tex` como documento principal.

## Estrutura dos notebooks

1. `01_dataset_overview.ipynb`: schemas, dimensoes, amostras, labels, periodos, fontes e comparacao inicial.
2. `02_data_quality.ipynb`: ausentes, duplicatas completas/campo/cruzadas, labels e candidatos quase duplicados.
3. `03_text_eda.ipynb`: comprimento, pontuacao, URLs, numeros, caixa, vocabulario, n-gramas e TF-IDF.
4. `04_bias_and_leakage.ipynb`: associacoes label–fonte/autor/categoria/tempo/dataset e tabela de riscos.
5. `05_dataset_comparison.ipynb`: comparacao e evidencias para uso separado, combinacao seletiva ou holdout externo.

Execute-os na ordem. Cada notebook pode reconstruir `data/interim` se necessario, mas nunca escreve em `data/raw`.

## Metodologia

- Preservar label bruto e documentar sua interpretacao por fonte.
- Nao combinar datasets ou mapear `other` para binario automaticamente.
- Nao remover stopwords, fazer stemming/lematizacao ou eliminar colunas antes da auditoria.
- Detectar sobreposicao por URL, hash de texto normalizado e candidatos por n-gramas de caracteres.
- Adiar o split ate definir grupos de noticia, fonte e periodo.
- Interpretar cada achado como **evidencia → interpretacao → possivel impacto**.

Estrategias futuras incluem split estratificado depois do agrupamento, group split por noticia ou fonte, split temporal e dataset externo como holdout. Cada uma resolve riscos diferentes; nenhuma foi selecionada nesta etapa.

Os primeiros baselines recomendados sao TF-IDF + Logistic Regression, TF-IDF + Linear SVM e Bag-of-Words + Logistic Regression. As metricas futuras incluem Accuracy, Precision, Recall, F1, Macro-F1, matriz de confusao, ROC-AUC e PR-AUC quando aplicaveis. Accuracy nunca sera usada isoladamente.

Uma probabilidade emitida por `predict_proba()` e uma estimativa do classificador, **nao um score de confiabilidade factual da noticia**. Calibracao, Brier Score, Log Loss, reliability diagrams, Platt Scaling e Isotonic Regression ficam para uma etapa posterior.

## Versionamento

- **Git:** codigo, configuracoes, notebooks, testes e relatorio textual.
- **DVC (futuro):** arquivos raw, interim, processed e seus checksums; nenhum remote e configurado agora.
- **MLflow (futuro):** parametros, metricas, artefatos e linhagem dos experimentos de baseline.

Os dados e modelos estao no `.gitignore`. `reports/tables/raw_manifest.csv` registra tamanho e SHA-256 das fontes adquiridas.

## Proximas etapas

1. Revisar evidencias e decidir a unidade de classificacao.
2. Validar manualmente conflitos e quase duplicatas.
3. Definir grupos e protocolo de split livre de leakage.
4. Materializar um dataset `processed` versionado por DVC.
5. Implementar os tres baselines tradicionais e calibracao.
6. Somente depois comparar BERTimbau e outros Transformers.

Consulte [o relatorio de EDA](reports/eda_summary.md) para resultados, limitacoes e recomendacoes.
