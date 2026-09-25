"""Gera os cinco notebooks versionados a partir de celulas declarativas."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def py(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


SETUP = '''from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, ensure_output_directories
from src.data.make_dataset import build_interim_datasets, load_interim_datasets

cfg = load_config()
ensure_output_directories(cfg)
if not list(Path(cfg["data"]["interim"]).glob("*.parquet")):
    build_interim_datasets()
datasets = {str(frame["dataset"].iloc[0]): frame for frame in load_interim_datasets().values()}
list(datasets)
'''


OVERVIEW = [
    md('''# 01 — Dataset Overview

Pergunta: **o que cada fonte realmente contem e quais comparacoes sao validas?**

Schemas e labels originais sao preservados. FactChecks.br e uma colecao de tarefas heterogeneas; o FakeRecogna independente permanece separado.'''),
    py(SETUP),
    py('''from src.data.load_data import load_factchecks, load_fakerecogna, discover_factchecks_files
from src.analysis.profiling import dataset_overview, label_distribution, top_values

raw = {**load_factchecks(include_aliases=True), "fakerecogna_original": load_fakerecogna()}
pd.DataFrame({"arquivo": [p.name for p in discover_factchecks_files()], "bytes": [p.stat().st_size for p in discover_factchecks_files()]})'''),
    md('''## Schemas, tipos e amostras

Nao pressupomos colunas comuns e nao eliminamos campos antes da auditoria.'''),
    py('''for name, frame in raw.items():
    print(f"\\n### {name}: {frame.shape[0]:,} linhas x {frame.shape[1]} colunas")
    display(frame.dtypes.rename("dtype").to_frame())
    display(frame.head(3))'''),
    md('''## Variavel alvo e labels

`is_fake` e o alvo no release consolidado; `Classe` e o alvo no FakeRecogna original. A mesma codificacao numerica nao possui a mesma semantica entre fontes.'''),
    py('''pd.concat([label_distribution(frame, name) for name, frame in datasets.items()], ignore_index=True)'''),
    md('## Periodo temporal e principais fontes'),
    py('''periods, sources = [], []
for name, frame in datasets.items():
    dates = frame["date_parsed"].dropna()
    periods.append({"dataset": name, "datas_validas": len(dates), "inicio": dates.min() if len(dates) else pd.NaT, "fim": dates.max() if len(dates) else pd.NaT})
    sources.append(top_values(frame, "domain", name, limit=10))
display(pd.DataFrame(periods))
display(pd.concat(sources, ignore_index=True))'''),
    md('''## Resumo comparativo

Informacoes ausentes permanecem vazias; nada e inferido.'''),
    py('''dataset_overview(datasets).rename(columns={"dataset": "Dataset", "records": "Registros", "raw_labels": "Classes", "period_start": "Inicio", "period_end": "Fim", "main_columns": "Colunas principais"})'''),
]


QUALITY = [
    md('''# 02 — Data Quality

Pergunta: **onde estao ausencias, duplicatas e divergencias de label capazes de invalidar a avaliacao?** Nenhuma linha e removida.'''),
    py(SETUP),
    py('''from src.analysis.profiling import missingness_table, label_distribution
from src.analysis.duplicates import duplicate_summary, cross_dataset_duplicates, near_duplicate_candidates

missing = pd.concat([missingness_table(frame, name) for name, frame in datasets.items()], ignore_index=True)
missing.sort_values(["dataset", "missing_percent"], ascending=[True, False])'''),
    md('## Duplicatas completas, de titulo, texto e URL'),
    py('''duplicates = pd.concat([duplicate_summary(frame, name) for name, frame in datasets.items()], ignore_index=True)
duplicates'''),
    md('''## Sobreposicao entre datasets

`not_fake` e `real` sao equiparados somente para marcar conflitos; labels originais nao sao alterados.'''),
    py('''cross = cross_dataset_duplicates(datasets)
display(cross.groupby(["match_type", "datasets", "label_conflict"]).size().rename("groups").reset_index())
display(cross.head(20))'''),
    md('''## Candidatos quase duplicados

TF-IDF de n-gramas de caracteres com similaridade >= 0,96 gera candidatos para revisao, nao exclusoes automaticas.'''),
    py('''near = []
for name, frame in datasets.items():
    if frame["text"].fillna("").str.len().ge(80).sum() >= 2:
        near.append({"dataset": name, "candidate_pairs": len(near_duplicate_candidates(frame, threshold=0.96))})
pd.DataFrame(near)'''),
    md('''## Definicoes e distribuicoes dos labels

Nao ha unificacao automatica. `other` permanece uma terceira semantica.'''),
    py('''pd.concat([label_distribution(frame, name) for name, frame in datasets.items()], ignore_index=True)'''),
]


TEXT_EDA = [
    md('''# 03 — EDA textual

Pergunta: **comprimento, pontuacao, caixa, numeros e vocabulario diferem por classe — e parecem conteudo ou artefato?**

Nao removemos stopwords, nao aplicamos stemming/lematizacao e nao usamos WordCloud como evidencia.'''),
    py(SETUP),
    py('''from src.analysis.text_metrics import add_text_features, summarize_text_features
from src.analysis.vocabulary import top_terms_by_class, class_distinctive_terms
from src.visualization.plots import plot_text_length_by_class, plot_punctuation_by_class

featured = {name: add_text_features(frame) for name, frame in datasets.items()}
pd.concat([summarize_text_features(frame, name) for name, frame in featured.items()], ignore_index=True)'''),
    md('''## Comprimento e pontuacao por classe

Os graficos respondem se atributos superficiais separam classes dentro de cada dataset.'''),
    py('''figure_dir = Path(cfg["reports"]["figures"])
for name, frame in featured.items():
    if frame["text"].fillna("").str.strip().ne("").sum() < 10:
        continue
    slug = name.replace("/", "_").casefold()
    display(frame.groupby("label_semantic")[["n_characters", "n_words", "n_sentences", "title_length", "n_exclamations", "n_questions", "n_urls", "n_numbers", "uppercase_percent"]].median())
    plot_text_length_by_class(frame, figure_dir / f"{slug}_text_length.png")
    plot_punctuation_by_class(frame, figure_dir / f"{slug}_punctuation.png")'''),
    md('## Unigramas, bigramas e TF-IDF'),
    py('''vocabulary = []
for name, frame in featured.items():
    if frame["text"].fillna("").str.strip().ne("").sum() < 10:
        continue
    for method, ngram in (("count", (1, 1)), ("count", (2, 2)), ("tfidf", (1, 2))):
        table = top_terms_by_class(frame, method=method, ngram_range=ngram, top_n=20)
        table.insert(0, "dataset", name)
        vocabulary.append(table)
pd.concat(vocabulary, ignore_index=True)'''),
    md('''## Termos potencialmente artificiais

Diferenca de TF-IDF medio destaca marcadores candidatos a atalho, como nomes de agencias, boilerplate e tags editoriais.'''),
    py('''distinctive = []
for name, frame in featured.items():
    if frame["text"].fillna("").str.strip().ne("").sum() < 10:
        continue
    table = class_distinctive_terms(frame, top_n=25)
    table.insert(0, "dataset", name)
    distinctive.append(table)
pd.concat(distinctive, ignore_index=True)'''),
]


LEAKAGE = [
    md('''# 04 — Bias & Data Leakage

Pergunta: **o label pode ser previsto por fonte, dominio, autor, periodo, categoria, tamanho, titulo ou origem?**

Severidade: alta quando um atalho afeta grande parte da base ou compartilha noticias entre folds; media quando localizado/condicional; baixa quando a cobertura e pequena.'''),
    py(SETUP),
    py('''from src.analysis.duplicates import cross_dataset_duplicates
from src.analysis.leakage import metadata_label_association, leakage_risk_table, weighted_purity

parts = [metadata_label_association(frame, field, name, min_count=5) for name, frame in datasets.items() for field in ("domain", "author", "category")]
associations = pd.concat([part for part in parts if not part.empty], ignore_index=True)
associations.sort_values(["purity", "total"], ascending=False).head(50)'''),
    md('## O label pode ser descoberto apenas por fonte, autor ou categoria?'),
    py('''associations.groupby(["dataset", "field"]).apply(lambda table: pd.Series({"weighted_purity": weighted_purity(table), "records_in_pure_groups": table.loc[table["purity"].eq(1), "total"].sum(), "groups": len(table)}), include_groups=False).reset_index()'''),
    md('## As classes cobrem periodos diferentes?'),
    py('''periods = []
for name, frame in datasets.items():
    dated = frame.dropna(subset=["date_parsed", "label_semantic"])
    if not dated.empty:
        table = dated.groupby("label_semantic")["date_parsed"].agg(["count", "min", "max"]).reset_index()
        table.insert(0, "dataset", name)
        periods.append(table)
pd.concat(periods, ignore_index=True)'''),
    md('## O dataset de origem determina a classe?'),
    py('''combined = pd.concat(datasets.values(), ignore_index=True)
pd.crosstab(combined["dataset"], combined["label_semantic"], normalize="index")'''),
    md('## Textos semelhantes e tabela de riscos'),
    py('''cross = cross_dataset_duplicates(datasets)
display(cross.groupby(["match_type", "label_conflict"]).size().rename("groups").reset_index())
leakage_risk_table(datasets, cross).rename(columns={"risk": "Risco", "evidence": "Evidencia", "severity": "Severidade", "recommendation": "Recomendacao"})'''),
    md('''## Regras antes de modelar

- Criar grupos por URL, hash e cluster de quase duplicatas.
- Comparar modelos com e sem marcadores/metadados.
- Nao usar split aleatorio por linha.
- Avaliar group split, split temporal e holdout externo descontaminado.'''),
]


COMPARISON = [
    md('''# 05 — Comparacao dos datasets

Pergunta: **quais combinacoes sao metodologicamente defensaveis?** Apresentamos evidencias sem escolher automaticamente uma estrategia.'''),
    py(SETUP),
    py('''from src.analysis.profiling import dataset_overview, missingness_table, label_distribution
from src.analysis.duplicates import duplicate_summary, cross_dataset_duplicates
from src.analysis.text_metrics import add_text_features, summarize_text_features

overview = dataset_overview(datasets)
missing = pd.concat([missingness_table(frame, name) for name, frame in datasets.items()], ignore_index=True)
labels = pd.concat([label_distribution(frame, name) for name, frame in datasets.items()], ignore_index=True)
duplicates = pd.concat([duplicate_summary(frame, name) for name, frame in datasets.items()], ignore_index=True)
text_summary = pd.concat([summarize_text_features(add_text_features(frame), name) for name, frame in datasets.items()], ignore_index=True)
display(overview); display(labels); display(duplicates); display(text_summary)'''),
    md('## Cobertura, fontes e dominios'),
    py('''pd.DataFrame([{"dataset": name, "unique_domains": frame["domain"].replace("", pd.NA).nunique(), "unique_authors": frame["author"].replace("", pd.NA).nunique(), "valid_dates": frame["date_parsed"].notna().sum(), "missing_text_percent": 100 * frame["text"].fillna("").str.strip().eq("").mean()} for name, frame in datasets.items()])'''),
    md('''## A. Usar separadamente

Preserva tarefa, label e coleta. Favorece diagnostico intra-dominio, mas nao mede transferencia.'''),
    md('''## B. Combinar subconjuntos

Somente apos alinhar unidade textual/taxonomia e criar grupos de duplicidade. `other` nao deve ser binarizado por conveniencia; datasets sem texto nao equivalem a classificacao textual.'''),
    md('''## C. Treino e teste externo

Mede generalizacao entre fontes. Antes, remover toda sobreposicao e confirmar equivalencia de tarefa e label.'''),
    py('''cross = cross_dataset_duplicates(datasets)
cross.groupby(["datasets", "match_type", "label_conflict"]).size().rename("groups").reset_index()'''),
    md('''## Decisao pendente

Revisar o relatorio, conflitos e quase duplicatas. A decisao deve declarar unidade de predicao, taxonomia, deduplicacao, split e populacao-alvo.'''),
]


NOTEBOOKS = {
    "01_dataset_overview.ipynb": OVERVIEW,
    "02_data_quality.ipynb": QUALITY,
    "03_text_eda.ipynb": TEXT_EDA,
    "04_bias_and_leakage.ipynb": LEAKAGE,
    "05_dataset_comparison.ipynb": COMPARISON,
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}
    for name, cells in NOTEBOOKS.items():
        for index, cell in enumerate(cells):
            cell["id"] = f"cell-{index:02d}"
        notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
        (NOTEBOOK_DIR / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(NOTEBOOK_DIR / name)


if __name__ == "__main__":
    main()
