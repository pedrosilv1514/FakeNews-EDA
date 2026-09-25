"""Executa a EDA reprodutivel e materializa tabelas, figuras e relatorio."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"),
)

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src.analysis.duplicates import (  # noqa: E402
    cross_dataset_duplicates,
    duplicate_summary,
    near_duplicate_candidates,
)
from src.analysis.leakage import (  # noqa: E402
    leakage_risk_table,
    metadata_label_association,
)
from src.analysis.profiling import (  # noqa: E402
    dataset_overview,
    label_distribution,
    missingness_table,
    top_values,
)
from src.analysis.text_metrics import add_text_features, summarize_text_features  # noqa: E402
from src.analysis.vocabulary import class_distinctive_terms, top_terms_by_class  # noqa: E402
from src.config import ensure_output_directories, load_config  # noqa: E402
from src.data.load_data import raw_file_inventory  # noqa: E402
from src.data.make_dataset import build_interim_datasets, load_interim_datasets  # noqa: E402
from src.visualization.plots import (  # noqa: E402
    plot_class_distribution,
    plot_punctuation_by_class,
    plot_text_length_by_class,
    plot_top_domains,
)


def _name_datasets(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {str(frame["dataset"].iloc[0]): frame for frame in frames.values()}


def _concat(tables: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [table for table in tables if not table.empty]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


def _write(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8")


def _safe_slug(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").casefold()


def _near_duplicate_audit(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, frame in datasets.items():
        if frame["text"].fillna("").str.len().ge(80).sum() < 2:
            continue
        candidates = near_duplicate_candidates(frame, threshold=0.96)
        if candidates.empty:
            continue
        candidates.insert(0, "dataset", name)
        candidates["left_label"] = candidates["left_index"].map(frame["label_semantic"])
        candidates["right_label"] = candidates["right_index"].map(frame["label_semantic"])
        candidates["label_conflict"] = (
            candidates["left_label"].replace({"not_fake": "real"})
            != candidates["right_label"].replace({"not_fake": "real"})
        )
        rows.append(candidates)
    return _concat(rows)


def _period_by_label(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, frame in datasets.items():
        dated = frame.dropna(subset=["date_parsed", "label_semantic"])
        if dated.empty:
            continue
        table = dated.groupby("label_semantic")["date_parsed"].agg(["count", "min", "max"]).reset_index()
        table.insert(0, "dataset", name)
        rows.append(table)
    return _concat(rows)


def _dataset_comparison(
    overview: pd.DataFrame,
    missing: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> pd.DataFrame:
    missing_core = (
        missing[missing["column"].isin(["text", "raw_label", "url", "date_raw"])]
        .groupby("dataset")["missing_percent"]
        .mean()
        .rename("mean_core_missing_percent")
    )
    duplicate_text = (
        duplicates[duplicates["scope"].eq("text")]
        .set_index("dataset")["duplicated_rows"]
        .rename("duplicated_text_rows")
    )
    return overview.set_index("dataset").join([missing_core, duplicate_text]).reset_index()


def _report_markdown(tables: dict[str, pd.DataFrame]) -> str:
    overview = tables["overview"]
    labels = tables["labels"]
    duplicates = tables["duplicates"]
    cross = tables["cross_duplicates"]
    near = tables["near_duplicates"]
    risks = tables["risks"]
    periods = tables["periods"]
    text_summary = tables["text_summary"]
    associations = tables["associations"]
    missing = tables["missing"]

    total_records = int(overview["records"].sum())
    original = overview.loc[overview["dataset"].eq("fakerecogna/original")]
    standalone_records = int(original["records"].iloc[0]) if not original.empty else 0
    cross_conflicts = int(cross["label_conflict"].sum()) if not cross.empty else 0
    near_conflicts = int(near["label_conflict"].sum()) if not near.empty else 0

    high_risks = risks[risks["severity"].eq("alta")] if not risks.empty else risks
    top_risk_text = "; ".join(high_risks["risk"].head(4)) or "nenhum risco alto calculavel"

    label_lines = []
    for _, row in labels.iterrows():
        label_lines.append(
            f"- **{row['dataset']} / label bruto `{row['raw_label']}`:** "
            f"{int(row['count'])} registros ({row['percent']:.1f}%), interpretado como "
            f"`{row['label_semantic']}` conforme a documentacao da fonte."
        )

    missing_core = missing[missing["column"].isin(["text", "raw_label", "url", "date_raw"])]
    worst_missing = missing_core.sort_values("missing_percent", ascending=False).head(5)
    missing_lines = [
        f"- **Evidencia:** {row.dataset} tem {row.missing_percent:.1f}% de ausencia em `{row.column}`. "
        "**Interpretacao:** o campo nao esta uniformemente disponivel. "
        "**Impacto:** comparacoes e splits que dependam dele precisam excluir ou sinalizar esses casos."
        for row in worst_missing.itertuples()
    ]

    duplicate_lines = []
    for row in duplicates[duplicates["scope"].isin(["text", "url"])].itertuples():
        duplicate_lines.append(
            f"- **Evidencia:** {row.dataset} possui {row.duplicated_rows} linhas envolvidas em duplicacao de "
            f"`{row.scope}` ({row.duplicate_groups} grupos). **Interpretacao:** linhas nao sao unidades "
            "independentes. **Impacto:** um split aleatorio pode inflar as metricas."
        )

    temporal_lines = []
    for row in periods.itertuples():
        temporal_lines.append(
            f"- {row.dataset} / {row.label_semantic}: {row.count} datas validas, "
            f"de {row.min.date().isoformat()} a {row.max.date().isoformat()}."
        )

    association_lines = []
    if not associations.empty:
        pure = associations[associations["purity"].eq(1)].groupby(["dataset", "field"])["total"].sum()
        for (dataset, field), count in pure.sort_values(ascending=False).head(8).items():
            association_lines.append(
                f"- **Evidencia:** {count} registros de {dataset} pertencem a valores frequentes de `{field}` "
                "observados em uma unica classe. **Interpretacao:** o metadado funciona como proxy do label. "
                "**Impacto:** o modelo pode identificar origem/coleta em vez da veracidade do conteudo."
            )

    risk_table = risks.rename(
        columns={"risk": "Risco", "evidence": "Evidencia", "severity": "Severidade", "recommendation": "Recomendacao"}
    )

    return f"""# Relatorio de EDA — Noticias brasileiras

> Gerado de forma reprodutivel por `python -m src.analysis.pipeline`. Os dados brutos nao foram alterados. As tabelas completas estao em `reports/tables/`.

## Executive Summary

Foram perfilados {len(overview)} datasets/subconjuntos, totalizando {total_records:,} registros contados separadamente. Esse total **nao representa noticias unicas**, pois ha aliases e sobreposicoes entre fontes. O FakeRecogna independente possui {standalone_records:,} linhas, enquanto a versao incorporada ao FactChecks.br tem outra selecao e outro schema.

**Evidencia → interpretacao → impacto:** foram encontradas {len(cross):,} assinaturas exatas compartilhadas entre datasets ({cross_conflicts} com semantica de label conflitante apos equiparar `not_fake` a `real`) e {len(near):,} pares candidatos a quase duplicata dentro dos datasets ({near_conflicts} conflitos). Isso confirma que linha nao deve ser usada como unidade de sorteio; grupos de noticia/URL precisam permanecer no mesmo fold.

Os principais riscos classificados como altos foram: {top_risk_text}. A severidade segue um criterio operacional: **alta** quando um atalho pode afetar grande parte da base ou transferir a mesma noticia entre treino e teste; **media** quando a evidencia e localizada ou depende da estrategia de split; **baixa** quando o sinal existe, mas tem cobertura pequena. Severidade mede risco metodologico, nao qualidade moral da fonte.

## Datasets analisados

{overview.to_markdown(index=False)}

O FactChecks.br inclui tarefas diferentes: `fakebr` contem texto de alegacoes/noticias; `FakeRecogna` e `central_de_fatos` contem textos de revisao/fonte; `fact_check_tweet_pt` e `FakeNewsSet` contêm pares/IDs e URLs, sem texto integral no release. Portanto, nem todos servem diretamente para a mesma tarefa de classificacao textual.

## Qualidade dos dados

{chr(10).join(missing_lines)}

A linha sem label do FakeRecogna original e mantida na camada intermediaria, nao imputada. Datas que nao puderam ser interpretadas permanecem `NaT`, enquanto `date_raw` preserva o valor recebido.

## Distribuicao das classes

{chr(10).join(label_lines)}

As definicoes nao sao unificadas automaticamente. No FactChecks.br, `1 = fake`, `-1 = not_fake` e `0 = other`; no FakeRecogna original, `0 = fake` e `1 = real`. Alem disso, `central_de_fatos` e predominantemente uma colecao de checagens classificadas como falsas, enquanto outros corpora sao balanceados por construcao.

![Distribuicao de classes](figures/class_distribution.png)

## Caracteristicas textuais

{text_summary.to_markdown(index=False)}

As medianas e intervalos interquartis devem ser lidos por dataset: diferencas grandes entre classes podem refletir o processo de coleta (por exemplo, texto de checagem versus noticia publicada), nao uma propriedade universal da desinformacao. Os graficos por dataset em `reports/figures/` testam especificamente comprimento e pontuacao. Stopwords, caixa, stemming e lematizacao nao foram removidos nesta etapa, para que potenciais artefatos permaneçam visiveis.

## Analise de fontes

{chr(10).join(association_lines) if association_lines else '- Nao havia metadados suficientes para estimar associacoes.'}

Os graficos `*_domains.png` mostram a composicao das classes nos dominios mais frequentes. Uma fonte exclusivamente associada a uma classe e um forte atalho de coleta; esse campo nao deve entrar inadvertidamente no texto de treinamento.

## Duplicatas

{chr(10).join(duplicate_lines)}

Entre datasets, ha {len(cross):,} assinaturas de URL ou texto compartilhadas. A tabela `cross_dataset_duplicates.csv` e a lista diagnostica `near_duplicate_candidates.csv` devem alimentar um futuro `group_id`. Candidatos aproximados foram detectados com TF-IDF de n-gramas de caracteres e similaridade >= 0,96; eles exigem revisao antes de qualquer exclusao.

## Possiveis vieses

- **Vies de fonte:** noticias rotuladas como verdadeiras e falsas foram coletadas de tipos de portal distintos. O classificador pode aprender o dominio/estilo editorial.
- **Vies de categoria:** distribuicoes tematicas diferentes podem transformar assunto em proxy do label.
- **Vies temporal:** classes podem cobrir janelas distintas; mudancas de vocabulário e eventos tornam a data um atalho.
- **Vies de representacao:** os datasets representam fontes e agendas especificas, nao toda noticia brasileira nem toda nocao de confiabilidade.

## Possiveis data leakages

{risk_table.to_markdown(index=False) if not risk_table.empty else 'Nao foi possivel calcular riscos com os campos disponiveis.'}

Nunca executar `train_test_split()` por linha antes de criar grupos de duplicidade. URLs, autores, dominios e marcadores editoriais (como `#boato`) devem ser auditados com ablations.

## Comparacao entre datasets

{tables['comparison'].to_markdown(index=False)}

Tres caminhos devem permanecer em aberto:

**A. Usar separadamente.** Preserva a definicao e a tarefa de cada corpus; facilita diagnosticar desempenho intra-dominio, mas reduz diversidade e nao mede transferencia.

**B. Combinar subconjuntos compativeis.** Pode aumentar cobertura, desde que labels, unidade textual e duplicatas sejam harmonizados com regras versionadas. Nao se recomenda concatenacao direta dos cinco benchmarks.

**C. Treinar em um dataset e testar externamente em outro.** Mede generalizacao entre coletas e fontes, mas exige remover toda sobreposicao antes e interpretar divergencias de label/tarefa.

Nenhuma opcao foi escolhida automaticamente.

## Limitacoes

- Metadados ausentes nao foram inferidos da web.
- A deteccao de quase duplicatas e baseada em similaridade lexical e pode conter falsos positivos/negativos.
- `review_text` pode ser texto de checagem, enquanto `Noticia` no FakeRecogna disponibilizado ja aparenta preprocessamento linguistico; comparacoes estilisticas precisam considerar essa diferenca.
- A associacao estatistica entre fonte e label nao prova causalidade nem veracidade.
- O total agregado conta versoes sobrepostas do FakeRecogna e nao deve ser usado como tamanho de um corpus combinado.

## Recomendacoes para preparacao dos dados

1. Definir a unidade de predicao: alegacao, noticia original ou texto de checagem.
2. Criar `group_id` por URL canonica, hash de texto e cluster de quase duplicatas antes do split.
3. Revisar manualmente conflitos de label e documentar uma taxonomia; nunca mapear `other` para binario por conveniencia.
4. Comparar: split estratificado **apenas apos agrupamento**, group split por noticia/fonte, split temporal e dataset externo como holdout.
5. O split estratificado preserva proporcoes, mas nao resolve duplicatas; group split reduz vazamento por fonte/noticia; temporal split testa drift; holdout externo testa transferencia, desde que descontaminado.
6. Preservar texto original e registrar cada transformacao; avaliar normalizacao, stopwords e lematizacao somente via ablation.

## Proximos experimentos

- **Baseline 1 — TF-IDF + Logistic Regression:** rapido, interpretavel por coeficientes e capaz de produzir probabilidades que podem ser calibradas.
- **Baseline 2 — TF-IDF + Linear SVM:** baseline forte para alta dimensionalidade; margens nao devem ser tratadas como probabilidades sem calibracao.
- **Baseline 3 — Bag-of-Words + Logistic Regression:** ancora simples para medir o ganho real do TF-IDF.

Avaliar Accuracy, Precision, Recall, F1, Macro-F1, matriz de confusao, ROC-AUC e PR-AUC quando aplicaveis. **Accuracy nunca isoladamente.** Reportar resultados por fonte, periodo e dataset alem da media global.

`predict_proba()` representa a **probabilidade produzida pelo classificador sob seus dados e hipoteses**, nao a confiabilidade factual da noticia. Uma probabilidade alta pode refletir atalho, desbalanceamento ou erro de calibracao. Em etapa futura, estudar Brier Score, Log Loss, reliability diagrams, Platt Scaling e Isotonic Regression.

BERTimbau e outros Transformers devem ser avaliados apenas depois desses baselines e do protocolo de split sem leakage.
"""


def run_pipeline(*, run_near_duplicates: bool = True) -> dict[str, pd.DataFrame]:
    """Executa todas as analises sem modificar os dados brutos."""
    cfg = load_config()
    ensure_output_directories(cfg)
    build_interim_datasets()
    datasets = _name_datasets(load_interim_datasets())
    featured = {name: add_text_features(frame) for name, frame in datasets.items()}

    overview = dataset_overview(datasets)
    missing = _concat([missingness_table(frame, name) for name, frame in datasets.items()])
    labels = _concat([label_distribution(frame, name) for name, frame in datasets.items()])
    duplicates = _concat([duplicate_summary(frame, name) for name, frame in datasets.items()])
    cross = cross_dataset_duplicates(datasets)
    existing_near = Path(cfg["reports"]["tables"]) / "near_duplicates.csv"
    near = (
        _near_duplicate_audit(datasets)
        if run_near_duplicates
        else pd.read_csv(existing_near) if existing_near.is_file() else pd.DataFrame()
    )
    text_summary = _concat(
        [
            summarize_text_features(frame, name)
            for name, frame in featured.items()
            if frame["text"].fillna("").str.strip().ne("").sum() >= 10
        ]
    )
    periods = _period_by_label(datasets)

    sources = _concat(
        [top_values(frame, field, name) for name, frame in datasets.items() for field in ("domain", "author", "category")]
    )
    associations = _concat(
        [
            metadata_label_association(frame, field, name, min_count=5)
            for name, frame in datasets.items()
            for field in ("domain", "author", "category")
        ]
    )
    risks = leakage_risk_table(datasets, cross)
    comparison = _dataset_comparison(overview, missing, duplicates)

    textual = {
        name: frame
        for name, frame in featured.items()
        if frame["text"].fillna("").str.strip().ne("").sum() >= 10
    }
    vocabulary_frames: list[pd.DataFrame] = []
    distinctive_frames: list[pd.DataFrame] = []
    for name, frame in textual.items():
        for method, ngram in (("count", (1, 1)), ("count", (2, 2)), ("tfidf", (1, 2))):
            table = top_terms_by_class(frame, method=method, ngram_range=ngram)
            table.insert(0, "dataset", name)
            vocabulary_frames.append(table)
        distinct = class_distinctive_terms(frame)
        distinct.insert(0, "dataset", name)
        distinctive_frames.append(distinct)
    vocabulary = _concat(vocabulary_frames)
    distinctive = _concat(distinctive_frames)

    tables = {
        "overview": overview,
        "missing": missing,
        "labels": labels,
        "duplicates": duplicates,
        "cross_duplicates": cross,
        "near_duplicates": near,
        "text_summary": text_summary,
        "periods": periods,
        "sources": sources,
        "associations": associations,
        "risks": risks,
        "comparison": comparison,
        "vocabulary": vocabulary,
        "distinctive_terms": distinctive,
        "raw_manifest": raw_file_inventory(),
    }

    table_dir = Path(cfg["reports"]["tables"])
    for key, table in tables.items():
        _write(table, table_dir / f"{key}.csv")

    figure_dir = Path(cfg["reports"]["figures"])
    plot_class_distribution(labels, figure_dir / "class_distribution.png")
    for name, frame in textual.items():
        slug = _safe_slug(name)
        plot_text_length_by_class(frame, figure_dir / f"{slug}_text_length.png")
        plot_punctuation_by_class(frame, figure_dir / f"{slug}_punctuation.png")
        if frame["domain"].astype("string").str.strip().ne("").sum() >= 10:
            plot_top_domains(frame, figure_dir / f"{slug}_domains.png")

    Path(cfg["reports"]["summary"]).write_text(_report_markdown(tables), encoding="utf-8")
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-near-duplicates",
        action="store_true",
        help="Pula a etapa mais custosa; duplicatas exatas continuam sendo auditadas.",
    )
    args = parser.parse_args()
    tables = run_pipeline(run_near_duplicates=not args.skip_near_duplicates)
    print(tables["overview"].to_string(index=False))


if __name__ == "__main__":
    main()
