# Classificação de notícias em português brasileiro

<div class="hero">
  <h2>Do diagnóstico dos dados à escolha do modelo</h2>
  <p>
    Esta documentação apresenta o pipeline experimental, os modelos avaliados,
    o stacking sem vazamento e os critérios que levaram à escolha do Linear SVM
    calibrado como candidato para a próxima etapa.
  </p>
</div>

<div class="metric-grid">
  <div class="metric-card"><strong>11.858</strong>registros modelados</div>
  <div class="metric-card"><strong>11.615</strong>grupos independentes</div>
  <div class="metric-card"><strong>0,9876</strong>Macro-F1 do candidato</div>
  <div class="metric-card"><strong>0,9815</strong>Macro-F1 temporal</div>
</div>

## Objetivo

O projeto investiga classificação supervisionada de notícias em português
brasileiro nas classes `fake` e `real`. A implementação cobre:

- auditoria e preparação dos datasets;
- prevenção de vazamento por duplicatas e notícias relacionadas;
- três baselines de Machine Learning tradicional;
- calibração probabilística do Linear SVM;
- stacking manual com previsões Out-of-Fold;
- avaliação temporal, comparação de erros e persistência dos modelos;
- interface de inferência preparada para futura integração com FastAPI.

!!! warning "Classificação não é checagem factual"
    A probabilidade representa pertencimento à classe aprendida no corpus. O
    sistema não recupera evidências, não consulta fontes atuais e não comprova a
    veracidade factual de uma notícia.

## Resultado principal

O experimento **E02 — TF-IDF + Linear SVM calibrado** foi selecionado porque
apresentou o melhor equilíbrio entre desempenho, calibração, estabilidade
temporal e custo operacional.

| Experimento | Modelo | Macro-F1 | Brier | Situação |
|---|---|---:|---:|---|
| E01 | Logistic Regression | 0,9752 | 0,0271 | baseline |
| E02 | Linear SVM calibrado | **0,9876** | **0,0108** | **selecionado** |
| E03 | Multinomial Naive Bayes | 0,9522 | 0,0362 | complementar |
| E04 | Stacking OOF | 0,9852 | 0,0112 | referência |

[Entender a decisão técnica](decisao.md){ .md-button .md-button--primary }
[Explorar todos os resultados](resultados.md){ .md-button }

## Navegação sugerida

1. Comece pela [origem e qualidade dos dados](dados.md).
2. Entenda o [protocolo contra data leakage](metodologia.md).
3. Compare os [modelos individuais e o stacking](modelos/index.md).
4. Consulte os [resultados quantitativos](resultados.md).
5. Veja a [decisão e seus trade-offs](decisao.md).
