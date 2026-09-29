# Decisão técnica

## Candidato escolhido: E02

<div class="decision">
  <strong>TF-IDF + Linear SVM com calibração sigmoid</strong><br>
  Selecionado para a próxima etapa por desempenho, calibração, estabilidade
  temporal e simplicidade operacional.
</div>

## Critérios considerados

| Critério | E02 — SVM | E04 — Stacking | Melhor escolha |
|---|---:|---:|---|
| Macro-F1 | **0,9876** | 0,9852 | E02 |
| ROC-AUC | **0,9982** | 0,9979 | E02 |
| Brier Score ↓ | **0,0108** | 0,0112 | E02 |
| Log Loss ↓ | **0,0460** | 0,0483 | E02 |
| Erros no teste | **21** | 25 | E02 |
| Ajuste completo | **19,71 s** | 82,28 s | E02 |
| Componentes em inferência | **1 pipeline** | 2 bases + meta | E02 |

## Por que não escolher apenas pela Accuracy?

Accuracy pode esconder comportamento desigual entre classes e não mede a
qualidade das probabilidades. A decisão combinou:

- Macro-F1 para dar peso igual às classes;
- Recall da classe `fake`;
- ROC-AUC e PR-AUC para ordenação;
- Brier e Log Loss para qualidade probabilística;
- teste temporal para drift;
- custo de treinamento, tamanho e manutenção;
- análise de erros e ablações.

## Por que o stacking não venceu?

O NB trouxe diversidade, mas o meta-modelo permaneceu fortemente dependente do
SVM. A combinação reduziu alguns erros e introduziu outros. Adicionar Logistic
Regression também não melhorou a validação, indicando redundância entre os
classificadores lineares.

!!! success "Decisão"
    Manter E02 como candidato de integração e E04 como benchmark experimental.
    Reavaliar stacking somente se uma nova base — por exemplo, BERTimbau —
    produzir erros genuinamente complementares por meio de previsões OOF.

## Condições antes de produção

1. Criar holdout externo com fontes não observadas no treinamento.
2. Avaliar notícias posteriores a 2021 e monitorar drift.
3. Medir desempenho por fonte, tema, comprimento e período.
4. Definir política de baixa confiança e possível abstenção.
5. Separar classificação estatística da futura recuperação de evidências.
6. Versionar dados e modelos em storage/DVC, não diretamente no Git.
