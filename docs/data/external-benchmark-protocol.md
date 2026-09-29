# Protocolo do benchmark externo v1

## Estado e finalidade

| Campo | Valor |
|---|---|
| Identificador | `external-v1` |
| Versão do protocolo | `1.0.0-draft` |
| Estado | protocolo definido; dados ainda não coletados |
| Período elegível | 1º de janeiro de 2022 a 31 de dezembro de 2025 |
| Uso | teste externo congelado |
| Baseline a avaliar | E02 — TF-IDF + Linear SVM calibrado, versão 1.0.0 |

O `external-v1` medirá generalização temporal, editorial e temática fora do
FakeRecogna. Ele não será usado para ajustar vocabulário, hiperparâmetros,
threshold, calibração ou regras de pré-processamento.

O intervalo termina em 2025 para usar anos civis completos e reservar notícias
publicadas a partir de 1º de janeiro de 2026 para um futuro benchmark prospectivo
`external-v2`. O corpus atual termina em 7 de dezembro de 2021; portanto, não há
sobreposição temporal intencional.

!!! warning "Benchmark ainda não existe"
    Este documento define o protocolo antes da coleta. Quantidades, métricas e
    distribuições só poderão ser apresentadas como resultados após a construção,
    revisão e congelamento do conjunto na Etapa 4.

## Pergunta experimental

> O artefato E02, treinado com o FakeRecogna, mantém discriminação, calibração e
> latência aceitáveis em textos jornalísticos em português posteriores a 2021,
> provenientes de veículos não observados no treinamento?

O benchmark avalia **classificação estatística de documentos**. Ele não mede um
sistema completo de verificação factual, recuperação de evidências ou reputação
de veículos.

## Unidade de análise

A unidade primária é um documento em português que apresenta uma alegação
factual verificável. O documento-alvo deve ser o conteúdo que seria submetido ao
classificador — nunca o texto escrito pela agência de checagem para explicar o
veredito.

Cada registro deve possuir três níveis de identidade:

- `article_id`: documento específico que será classificado;
- `claim_id`: alegação factual canônica;
- `event_id`: acontecimento ao qual uma ou mais alegações pertencem.

Mais de um documento pode representar a mesma alegação, limitado a três por
`claim_id` e cinco por `event_id`. Esses limites evitam que um boato muito
republicado domine as métricas. As métricas primárias serão calculadas por
documento, com análise de sensibilidade por alegação e reamostragem por evento.

## População-alvo

O escopo primário inclui notícias e textos informativos em português brasileiro
que um usuário poderia submeter ao futuro Hub de Notícias. O conjunto deve conter
textos brutos, antes de stemming ou lematização, para medir a diferença de
distribuição em relação ao corpus atual.

Conteúdo de redes sociais pode ser mantido em uma partição exploratória, mas não
entra na métrica primária da versão 1, pois o modelo atual foi desenvolvido para
título e corpo de notícia.

## Fontes e independência editorial

As fontes de descoberta e evidência podem incluir APIs públicas, marcação
estruturada [`ClaimReview`](https://schema.org/ClaimReview), RSS e registros de
organizações de checagem. A Collection v1 definirá conectores e políticas de
acesso; este protocolo não autoriza scraping por si só.

Há diferença entre:

- **veículo do conteúdo:** publicou o documento que será classificado;
- **organização verificadora:** publicou a checagem e o rótulo;
- **fonte de evidência:** sustenta ou contradiz a alegação.

Para a análise externa primária, o domínio canônico do veículo do conteúdo deve
estar ausente do inventário do treinamento. A presença da organização
verificadora no material de EDA não torna o veículo do conteúdo conhecido, desde
que o texto da checagem não seja fornecido ao modelo.

Registros de fontes já vistas podem ser preservados com `source_seen=true`, mas
ficam fora da métrica externa primária e entram apenas em análise secundária.
Fontes sem domínio ou identidade editorial verificável são excluídas.

## Critérios de inclusão

Um registro só poderá integrar o benchmark congelado quando satisfizer todos os
critérios:

1. publicação entre `2022-01-01` e `2025-12-31` em UTC;
2. idioma português confirmado pelo coletor e por revisão humana;
3. título e/ou corpo disponíveis, com corpo normalizado de pelo menos 200
   caracteres;
4. presença de alegação factual passível de confirmação ou refutação;
5. URL canônica, veículo e data de publicação identificados;
6. rótulo original, escala de avaliação e organização verificadora preservados;
7. pelo menos uma URL de evidência acessível ou referência primária identificada;
8. licença, termos ou base de armazenamento registrados;
9. `claim_id` e `event_id` atribuídos e revisados;
10. ausência de sobreposição proibida com os dados de desenvolvimento;
11. concordância de dois revisores ou decisão de um adjudicador.

Quando o texto completo não puder ser armazenado ou redistribuído, o registro
pode ser mantido em storage controlado. O Git receberá apenas código,
configuração, manifesto, hashes e estatísticas permitidas.

## Critérios de exclusão

São excluídos da avaliação binária primária:

- texto da própria checagem usado como se fosse a notícia avaliada;
- páginas sem corpo, listas de links ou conteúdo puramente audiovisual sem
  transcrição confiável;
- opinião, previsão, sátira explicitamente identificada ou conteúdo sem alegação
  factual verificável;
- tradução automática sem versão original rastreável;
- rótulo inferido apenas pela reputação do domínio;
- registro cujo único fundamento seja outro dataset sem evidência rastreável;
- texto duplicado ou semanticamente equivalente a item de treino;
- conflito de rótulo sem adjudicação;
- fonte ou data não verificável;
- `unverified` ou `insufficient_evidence` convertido artificialmente em `false`.

Registros ambíguos são preservados fora da avaliação binária, não descartados
silenciosamente.

## Taxonomia de rótulos

O rótulo fornecido pela origem permanece em `label_original`. A normalização
utiliza a seguinte taxonomia:

| Rótulo normalizado | Definição operacional | Classe binária |
|---|---|---|
| `true` | Alegação sustentada pelas evidências no contexto declarado | `real` |
| `false` | Alegação contradita pelas evidências | `fake` |
| `misleading` | Elementos verdadeiros apresentados de forma capaz de induzir conclusão incorreta | excluída |
| `out_of_context` | Conteúdo autêntico reutilizado em contexto incompatível | excluída |
| `outdated` | Já foi válido, mas está incorreto no contexto temporal avaliado | excluída |
| `unverified` | Não passou por verificação suficiente | excluída |
| `insufficient_evidence` | Evidência disponível não permite concluir | excluída |

A métrica binária primária inclui somente `true → real` e `false → fake`. Os
demais rótulos formam um conjunto de desafio para futura política de abstenção.
Não serão usados para ajustar o threshold do `external-v1`.

## Fundamentação do rótulo

Um rótulo elegível exige uma das rotas abaixo:

1. avaliação explícita de organização cadastrada no registro de fontes, com
   escala original, URL da checagem e evidências preservadas; ou
2. duas fontes primárias independentes, registradas por revisores, que sustentem
   a mesma conclusão.

Uma republicação da mesma nota não conta como fonte independente. Para `true`, a
ausência de refutação não é evidência de veracidade. Para `false`, o domínio ou o
estilo do texto nunca são fundamento suficiente.

## Processo de anotação e revisão humana

1. O coletor propõe metadados e, quando existir, o rótulo da fonte.
2. Dois revisores avaliam independentemente documento, alegação, contexto,
   evidências, rótulo, `claim_id` e `event_id`.
3. Divergências são resolvidas por um terceiro adjudicador.
4. O registro recebe `label_status=adjudicated` somente após a decisão final.
5. O benchmark congelado aceita apenas registros `adjudicated`.

Os revisores devem registrar motivo estruturado, referências consultadas e
conflitos. O relatório de qualidade publicará concordância bruta e Cohen's kappa
antes da adjudicação, sem transformar concordância em prova de correção factual.

## Cobertura e amostragem

O alvo de construção é 1.200 registros binários, 600 por classe. A versão só pode
ser congelada com no mínimo 800 registros, pelo menos 350 por classe.

Para evitar um benchmark dominado pelo processo de coleta:

- no mínimo 10 veículos de conteúdo na avaliação primária;
- nenhum veículo pode representar mais de 10% das linhas;
- nenhum tópico pode representar mais de 30% das linhas;
- pelo menos cinco tópicos devem possuir 50 ou mais exemplos;
- nenhum ano pode representar mais de 35% das linhas;
- cada ano entre 2022 e 2025 deve representar pelo menos 10%;
- no máximo três documentos por `claim_id`;
- no máximo cinco documentos por `event_id`.

Os tópicos controlados são: política, saúde, economia, ciência e tecnologia,
segurança, meio ambiente, desastres, internacional, entretenimento/esportes e
outros. Amostragem balanceada permite comparar classes, mas não estima a
prevalência real de desinformação na população.

## Deduplicação e prevenção de vazamento

A descontaminação compara candidatos com todos os dados `raw`, `interim` e
`processed` usados na EDA ou treinamento, além dos demais itens externos.

### Bloqueios automáticos

- mesma URL canônica;
- mesmo SHA-256 do texto normalizado;
- mesmo SHA-256 de título + corpo;
- `claim_id` ou `event_id` presente em treino, validação ou teste internos.

### Candidatos para revisão

- similaridade de cosseno de TF-IDF de n-gramas de caracteres maior ou igual a
  0,90;
- Jaccard/MinHash estimado maior ou igual a 0,80;
- vizinhança semântica acima do limiar calibrado em uma amostra anotada.

Similaridade semântica nunca remove um item automaticamente. Pares candidatos
são revisados, recebem `duplicate_decision` e mantêm trilha de auditoria. Se dois
textos discutirem a mesma alegação ou acontecimento, compartilham grupo mesmo
quando a redação for diferente.

Antes do congelamento devem ser verdadeiras as seguintes invariantes:

```text
interseção(article_id externo, treino/validação/teste) = 0
interseção(claim_id externo, treino/validação/teste) = 0
interseção(event_id externo, treino/validação/teste) = 0
interseção(content_hash externo, treino/validação/teste) = 0
```

## Esquema obrigatório

O registro externo deve conter, no mínimo:

```text
article_id
claim_id
event_id
title
body
canonical_url
content_publisher
content_publisher_domain
published_at
collected_at
language
topic
label_original
label_normalized
binary_label
label_status
rating_scale
factcheck_organization
factcheck_url
evidence_urls
source_seen
license_or_terms
content_hash
collector_version
annotation_protocol_version
```

Campos de texto protegidos podem ser substituídos por referência ao storage no
manifesto público, mas devem continuar disponíveis no ambiente controlado de
avaliação.

## Congelamento e versionamento

O benchmark passa por quatro estados:

```text
collecting → labeling → quality_review → frozen
```

Depois de `frozen`:

- conteúdo, IDs, rótulos e grupos não podem ser alterados silenciosamente;
- correções exigem nova versão e changelog;
- o manifesto recebe SHA-256 do arquivo, contagens e versão de dependências;
- o acesso ao rótulo durante inferência deve ser separado da geração das
  previsões;
- as previsões brutas de cada candidato são preservadas.

O manifesto futuro será salvo em `data/manifests/external-v1.json`. O dataset
completo permanecerá fora do Git e deverá ser versionado por DVC ou storage com
controle de acesso.

Se `external-v1` influenciar alterações de features, threshold, calibração ou
hiperparâmetros, ele passa a ser considerado **consumido para seleção**. Uma nova
aceitação de produção exigirá o `external-v2`, formado por notícias de 2026 em
diante.

## Protocolo de avaliação

### Regras fixas

- carregar o artefato E02 sem retreinamento;
- usar o pré-processamento incorporado ao artefato;
- manter threshold 0,5;
- não recalibrar no conjunto externo;
- gerar todas as previsões antes de abrir os rótulos para análise;
- registrar commit, hash do modelo, ambiente, hardware e duração;
- preservar falhas de inferência como erros auditáveis, não removê-las depois de
  observar o rótulo.

Modelos futuros devem ter configuração e artefato congelados antes de receberem
as métricas externas. Cada candidato tem uma execução confirmatória por versão.

### Métricas primárias

- Macro-F1;
- recall da classe `false/fake`;
- Brier Score.

### Métricas secundárias

- Accuracy, Precision, Recall e F1 da classe positiva;
- F1 por classe;
- ROC-AUC e PR-AUC;
- Log Loss;
- matriz de confusão;
- classification report;
- Expected Calibration Error com 15 bins de quantis;
- latência de inferência por documento.

Intervalos de confiança de 95% serão estimados com 2.000 reamostragens bootstrap
por `event_id`, seed 42. A unidade de reamostragem é o evento, não a linha.

A latência será medida após 10 warm-ups, em 100 execuções com lote 1 e lote 32.
Devem ser publicados mediana, p95, hardware, sistema operacional, versão do
Python e quantidade de threads.

### Segmentação

As métricas também serão calculadas por:

- veículo;
- tópico;
- ano e trimestre;
- faixa de comprimento;
- rota de fundamentação do rótulo;
- fonte vista versus inédita, apenas como análise secundária.

Uma métrica segmentada só será publicada com pelo menos 30 registros e 10 itens
de cada classe. Segmentos menores serão apresentados apenas como contagem, para
evitar conclusões instáveis.

## Critérios de decisão

As regras são avaliadas pela condição mais severa encontrada.

### `approve`

Todos os critérios devem ser satisfeitos:

- Macro-F1 maior ou igual a 0,80;
- limite inferior do IC 95% do Macro-F1 maior ou igual a 0,75;
- recall de `fake` maior ou igual a 0,85;
- Brier Score menor ou igual a 0,15;
- nenhum segmento publicável com Macro-F1 abaixo de 0,70;
- diferença máxima de Macro-F1 entre segmentos comparáveis de 0,15;
- nenhuma falha metodológica ou contaminação detectada.

### `approve_with_restrictions`

As métricas globais atendem aos limites de `approve`, mas existe degradação
localizada por fonte, tópico, período ou comprimento. A aprovação deve declarar
explicitamente os segmentos não suportados e ativar política de abstenção.

### `retrain_required`

Qualquer uma das condições abaixo é suficiente:

- Macro-F1 entre 0,65 e 0,80;
- recall de `fake` entre 0,70 e 0,85;
- Brier Score entre 0,15 e 0,25;
- queda de Macro-F1 superior a 0,10 em relação ao teste interno;
- incompatibilidade relevante entre texto bruto externo e pré-processamento do
  corpus de treinamento.

### `reject_for_production`

Qualquer uma das condições abaixo é suficiente:

- Macro-F1 abaixo de 0,65;
- recall de `fake` abaixo de 0,70;
- Brier Score acima de 0,25;
- contaminação do benchmark;
- proveniência ou qualidade dos rótulos insuficiente;
- falha sistemática em segmento relevante do produto.

Esses limites são critérios mínimos de engenharia definidos antes da coleta;
eles não constituem garantia de segurança ou adequação para decisões de alto
impacto.

## Relatório de qualidade obrigatório

Antes do congelamento, a Etapa 4 deverá publicar:

- quantidade total e elegível para avaliação binária;
- distribuição por rótulo original e normalizado;
- distribuição por fonte, tópico, ano e trimestre;
- ausências por coluna;
- conflitos e adjudicações;
- concordância entre revisores;
- duplicatas removidas e candidatos revisados;
- sobreposição com todos os datasets existentes;
- cobertura das cotas;
- licenças e restrições de redistribuição;
- hashes e versões de coleta/anotação.

Se qualquer critério mínimo de tamanho, classe, fonte, período ou qualidade não
for alcançado, o benchmark permanece em `quality_review` e não pode ser usado
para decidir a promoção do modelo.

## Separação entre etapas

- **Etapa 2:** implementará a arquitetura de coleta com fixtures locais.
- **Etapa 3:** adicionará um conector real e coleta pequena, ainda sem treinar.
- **Etapa 4:** construirá, revisará e congelará o `external-v1`.
- **Etapa 5:** avaliará o E02 uma única vez conforme este protocolo.

Qualquer mudança neste documento depois de iniciada a avaliação deve incrementar
a versão do protocolo e ser registrada antes de uma nova execução.
