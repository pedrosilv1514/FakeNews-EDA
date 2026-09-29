# Dados e diagnóstico da EDA

## Fontes disponíveis

A análise exploratória encontrou seis datasets ou subconjuntos, totalizando
42.591 linhas quando contados separadamente. Esse total não representa notícias
únicas: existem aliases, versões sobrepostas e tarefas diferentes.

| Dataset | Registros | Uso na modelagem |
|---|---:|---|
| FactChecks/FakeNewsSet | 598 | excluído: sem texto integral |
| FactChecks/FakeRecogna | 11.773 | excluído: sobreposição com a versão original |
| FactChecks/central_de_fatos | 10.461 | excluído: tarefa distinta e 98,3% `fake` |
| FactChecks/fact_check_tweet_pt | 656 | excluído: sem texto integral |
| FactChecks/fakebr | 7.200 | mantido na EDA, não combinado por diferenças de coleta |
| FakeRecogna original | 11.903 | corpus canônico dos experimentos |

![Distribuição das classes](assets/generated/class_distribution.png)

## Por que usar o FakeRecogna original?

Ele oferece título, subtítulo, corpo, URL, data e rótulo binário balanceado. A
decisão de não concatenar todos os datasets evita misturar:

- notícia original com texto de checagem;
- tarefas binárias e ternárias;
- datasets sem conteúdo textual;
- a mesma notícia presente em fontes diferentes;
- definições de rótulo que não são semanticamente equivalentes.

## Auditoria aplicada

| Etapa | Resultado |
|---|---:|
| Linhas iniciais | 11.903 |
| Rótulos ausentes/não binários removidos | 1 |
| Arestas de quase duplicidade incorporadas | 76 |
| Linhas em grupos com conflito de rótulo removidas | 44 |
| Linhas finais | 11.858 |
| Grupos finais | 11.615 |

URL normalizada, hash do texto, hash do corpo e candidatos com similaridade
lexical maior ou igual a 0,96 formam componentes de notícias relacionadas. Um
componente nunca é dividido entre treino e avaliação.

## Riscos identificados

!!! danger "Atalhos do corpus"
    Fonte, autor, domínio, período, comprimento e estilo editorial estão
    associados à classe. Um modelo pode aprender o processo de coleta em vez de
    uma propriedade universal da desinformação.

O corpo distribuído pelo FakeRecogna já aparenta pré-processamento linguístico.
Não é possível recuperar integralmente o texto original a partir dele, o que
limita a transferência para notícias atuais sem esse tratamento.
