# Limitações e uso responsável

## O que o modelo faz

O classificador estima se um texto se parece mais com exemplos rotulados como
`fake` ou `real` no FakeRecogna. Ele identifica padrões lexicais e estilísticos
aprendidos durante o treinamento.

## O que o modelo não faz

- não pesquisa fontes atuais;
- não recupera evidências;
- não verifica datas, pessoas ou eventos;
- não garante que uma notícia classificada como `real` seja factual;
- não produz justificativa factual;
- não substitui checagem humana ou jornalística.

## Viés de coleta

Notícias falsas e reais foram coletadas de tipos diferentes de portal. Domínio,
autor, período, comprimento e estilo podem atuar como proxies do rótulo. Mesmo
sem fornecer esses metadados diretamente, marcas editoriais podem permanecer no
texto.

## Mudança temporal

O corpus termina em 2021. Vocabulário, eventos e padrões de desinformação mudam.
O bom teste temporal interno não elimina a necessidade de avaliar dados atuais.

## Texto pré-processado na origem

O corpo do FakeRecogna distribuído aparenta stemming ou normalização linguística
prévia. Usuários futuros fornecerão texto natural, criando possível diferença de
distribuição entre treinamento e produção.

## Interpretação da probabilidade

`probability_fake = 0.90` significa que o classificador atribuiu 90% de massa à
classe `fake` sob o corpus, modelo e calibração atuais. Não significa “90% de
chance de a afirmação ser falsa” no sentido factual.

!!! danger "Uso de alto impacto"
    O modelo não deve decidir sozinho remoção de conteúdo, punição de usuários,
    reputação de fontes ou qualquer ação de alto impacto.
