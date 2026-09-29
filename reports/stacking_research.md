# Fundamentacao tecnica — Ensemble Stacking

## Stacked Generalization

Stacking aprende uma segunda funcao sobre as saidas de modelos-base. No nivel 0,
cada base learner aprende a tarefa original; no nivel 1, o meta-learner aprende
como combinar seus erros e vieses. O artigo fundador de Wolpert descreve essa
segunda representacao como um espaco formado pelas predicoes feitas sobre partes
nao usadas no ajuste do respectivo modelo-base.

O `StackingClassifier` do scikit-learn ajusta os estimadores-base no conjunto
completo, mas ajusta o estimador final sobre predicoes obtidas por
`cross_val_predict`. O argumento `cv` controla a geracao dessas features OOF; ele
nao e uma avaliacao do ensemble. A opcao `cv="prefit"` usa predicoes dos modelos
ja ajustados sobre os mesmos dados e apresenta alto risco de overfitting.

Neste projeto, a implementacao e manual: o `StackingClassifier` nao expressa
diretamente o requisito de manter URL, texto repetido e quase duplicatas no mesmo
fold em todas as camadas. Cinco folds de `StratifiedGroupKFold` geram exatamente
uma predicao OOF para cada noticia de desenvolvimento. O teste isolado nunca gera
features do meta-learner nem participa de ablation/tuning.

## Bagging, boosting e stacking

- **Bagging:** ajusta copias do mesmo tipo de modelo em amostras bootstrap e
  agrega voto/media; reduz principalmente variancia de aprendizes instaveis.
- **Boosting:** ajusta aprendizes sequencialmente, dando mais foco aos erros ou
  ao gradiente residual; tenta converter aprendizes fracos em um forte.
- **Stacking:** pode combinar familias heterogeneas e aprende a combinacao por um
  meta-modelo. O ganho depende de erros complementares, nao apenas de bons scores
  individuais.

Aqui, Logistic Regression, Linear SVM e Multinomial NB usam representacoes
TF-IDF ajustadas exclusivamente dentro de cada fold. O meta-learner e Logistic
Regression. LR e NB fornecem `predict_proba`; o SVM recebe calibracao sigmoid
interna por grupos. Assim, as tres entradas do nivel 1 estao em `[0, 1]` e têm a
mesma orientacao (`P(fake)`).

## Validacao, leakage e calibracao

`StratifiedKFold` preserva aproximadamente a proporcao das classes, mas nao
impede copias da mesma noticia em folds diferentes. `StratifiedGroupKFold`
adiciona essa restricao de grupos. Para tempo, uma avaliacao separada treina em
datas ate 2021-02-19 e avalia datas posteriores; grupos que atravessam o corte
ficam integralmente no futuro. `TimeSeriesSplit` e apropriado para observacoes
ordenadas e igualmente espacadas, o que nao descreve bem este corpus editorial.

`CalibratedClassifierCV` ajusta classificador e calibrador por validacao cruzada.
Foi usada calibracao sigmoid para o Linear SVM; isotonic nao foi escolhido porque
e mais propenso a overfitting com amostras pequenas por fold. Brier Score, Log
Loss e reliability curves medem a qualidade probabilistica alem de F1/AUC.

## Vantagens e limites para texto

TF-IDF linear oferece diversidade barata: LR otimiza log-loss, SVM a margem e NB
uma hipotese generativa sobre frequencias. Stacking pode explorar desacordos sem
concatenar vetores esparsos enormes. Em contrapartida, multiplica tempo/memoria,
pode aprender ruido do OOF, complica calibracao e nao corrige vieses comuns a
todos os modelos. Se todos exploram marcadores de fonte/coleta, o ensemble apenas
reforca o atalho. Por isso, a decisao final compara teste, calibracao, custo,
avaliacao temporal e ablations; nao presume superioridade do E04.

## Referencias

- [Scikit-learn — StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [Scikit-learn — Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Scikit-learn — Cross-validation e StratifiedGroupKFold](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Scikit-learn — Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Wolpert (1992), Stacked Generalization](https://doi.org/10.1016/S0893-6080(05)80023-1)
- [Breiman (1996), Bagging Predictors](https://doi.org/10.1007/BF00058655)
- [Freund e Schapire (1997), boosting](https://doi.org/10.1006/jcss.1997.1504)
