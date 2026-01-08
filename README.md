# Case COVID-19

Este repositório guarda o Jupyter Notebook que fiz para um case de estudos de um processo seletivo que participei quando estava trabalhando no mercado financeiro. Nessa etapa do processo, nos foi apresentado um case de estudos que teríamos que desenvolver em Python.
Utilizando um conjunto de dados de saúde da pandemia de COVID-19 do governo do estado de São Paulo, teríamos que responder três perguntas e desenvolver um modelo que fosse capaz de calcular a probabilidade de uma pessoa vir a óbito pela infecção de COVID-19 dado o seu gênero, sua idade e doenças pré-existentes. Deixou-se livre para nós escolhermos o modelo que iríamos criar para atingir o objetivo e as bibliotecas em Python que iríamos utilizar.

Para responder as perguntas, utilizei bibliotecas simples de análise de dados que utilizava todos os dias em meu trabalho: pandas, numpy e matplotlib. Na época, eu não tinha conhecimento de modelos de Aprendizado de Máquina, então pensei no modelo estatístico mais fundamental que vi nas aulas de probabilidades em minha graduação: o modelo de Bayes. Eu sabia que com ele eu conseguiria calcular a probabilidade de uma certa asserção dada hipóteses conhecidas e conhecimentos prévios de um conjunto de dados, então bastou eu pensar em como implementar e aplicar o modelo para o caso específico apresentado. 

Irei descrever brevemente o arcabouço teórico que sustentou minhas escolhas na hora da implementação do modelo de Bayes.

No Notebook, pode-se encontrar os códigos feitos por mim para as tarefas, bem como algumas conclusões que tirei dos resultados.

***

## Modelo de Bayes

O modelo de Bayes pode ser inferido da lei de probabilidades totais, e sua expressão geralmente é escrita da seguinte forma:

$P(X|Y) = \frac{P(Y|X)}{P(Y)} P(X) $.

Na visão bayesiana, a probabilidade mede nosso *grau de crença*, então nesse sentido a expressão acima codifica matematicamente o fenômeno de "aprendizado", pois  probabilidade P(X) e P(Y) representa nossa crença de que uma dada asserção X e Y seja verdadeira e P(X|Y) ou P(Y|X) representa nossa crença de que tal asserção seja verdadeira dado um conhecimento prévio da asserção à direita da barra. É como se a expressão acima fosse uma "atualização" de nossa crença X agora que temos dados da crença de Y: essa crença atual P(X|Y) seria proporcional à nossa crença "a priori" P(X), e a constante de proporcionalidade $\frac{P(Y|X)}{P(Y)}$ seria dada por crenças "a priori" da informação obtida Y. Cada expressão individual é conhecida por um nome, sendo P(Y|X) conhecida como a função de verossimilhança (likelihood function), P(Y) a probabilidade dos meus dados e P(X) minha probabilidade a priori.

Para aplicar o modelo de Bayes no nosso caso de estudo, pense em X como a asserção "morrer pela infecção de COVID-19" e em Y como os dados que devemos colocar como input. Dessa forma, a expressão do modelo de Bayes representaria a seguinte frase: "A probabiliadde de morrer pela infecção de COVID-19 dado meu gênero, idade e doenças pré-existentes (P(X|Y)) é igual à probabilidade de ter minha idade, gênero e doenças pré-existentes e ter morrido por COVID-19 (P(Y|X)) vezes a probabilidade de morrer de COVID-19 (P(X)) dividido pela probabilidade de ter minha idade, gênero e doenças pré-existentes (P(Y))"

***

## Eventos independentes e probabilidade condicional

Em nosso estudo de caso, entretanto, ao invés de uma asserção simples como informação, temos diversas asserções. Queremos saber qual a probabilidade de uma pessoa vir a óbito por COVID-19 dado sua idade **E** seu gênero **E** se tem comorbidade 1 **E** se tem comorbidade 2 e por aí vai. Na tradução para a linguagem matemática, significa que nosso conhecimento que irá atualizar P(X) não será um conjunto Y, mas a intersecção de vários subconjuntos $Y_i$ do meu espaço amostral, e ao analisar essas probabilidades condicionadas por outras asserções, é importante sabermos se os eventos são independentes ou não.

Definimos eventos independentes como eventos os quais não afetam a probabilidade de um evento posterior. A jogada múltipla de um dado não enviesado é um exemplo clássico de eventos independentes, pois o resultado da primeira jogada não afeta o resultado da jogada seguinte, então a probabilidade de se obter um 6 na segunda jogada continua sendo 1/6, mesmo sabendo que na primeira jogada obtivemos um 4. Por conclusão, podemos definir a probabilidade condicional de dois eventos independentes como sendo a probabilidade do próprio evento ocorrer, sem conhecimento do evento anterior, i.e., $P(A|B) = P(A)$ e a probabilidade da intersecção de A e B é a multiplicação da probabilidade de cada evento separado, i.e., $P(A\bigcap B)=P(A)*P(B)$. Porém, não podemos simplesmente dizer que a pessoa ter 60 anos e ter alguma comorbidade são eventos independentes, assim como não podemos dizer que uma pessoa ter uma comorbidade 1 e também ter uma comorbidade 2 são eventos independentes.

Para levar em conta essa dependência entre os subconjuntos dentro de nosso espaço amostral, devemos usar a definição de probabilidade condicional de um evento A dado um evento B onde ambos são dependentes entre si. Para todo P(A)>0, essa probabilidade é definida como

$P(A|B) = \frac{P(A \bigcap B)}{P(A)},$

sendo $P(A\bigcap B) \neq P(A)*P(B)$. Para múltiplos eventos, a regra da multiplicação de probabilidades condicionais nos diz que 

$P(A_1 \bigcap A_2 \bigcap A_3 \bigcap$ ... $\bigcap A_n)$ = $P(A_1)*P(A_2|A_1)*P(A_3|A_2\bigcap A_1)$ ... $P(A_n|A_{n-1}\bigcap A_{n-2}\bigcap$ ... $\bigcap A_1)$.

Para ilustrar o significado dessa expressão, pensemos no também exemplo clássico de eventos dependentes de uma urna com bolinhas diferentes. Dentro da urna colocamos 6 bolinhas, sendo 2 da cor vermelha, 2 da cor verde e 2 da cor azul. Em cada rodada pegamos uma bolinha da urna, olhamos a sua cor e deixamos ela fora da urna. Na primeira rodada a probabilidade de se obter uma bolinha vermelha é de 2/6, mas se soubermos que na primeira rodada foi retirada uma bolinha azul da urna, então na segunda rodada essa probabilidade é de 2/5! E mais, se soubermos que na segunda rodada a bolinha retirada era vermelha, então na terceira rodada essa probabilidade será de 1/4! Perceba que, conforme retiramos bolinhas e sabemos a sua cor, a minha *crença* de que nessa rodada a cor da bolinha que irei retirar será vermelha será atualizada, sempre com o espaço amostral novo, atualizado pelas informações anteriores. O que a expressão acima nos diz, em conclusão, é que a probabilidade de se retirar uma bolinha vermelha, verde e azul, respectivamente, é igual a multiplicação da probabilidade de se retirar uma vermelha na rodada 1, depois uma verde na rodada dois sabendo que na um foi a vermelha, e depois uma azul na rodada 3, sabendo que saíram a vermelha e a verde nas rodadas anteriores!

***

## Aplicação no caso em estudo

Visto tudo isso, foi então a hora de pensar na aplicação do modelo de Bayes em nosso conjunto de dados. A ideia era fornecer ao modelo os dados $Y_n$, que seriam idade, gênero e doenças pré-existentes, e obter uma probabilidade da pessoa vir a óbito ou não. Para isso, deveríamos ser capazes de calcular as três variáveis do lado direito da expressão. Como o conjunto de dados possuia aproximadamente 200 mil linhas preenchidas com todos os dados necessários, utilizamos a aproximação de que a frequência de ocorrência dos eventos dentro de nosso espaço amostral seria uma boa aproximação para a probabilidade desse evento ocorrer.

1. Probabilidade a priori $P(X)$:

  A probabilidade a priori é uma constante que seria basicamente a mortalidade da infecção de COVID-19 no estado de São Paulo. Na visão frequentista de probabilidades, a mortalidade é calculada pela quantidade de ocorrências de óbitos dentre todos os casos de infecção de COVID-19. Para calcular isso, apenas contamos quantos óbitos existiam no conjunto de dados e dividimos pelo total de casos de infecção.

2. Probabilidade dos dados $P(\bigcap_{i}^{n}Y_i)$:

  A probabilidade dos dados seria a probabilidade da combinação de informações fornecidas pelo modelo ocorrerem dentro de todos os casos de infecção de COVID-19. Como visto na seção anterior, essa probabilidade não seria uma multiplicação das ocorrências individuais, mas sim uma multiplicação de eventos condicionais atualizados pelos eventos anteriores. Dessa forma, para calcular a probabilidade dos dados, fizemos um loop que atulizava um DataFrame auxiliar de acordo com que as informações eram lidas do input. Ao ler a idade da pessoa, por exemplo, o DataFrame era filtrado por todas as pessoas que tinham a mesma idade, e então era calculado o número de linhas do DataFrame atualizado e dividido pelo número de linhas do DataFrame anterior (no caso, o número total de infeccção por COVID-19) para se obter a probabilidade da idade. Em seguida, o DataFrame era filtrado pela próxima informação, por exemplo o gênero, e então o número de linhas do DataFrame filtrado era novamente salvo e dividido pelo número de linhas do DataFrame anterior para se obter a probabilidade do gênero dado a idade. O loop seguia até a última informação, e as probabilidades de cada iteração iam sendo multiplicadas na variável *prob*, que era então retornada. É importante notar que, caso a variável *prob* desse 0, significaria que não existe a combinação do input no conjunto de dados, e então não seríamos capaz de calcular a probabilidade dessa pessoa vir a óbito.

3. Função de verossimilhança $P(\bigcap_{i}^{n}Y_i|X)$:

  A função de verossimilhança seguiu a mesma lógica da função de probabilidade dos dados, só que ao inicializar o DataFrame inicial, utilizamos o DataFrame filtrado apenas com os casos de óbito, pois agora queremos a frequência da combinação das informações de input dentre todas as pessoas que vieram a óbito por COVID-19. 

***

Note que, dessa forma, temos um modelo robusto que segue uma lógica fundamental para o cálculo da probabilidade da pessoa vir a óbito dado as informações iniciais. Caso a constante de proporcionalidade da expressão do modelo de Bayes seja maior que 1 então a probabilidade atualizada é maior que a nossa probabilidade a priori, o que significaria que a função de verossimilhança, i.e. a frequência com que essa combinação ocorreu dentre todos os casos de óbito, é maior que a probabilidade desse dado ocorrer, i.e. a frequência com que essa combinação ocorre dentre todos os casos totais de infecção por COVID-19. Caso a função de verossimilhança seja menor que a probabilidade dos dados, então a probabilidade atualizada da pessoa vir a óbito seria menor que a probabilidade a priori


> Se $\alpha = \frac{P(Y|X)}{P(Y)} > 1 \implies P(X|Y)>P(X)$


> Se $\alpha = \frac{P(Y|X)}{P(Y)} < 1 \implies P(X|Y)<P(X)$

O principal problema do modelo de Bayes é que não é possível calcular a probabilidade desejada caso a combinação de dados nunca tenha ocorrido no conjunto de dados, i.e. P(Y)=0. Nesse sentido, é como se o modelo não pudesse "prever" a probabilidade de casos que nunca ocorreu, apenas comparar o seus dados com dados que já ocorreram.
