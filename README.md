# Case COVID-19

Este repositório guarda o Jupyter Notebook que fiz para um case de estudos de um processo seletivo que participei quando estava trabalhando no mercado financeiro. Nessa etapa do processo, foi nos apresentado um case de estudos que teríamos que desenvolver em Python.
Utilizando um conjunto de dados de saúde da pandemia de COVID-19 do governo do estado de São Paulo, teríamos que responder três perguntas e desenvolver um modelo que fosse capaz de calcular a probabilidade de uma pessoa vir a óbito pela infecção de COVID-19 dado o seu gênero, sua idade e doenças pré-existentes. Deixou-se livre para nós escolhermos o modelo que iríamos criar para atingir o objetivo e as bibliotecas em Python que iríamos utilizar.

Para responder as perguntas, utilizei bibliotecas simples de análise de dados que utilizava todos os dias em meu trabalho: pandas, numpy e matplotlib. Na época, eu não tinha conhecimento de modelos de Aprendizado de Máquina, então pensei no modelo estatístico mais fundamental que vi nas aulas de probabilidades em minha graduação: o modelo de Bayes. Eu sabia que com ele eu conseguiria calcular a probabilidade de uma certa asserção dada hipóteses conhecidas e conhecimentos prévios de um conjunto de dados, então bastou eu pensar em como implementar e aplicar o modelo para o caso específico apresentado. 

Irei descrever brevemente o arcabouço teórico que sustentou minhas escolhas na hora da implementação do modelo de Bayes.

***

## Modelo de Bayes

O modelo de Bayes pode ser inferido da lei de probabilidades totais, e sua experssão geralmente é escrita da seguinte forma:

$P(X|Y) = \frac{P(Y|X)}{P(Y)} P(X) $.

Na visão bayesiana, a probabilidade mede nosso *grau de crença*, então nesse sentido a expressão acima codifica matematicamente o fenômeno de "aprendizado", pois  probabilidade P(X) e P(Y) representa nossa crença de que uma dada asserção X e Y seja verdadeira e P(X|Y) ou P(Y|X) representa nossa crença de que tal asserção seja verdadeira dado um conhecimento prévio da asserção à direita da barra. É como se a expressão acima como uma "atualização" de nossa crença X agora que temos dados da crença de Y: essa crença atual P(X|Y) seria proporcional à nossa crença "a priori" P(X), e a constante de proporcionalidade $\frac{P(Y|X)}{P(Y)}$ seria dada por crenças "a priori" da informação obtida Y. Cada expressão individual é conhecida por um nome, sendo P(Y|X) conhecida como a função de verossimilhança (likelihood function), P(Y) a probabilidade dos meus dados e P(X) minha probabilidade a priori.

Para aplicar o modelo de Bayes no nosso caso de estudo, pense em X como a asserção "morrer pela infecção de COVID-19" e em Y como os dados que devemos colocar como input. Dessa forma, a expressão do modelo de Bayes representaria a seguinte frase: "A probabiliadde de morrer pela infecção de COVID-19 dado meu gênero, idade e doenças pré-existentes (P(X|Y)) é igual à probabilidade de ter minha idade, gênero e doenças pré-existentes e ter morrido por COVID-19 (P(Y|X)) vezes a probabilidade de morrer de COVID-19 (P(X)) dividido pela probabilidade de ter minha idade, gênero e doenças pré-existentes (P(Y))"

***

## Eventos independentes e probabilidade condicional

Em nosso estudo de caso, entretanto, ao invés de uma asserção simples como informação, temos diversas asserções. Queremos saber qual a probabilidade de uma pessoa vir a óbito por COVID-19 dado sua idade **E** seu gênero **E** se tem comorbidade 1 **E** se tem comorbidade 2 e por aí vai. Na tradução para a linguagem matemática, significa que nosso conhecimento que irá atualizar P(X) não será um conjunto Y, mas a intersecção de vários subconjuntos $Y_i$ do meu espaço amostral, e ao analisar essas probabilidades condicionadas por outras asserções, é importante sabermos se os eventos são independentes ou não.

Definimos eventos independentes como eventos os quais não afetam a probabilidade de um evento posterior. A jogada múltipla de um dado não enviesado é um exemplo clássico de eventos independentes, pois o resultado da primeira jogada não afeta o resultado da jogada seguinte, então a probabilidade de se obter um 6 na segunda jogada continua sendo 1/6, mesmo sabendo que na primeira jogada obtivemos um 4. Por conclusão, podemos definir a probabilidade condicional de dois eventos independentes como sendo a probabilidade do próprio evento ocorrer, sem conhecimento do evento anterior, i.e., $P(A|B) = P(A)$ e a probabilidade da intersecção de A e B é a multiplicação da probabilidade de cada evento separado, i.e., $P(A\bigcap B)=P(A)*P(B)$. Porém, não podemos simplesmente dizer que a pessoa ter 60 anos e ter alguma comorbidade são eventos independentes, assim como não podemos dizer que uma pessoa ter uma comorbidade 1 e tambér ter uma comorbidade 2 são eventos independentes.

Para levar em conta essa dependência entre os subconjuntos dentro de nosso espaço amostral, devemos usar a definição de probabilidade condicional de um evento A dado um evento B onde ambos são dependentes entre si. Para todo P(A)>0, essa probabilidade é definida como

$P(A|B) = \frac{P(A \bigcap B)}{P(A)},$

sendo $P(A\bigcap B) \neq P(A)*P(B)$. Para múltiplos eventos, a regra da multiplicação de probabilidades condicionais nos diz que 

$P(A_1 \bigcap A_2 \bigcap A_3 \bigcap$ ... $\bigcap A_n)$ = 
