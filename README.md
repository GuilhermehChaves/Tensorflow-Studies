# Teorias
### Regressão Linear

Na regressão linear estudamos um evento y(Variável dependente) em função de
outras variáveis x(Variáveis independentes).

A regressão nos mostra qual a melhor curva para se aplicar a estes dados, de
forma que a distâcia entre os pontos seja a menor possível, podendo ser
utilizada futuramente para fazer previsões de y com base em novos valores de x.

Tendo como base as fórmulas:

**Regressão linear simples**

y = a + b * x

Quando temos apenas uma variável x.

**Regressão linear múltipla**

y = a + b * x + b2 * x2 + ... bn * xn

Quando temos várias variáveis independentes influenciando y.

Onde a e b definem a posição da reta ou da curvatura no gráfico, o objetivo é
encontrar os melhores valores para a e b.

### Classificação

Na classificação nosso objetivo é com base nos atributos previsores (variáveis
independentes) prevermos uma determinada classe (variável dependente), por exemplo 
a partir de dados de clientes de um banco prever os riscos que ela teria ao fazer 
um impréstimo, se seriam baixos, médios ou altos por exemplo.

Para fazermos a classificação usamos o algorítmo de **regressão logística** que
apesar do nome semelhante a regressão linear eles são diferentes e não devem ser
confundidos.

Mesmo um pouco parecido com o modelo de regressão linear um caso de classificação
não pode ser aplicado onde deveria ser um modelo de regressão linear e vice e versa.

Como nosso objetivo é classificarmos algo em uma determinada classe o gráfico
fica um pouco diferente do que visto na regressão linear.

![class_01](https://user-images.githubusercontent.com/48635609/69289111-45cc1480-0bda-11ea-8752-b8f0f47bc335.PNG)

Podemos observar que os pontos ficam em uma classe ou outra, não tendo a possibilidade
de ficarem em qualquer lugar no meio do gráfico por isso o tratamento deve ser diferente.

Caso usamos um modelo de regressão linear, neste exemplo o gráfico seria o seguinte:

![class_02](https://user-images.githubusercontent.com/48635609/69289653-00104b80-0bdc-11ea-9ef3-6387e365f836.PNG)

Onde uma parcela dos dados nos extremos da reta ficam de fora, e o valor de resposta
será um numero qualquer, por exemplo 0.5, 0.7 e não somente 0 ou 1 como é o caso onde 
temos 2 classes e queremos saber se será uma ou outra.

Já usando o modelo de regressão logística temos um gráfico assim:

![class_03](https://user-images.githubusercontent.com/48635609/69290070-6e094280-0bdd-11ea-82c4-a1ee6addcd60.PNG)

Onde os dados não ficam de fora da reta.







