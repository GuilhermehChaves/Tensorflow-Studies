# Regressão Linear simples
import pandas as pd
print(pd.__version__)

# Lendo arquivo .csv
base = pd.read_csv('house-prices.csv')
# Mostra os primeiros registros da base de dados
base.head()
# Quantidade total de registros da base de dados
base.count()
# Quantidade total de maneira reduzida
base.shape
# Define uma variável X com o valor da metragem quadrada da casa
X = base.iloc[:, 5].values
# Altera linhas e colunas -1 pois n vamos mexer nas linhas
# Adicionando uma coluna
X = X.reshape(-1, 1)
X.shape

Y = base.iloc[:, 2:3].values
Y.shape

# Efetuando o escalonamento dos valores de X e Y
from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
Y = scaler_y.fit_transform(Y)

import matplotlib.pyplot as plt
plt.scatter(X,Y)

# Regressão Linear simple
# y = b0 + b1 * x

import numpy as np
# Para repetir os mesmos valores
np.random.seed(1)
# Passamos número 2 pois queremos dois números
np.random.rand(2)

import tensorflow.compat.v1 as tf
b0 = tf.Variable(0.41)
b1 = tf.Variable(0.72)

# Executa ao poucos de 32 em 32
batch_size = 32
# Placeholder passamos o tipo e o tamanho (receberá os valores de X) 
xph = tf.placeholder(tf.float32, [batch_size, 1])
yph = tf.placeholder(tf.float32, [batch_size, 1])

y_modelo = b0 + b1 * xph
# Realiza o mean squared error da resposta originial com a previsão feita
erro  = tf.losses.mean_squared_error(yph, y_modelo)
# Otimizador definido como a descida do gradiente
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
# Minimização do erro
treinamento = otimizador.minimize(erro)
# Inicializa as variáveis
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    # 10000 épocas
    for i in range(10000):
        # Seleciona número aleatorios para preencher os dados de 32 em 32
        indices = np.random.randint(len(X), size = batch_size)
        # alimentando o placeholder
        feed = {xph: X[indices], yph: Y[indices]}
        session.run(treinamento, feed_dict = feed)
    
    b0_final, b1_final = session.run([b0, b1])

b0_final
b1_final

# Executa a previsão dos registros
previsoes = b0_final + b1_final * X
# Montando o gráfico
plt.plot(X,Y, 'o')
plt.plot(X, previsoes, color = 'red')

# Desescalonando os valores
y1 = scaler_y.inverse_transform(Y)
previsoes1 = scaler_y.inverse_transform(previsoes)

y1
previsoes1
# Definindo o mean absolute error 
from sklearn.metrics import mean_absolute_error
meanAbsolutError = mean_absolute_error(y1, previsoes1)

meanAbsolutError










