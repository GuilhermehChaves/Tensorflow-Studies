import pandas as pd
base = pd.read_csv('house-prices.csv')
base.head()
base.shape

x = base.iloc[:, 5].values
x = x.reshape(-1, 1)

y = base.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

import tensorflow.compat.v1 as tf

colunas = [tf.feature_column.numeric_column('x', shape = [1])]
regressor = tf.estimator.LinearRegressor(feature_columns = colunas)

from sklearn.model_selection import train_test_split
# Defina uma porção de dados para treinamento e outra para teste
# 0.3 = 30% para o teste
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y, test_size = 0.3)

# Exibe o tamanho do x_treinamento 
x_treinamento.shape
# Mesmo nome que demos a variável colunas no parâmetro e de onde vem o valor dela .
funcao_treinamento = tf.estimator.inputs.numpy_input_fn({'x' : x_treinamento}, y_treinamento, batch_size = 32, num_epochs = None, shuffle = True)

funcao_teste = tf.estimator.inputs.numpy_input_fn({'x': x_teste}, y_teste, batch_size = 32, num_epochs = 1000, shuffle = False)
# Realiza efetivamente 1000 épocas, steps = quantidade de vezes que é executado em cada época
regressor.train(input_fn = funcao_treinamento, steps = 10000)

# Retorna as métricas do treinamento e dos teste.
metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento, steps = 10000)
metricas_teste = regressor.evaluate(input_fn = funcao_teste, steps = 10000)

metricas_treinamento
metricas_teste


import numpy as np
# Criando registros novos para previsão.
novas_casas = np.array([ [800], [900], [1000] ])
# Escalonando os valores dos novos registros
novas_casas = scaler_x.transform(novas_casas)
novas_casas

# Definindo a função da previsão
funcao_previsao = tf.estimator.inputs.numpy_input_fn({'x': novas_casas}, shuffle = False)

# Execuntando a previsão
previsoes = regressor.predict(input_fn = funcao_previsao)
list(previsoes)

# Listando a previsão
for p in regressor.predict(input_fn = funcao_previsao):
    print(scaler_y.inverse_transform(p['predictions']))

