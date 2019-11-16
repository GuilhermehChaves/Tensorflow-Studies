import pandas as pd
base = pd.read_csv('house-prices.csv')
base.head()

base.columns
colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']

base = pd.read_csv('house-prices.csv', usecols = colunas_usadas)
base.head()

# Pode se usar o Escalonamento ou a normalização do valores,
# para pois a escala de um valor pro outro é muito alta

# Usando a normalização
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
# Reaizando a normalização de todos os valores
# Esxeto o price pois o price é o y do gráfico
base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])

base.head()

scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']])

# Função drop elimina um valor da matrix neste caso o price, axis = 1 pois irá apagar a coluna inteira
x = base.drop('price', axis = 1)
y = base.price

# Pandas DataFrame = Várias colunas
# Series = Apenas uma coluna

# Variáveis independentes
x.head()

y.head()

# Busca em colunas_usadas índices de 1 até 17, pois não usaremos o indece 0 que seria o preço
previsores_colunas = colunas_usadas[1:17]
previsores_colunas

import tensorflow.compat.v1 as tf
# Executa um for para pegar cada coluna do elemento x
colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y, test_size = 0.3)

x_treinamento.shape
y_treinamento.shape

# Define a função de treinamento e teste
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y = y_treinamento, batch_size = 32, num_epochs = None, shuffle = True)
funcao_teste = tf.estimator.inputs.pandas_input_fn(x = x_teste, y = y_teste, batch_size = 32, num_epochs = 1000, shuffle = False)
# Define a regressão linear
regressor = tf.estimator.LinearRegressor(feature_columns = colunas)
# Efetua de fato o treinamento
regressor.train(input_fn = funcao_treinamento, steps = 10000)

# Define as métricas de teste e treinamento
metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento, steps = 10000)
metricas_teste = regressor.evaluate(input_fn = funcao_teste, steps = 10000)

metricas_treinamento
metricas_teste

# Define a função de previsão
funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, shuffle = False)
# Realiza as previsões
previsoes = regressor.predict(input_fn = funcao_previsao)

list(previsoes)

valores_previsoes = []

# Adicionando os valores das previsões na lista valores_previsoes
for p in regressor.predict(input_fn = funcao_previsao):
    valores_previsoes.append(p['predictions'])

valores_previsoes

import numpy as np
# Transformando a lista em array, e define-o como uma matriz
# pois para a desnormalização os dados precisam ser em uma matriz
valores_previsoes = np.asarray(valores_previsoes).reshape(-1, 1)
# Desnomalizando os valores
valores_previsoes = scaler_y.inverse_transform(valores_previsoes)
valores_previsoes

y_teste2 = y_teste.values.reshape(-1, 1)
# Desnormalizando os valores de teste de y
y_teste2 = scaler_y.inverse_transform(y_teste2)
y_teste2

# Efetua o cálculo do mean absolute error das previsões
from sklearn.metrics import mean_absolute_error
meanAbsoluteError = mean_absolute_error(y_teste2, valores_previsoes)
print(meanAbsoluteError)



    
    








