import pandas as pd
base = pd.read_csv('nota-enem.csv')
base.head()
base.columns

colunas_usadas = ['nota', 'horas_dia', 'qtd_dias', 'media_escola']


from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
base[['horas_dia', 'qtd_dias', 'media_escola']] = scaler_x.fit_transform(base[['horas_dia', 'qtd_dias', 'media_escola']])

base.head()

scaler_y = MinMaxScaler()
base[['nota']] = scaler_y.fit_transform(base[['nota']])

x = base.drop('nota', axis = 1)
y = base.nota

x.head()
y.head()

previsores_colunas = colunas_usadas[1:4]
previsores_colunas 

import tensorflow.compat.v1 as tf
colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y, test_size = 0.3)

x_treinamento.shape
y_treinamento.shape
x_teste.shape

funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = x_treinamento, y = y_treinamento, batch_size = 32, num_epochs = None, shuffle = True) 
funcao_teste = tf.estimator.inputs.pandas_input_fn(x = x_teste, y = y_teste, batch_size = 32, num_epochs = 1000, shuffle = False)        
                                    
regressor = tf.estimator.LinearRegressor(feature_columns = colunas)

regressor.train(input_fn = funcao_treinamento, steps = 10000)    

metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento, steps = 10000)
metricas_teste = regressor.evaluate(input_fn = funcao_teste, steps = 10000)

metricas_treinamento
metricas_teste

x_teste

funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = x_teste, shuffle = False)
previsoes = regressor.predict(input_fn = funcao_previsao)  

list(previsoes)       

valores_previsoes = []

for p in regressor.predict(input_fn = funcao_previsao):
    valores_previsoes.append(p['predictions'])
            
valores_previsoes

import numpy as np

valores_previsoes = np.asarray(valores_previsoes).reshape(-1, 1)
valores_previsoes = scaler_y.inverse_transform(valores_previsoes)

valores_previsoes

y_teste2 = y_teste.values.reshape(-1, 1)
y_teste2 = scaler_y.inverse_transform(y_teste2)

y_teste2

from sklearn.metrics import mean_absolute_error
meanAbsoluteError = mean_absolute_error(y_teste2, valores_previsoes)
print('---------- Margem de erro ----------')
print(meanAbsoluteError)
margem_erro = meanAbsoluteError
print('---------- Valores reais ----------')
print(y_teste2)
print('---------- Valores previstos ----------')
print(valores_previsoes)

print('---------- Novos registros ----------')
# Define uma variável data como um array do tipo numpy
data = np.array([[3,2,69]])
data.shape
# Normalizando os valores na mesma escala de x
data = scaler_x.transform(data)
# Definindo uma variável dataset como um DataFrame do pandas
dataset = pd.DataFrame({'horas_dia' : data[:,0], 'qtd_dias': data[:,1], 'media_escola': data[:,2]})
dataset
# Define uma função de previsão para os novos registros
nova_funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = dataset, shuffle = False)
# Efetuando a previsão
novas_previsoes = regressor.predict(input_fn = nova_funcao_previsao)  

list(novas_previsoes)

resultado = []

for result in regressor.predict(input_fn = nova_funcao_previsao):
    resultado.append(result['predictions'])

resultado = np.asarray(resultado).reshape(-1, 1)
resultado = scaler_y.inverse_transform(resultado)


print('---------- Previsão registro ----------')
print(resultado)
print('---------- Margem de erro ----------')
print(margem_erro)
