import pandas as pd
data = pd.read_csv('census.csv')

#Implementar padronização futuramente (x - media(x)) / (desvio_padrao(x))

# Normalizando um valor de um pandas Series
def pandas_normalization(dataset_column, value):
    normalized_value = (value - dataset_column.describe()['min']) / (dataset_column.describe()['max'] - dataset_column.describe()['min'])
    return normalized_value

def pandas_normalization_all(dataset_column):
    normalized_values = []
    for i in range(len(dataset_column)):
        normalized_values.append(pandas_normalization(dataset_column, dataset_column[i]))
        print(normalized_values[i])
    return normalized_values

pandas_normalization(data.age, 30)
pandas_normalization_all(data.age)


import numpy as np
np_data = np.array([1,500,3,40,6657,456,7,10000,9,20])

#Normalizando um dado de um numpy array
def numpy_normalization(dataset_column, value):
    normalized_value = (value - np.amin(dataset_column) ) / (np.amax(dataset_column) - np.amin(dataset_column))
    return normalized_value

#Normalizando todos os dados de um numpy array
def numpy_normalization_all(dataset_column):
    normalized_values = []
    for i in range(len(dataset_column)):
        normalized_values.append(numpy_normalization(dataset_column, dataset_column[i]))
    return normalized_values
    

numpy_normalization(np_data, 6657)
numpy_normalization_all(np_data)




    