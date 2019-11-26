import numpy as np

# Retorna 0 ou 1
def step(soma):
    if(soma >= 1):
        return 1
    return 0

# Retorna valores entre 0 e 1
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

# Retorna valores entre -1 e 1, entradas negativas retornam um valor negativo
def than(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))
    
# Retorna um valor entre 0 e n
def relu(soma):
    if(soma >= 0):
        return soma
    return 0 

# Retorna o próprio valor
def linear(soma):
    return soma

# Retorna as probabilidades de ser uma determinada classe
# np.exp() = Exponencial
def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum()
    
    

teste = step(30)
teste = sigmoid(30)
teste = than(30)
teste = relu(30)
teste = linear(30)
valores = [0.12, 0.445, 0.233]
print(softmax(valores))
teste


quiz = than(2.1)