import tensorflow.compat.v1 as tf
print('---------------------------------------------------')
print(tf.__version__)

# Algumas funções básicas do Tensorflow 1
# 1) add -> Realiza a soma entre tensores
# 2) multiply -> Realiza a multiplicação comum entre tensores (escalar e vetores)
# 3) matmul -> Realiza a multiplicação entre matrizes, respeitando a 
# algebra linear
# 4) constant -> Cria um tensor do tipo constante

# Reseta as configurações do grafo para não dar conflito 
tf.reset_default_graph()

# Define um escopo
with tf.name_scope('operacoes'):
  with tf.name_scope('escopo_A'):
    # Realiza a soma entre os números 2 e 3
    a = tf.add(2,3, name='add')
  with tf.name_scope('escopo_B'):
    # Realiza a multiplicação entre a variável a e o número 3
    b = tf.multiply(a, 3, name='mult1')
    # Realiza a multiplicação entre as variáveis b e a
    c = tf.multiply(b, a, name='mult2')

# Define um sessão no tensorflow necessária para que se possa executar
# as operações anteriores
with tf.Session() as session:
  # Salva dados do grafo em um diretório chamado output
  # para poder ser usado no tensorboard
  writer = tf.summary.FileWriter('./output', session.graph)
  print('--------------------> Resultado <---------------------------')
  # Executa o grafo criado pela variável c e printa na tela
  print(session.run(c))
  # Finaliza o writer
  writer.close()

# tensorboar -> tensorboard --logdir=diretorio --host localhost --port porta