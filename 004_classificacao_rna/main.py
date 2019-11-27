#Base para saber se a renda de uma pessoa será >=50 ou <= 50
import pandas as pd
data = pd.read_csv('census.csv')
data.head()

data.income.unique()
data.age.min

data.age.describe()['max']

def convert_class(label):
    if label == '>50K':
        return 1
    else:
        return 0

#Aplica a função convert_class no data.income
data.income = data.income.apply(convert_class)

x = data.drop('income', axis = 1)
y = data.income

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)



import tensorflow.compat.v1 as tf
#Atributos categóricos
workclass = tf.feature_column.categorical_column_with_hash_bucket(key = 'workclass', hash_bucket_size = 100)
education = tf.feature_column.categorical_column_with_hash_bucket(key = 'education', hash_bucket_size = 100)
marital_status = tf.feature_column.categorical_column_with_hash_bucket(key = 'marital-status', hash_bucket_size = 100)
occupation = tf.feature_column.categorical_column_with_hash_bucket(key = 'occupation', hash_bucket_size = 100)
relationship = tf.feature_column.categorical_column_with_hash_bucket(key = 'relationship', hash_bucket_size = 100)
race = tf.feature_column.categorical_column_with_hash_bucket(key = 'race', hash_bucket_size = 100)
country = tf.feature_column.categorical_column_with_hash_bucket(key = 'native-country', hash_bucket_size = 100)
sex = tf.feature_column.categorical_column_with_vocabulary_list(key = 'sex', vocabulary_list=[' Male', ' Female'])


#Atributos numéricos
age = tf.feature_column.numeric_column(key = 'age')
final_weight = tf.feature_column.numeric_column(key = 'final-weight')
education_num = tf.feature_column.numeric_column(key = 'education-num')
capital_gain = tf.feature_column.numeric_column(key = 'capital-gain')
capital_loos = tf.feature_column.numeric_column(key = 'capital-loos')
hour = tf.feature_column.numeric_column(key = 'hour-per-week')



embeded_workclass = tf.feature_column.embedding_column(workclass, dimension = len(data['workclass'].unique()))
embeded_education = tf.feature_column.embedding_column(education, dimension = len(data['education'].unique()))
embeded_marital = tf.feature_column.embedding_column(marital_status, dimension = len(data['marital-status'].unique()))
embeded_occupation = tf.feature_column.embedding_column(occupation, dimension = len(data['occupation'].unique()))
embeded_relationship = tf.feature_column.embedding_column(relationship, dimension = len(data['relationship'].unique()))
embeded_race = tf.feature_column.embedding_column(race, dimension = len(data['race'].unique()))
embeded_sex = tf.feature_column.embedding_column(sex, dimension = len(data['sex'].unique()))
embeded_country = tf.feature_column.embedding_column(country, dimension = len(data['native-country'].unique()))



rna_columns = [age, embeded_workclass, final_weight, embeded_education, education_num, 
               embeded_marital, embeded_occupation, embeded_relationship, embeded_race, embeded_sex,
               capital_gain, capital_loos, hour, embeded_country]

train_function = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train, batch_size = 32, num_epochs = None, shuffle = True)

classifier = tf.estimator.DNNClassifier(hidden_units = [8, 8], feature_columns = rna_columns, n_classes = 2)
classifier.train(input_fn = train_function, steps = 10000)



test_function = tf.estimator.inputs.pandas_input_fn(x = x_test, y = y_test, batch_size = 32, num_epochs = 1, shuffle = False)
classifier.evaluate(input_fn = test_function)


