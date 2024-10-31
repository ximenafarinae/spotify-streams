import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_data
from src.data.data_transformer import convert_columns_values_to_numeric, remove_outliers, normalize_input
from src.models.evaluator import evaluate
from src.models.model import StreamsNeuralNetwork
from src.config import *
from src.models.trainer import train
from src.utils.visualization import plot_accuracy

hidden_layer_size = 32
hidden_layer_2_size = 16
input_size = 4
output_size = 1

data = load_data(DATA_PATH)

# Convierto todos los datos a valores numericos
data = convert_columns_values_to_numeric(pd, data, data.select_dtypes(include='object').columns)

# Elimino todas las columnas que contengas datos NaN
data = data.dropna(axis=1, how='all')

#Estos fueron a mano porque contenian al menos un dato numerico.
data = data.drop(['track_name', 'instrumentalness_%'], axis=1)
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Elimino los valores atipicos
data = remove_outliers(data, numeric_columns)

# Cambiar los valores de la columna 'streams' a 0 o 1 según si pasa el umbral de 100 millones
data['streams'] = data['streams'].apply(lambda x: 1 if x > 100_000_000 else 0)

# Muevo la columna streams al final para que sea mas comodo seleccionar las columnas relevantes para el entrenamiento
data = data[[col for col in data.columns if col != 'streams'] + ['streams']]

#Tomo las columnas relevantes (segun matriz de correlacion)
relevant_columns = data.iloc[:, 5:9]

# Normalizo las columnas de entrada.
relevant_columns = normalize_input(relevant_columns)

all_inputs = relevant_columns.values

all_outputs = data.iloc[:, -1].values

# Dividir en un conjunto de entrenamiento y uno de prueba
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs,
    test_size=1/3)

n = X_train.shape[0]

model = StreamsNeuralNetwork(INPUT_SIZE, HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE, OUTPUT_SIZE)

train(np, model, X_train, Y_train, EPOCHS, L)

evaluate(np, model, X_test, Y_test)

#plot_accuracy(accuracy)
