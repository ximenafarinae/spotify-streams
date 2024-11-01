import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_data
from src.data.data_transformer import convert_columns_values_to_numeric, normalize_input, \
    remove_outliers_std
from src.models.model import StreamsNeuralNetwork
from src.config import *
from src.models.trainer import train
from src.utils.visualization import plot_accuracy_and_loss

data = load_data(DATA_PATH)

# Convierto todos los datos a valores numericos
data = convert_columns_values_to_numeric(pd, data, data.select_dtypes(include='object').columns)

# Elimino todas las columnas que contengas datos NaN
data = data.dropna(axis=1, how='all')

# Estos fueron a mano porque contenian al menos un dato numerico.
data = data.drop(['track_name', 'instrumentalness_%'], axis=1)
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Elimino los valores atípicos usando la técnica de la desviación estandar
data = remove_outliers_std(data, numeric_columns)

# Cambio los valores de la columna 'streams' a 0 o 1 según si pasa el umbral de 100 millones
data['streams'] = data['streams'].apply(lambda x: 1 if x > 100_000_000 else 0)

# Tomo las columnas relevantes (segun matriz de correlacion)
relevant_columns = data[["in_spotify_playlists", "in_apple_playlists", "in_deezer_playlists"]]

# Normalizo las columnas de entrada.
relevant_columns = normalize_input(relevant_columns)

# Datos de entrada
all_inputs = relevant_columns.values

# Datos de salida
all_outputs = data[["streams"]].values

# Dividir en un conjunto de entrenamiento y uno de prueba
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs,
    test_size=1/3)
# Se inicializa el modelo
model = StreamsNeuralNetwork(INPUT_SIZE, HIDDEN_LAYER1_SIZE, OUTPUT_SIZE)

# Entrenamiento de la red
train_accuracies, val_accuracies, train_losses, val_losses = train(np, model, X_train, Y_train, X_test, Y_test, EPOCHS, L, L2, DROPOUT)

# Grafica de perdida y precision
plot_accuracy_and_loss(train_accuracies, val_accuracies, train_losses, val_losses)
