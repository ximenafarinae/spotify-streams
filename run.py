import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_data
from src.data.data_transformer import convert_columns_values_to_numeric, normalize_input, remove_outliers_percentile
from src.models.model import StreamsNeuralNetwork
from src.config import *
from src.models.trainer import train
from src.utils.visualization import plot_accuracy_and_loss, data_distribution, data_box_plot

data = load_data(DATA_PATH)

# Convierto todos los datos a valores numericos
data = convert_columns_values_to_numeric(pd, data, data.select_dtypes(include='object').columns)

# Elimino todas las columnas que contengas datos NaN
data = data.dropna(axis=1, how='all')

# Estos fueron a mano porque contenian al menos un dato numerico.
data = data.drop(['track_name', 'instrumentalness_%'], axis=1)

# Cambio los valores de la columna 'streams' a 0 o 1 según si pasa el umbral de 100 millones
data['streams'] = data['streams'].apply(lambda x: 1 if x > 100_000_000 else 0)

# Tomo las columnas relevantes (segun matriz de correlacion)
relevant_columns = data[["in_spotify_playlists", "in_apple_playlists", "in_deezer_playlists", "released_year"]]

# Elimino los valores atípicos usando la técnica de percentile ya que mis datos tienen una fuerte asimetria hacia la derecha
data = remove_outliers_percentile(data, relevant_columns)

# Normalizo las columnas de entrada.
norm_data = normalize_input(data)
relevant_columns = norm_data[["in_spotify_playlists", "in_apple_playlists", "in_deezer_playlists", "released_year"]]
relevant_columns = relevant_columns.dropna(axis=0, how='any')

# Grafico la distribucion de los datos
"""data_distribution(relevant_columns, "in_spotify_playlists")
data_distribution(relevant_columns, "in_apple_playlists")
data_distribution(relevant_columns, "in_deezer_playlists")
data_distribution(relevant_columns, "released_year")

data_box_plot(norm_data, relevant_columns.columns)"""

# Datos de entrada
all_inputs = relevant_columns.values

# Datos de salida
all_outputs = data[["streams"]].values

# Dividir en un conjunto de entrenamiento y uno de prueba
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs,
    test_size=1/3)
n = X_train.shape[0]
m = X_test.shape[0]
print("Cantidad de datos de entrenamiento: ", len(X_train))
print("Cantidad de datos de prueba: ", len(X_test))

print("Distribución de clases en entrenamiento:")
print(pd.Series(Y_train.flatten()).value_counts())
print("Distribución de clases en validación:")
print(pd.Series(Y_test.flatten()).value_counts())

# Se inicializa el modelo
model = StreamsNeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE)

# Entrenamiento de la red
train_accuracies, val_accuracies, train_losses, val_losses = train(np, model, X_train, Y_train, X_test, Y_test, EPOCHS, L, n, m)

# Grafica de perdida y precision
plot_accuracy_and_loss(train_accuracies, val_accuracies, train_losses, val_losses)
