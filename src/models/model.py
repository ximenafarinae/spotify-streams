import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("../../data/raw/spotify_most_streamed_songs.csv")

categorical_cols_all_data = all_data.select_dtypes(include=['object']).columns
numerical_cols_all_data = all_data.select_dtypes(include=['int64', 'float64']).columns

#conversion de columnas a tipo numerico
for col in categorical_cols_all_data:
    all_data[col] = pd.to_numeric(all_data[col], errors='coerce')

#Se eliminan todas las columnas que contengan datos NaN
deleted_nan_data = all_data.dropna(axis=1, how='all')

#Estos fueron a mano porque contenian al menos un dato numerico.
cleaned_data = deleted_nan_data.drop(['track_name', 'instrumentalness_%'], axis=1)

numerical_cols_cleaned_data = cleaned_data.select_dtypes(include=['int64', 'float64']).columns

# Vamos a remover outliers
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

cleaned_data = remove_outliers(cleaned_data, numerical_cols_cleaned_data)

# Cambiar los valores de la columna 'streams' a 0 o 1 según si pasa el umbral de 100 millones
cleaned_data['streams'] = cleaned_data['streams'].apply(lambda x: 1 if x > 100_000_000 else 0)
cleaned_data = cleaned_data[[col for col in cleaned_data.columns if col != 'streams'] + ['streams']]


relevant_columns = cleaned_data.iloc[:, 5:9]

hidden_layer_size = 32
hidden_layer_2_size = 16
input_size = 4
output_size = 1

all_inputs = relevant_columns.values

all_outputs = cleaned_data.iloc[:, -1].values

# Dividir en un conjunto de entrenamiento y uno de prueba
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs,
    test_size=1/3)

n = X_train.shape[0] # número de registros de entrenamiento

# Construir una red neuronal con pesos y sesgos iniciados aleatoriamente
w_hidden = np.random.rand(input_size, hidden_layer_size)
b_hidden = np.random.rand(1, hidden_layer_size)

w_hidden2 = np.random.rand(hidden_layer_size, hidden_layer_2_size)
b_hidden2 = np.random.rand(1, hidden_layer_2_size)

w_output = np.random.rand(hidden_layer_2_size, output_size)
b_output = np.random.rand(1, output_size)

# Funciones de activacion
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Funcion que corre la red neuronal con los datos de entrada para predecir la salida
def forward_prop(X):
    Z1 = w_hidden.T @ X.T + b_hidden.T
    A1 = relu(Z1)

    Z2 = w_hidden2.T @ A1 + b_hidden2.T
    A2 = relu(Z2)

    Z3 = w_output.T @ A2 + b_output.T
    A3 = logistic(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def backward_prop(X, Y, Z1, A1, A2, A3, learning_rate=0.01):
    global w_hidden, b_hidden, w_hidden2, b_hidden2, w_output, b_output

    m = X.shape[0]  # Número de muestras

    # Calcular el error en la capa de salida
    dZ3 = A3 - Y.reshape(1, -1)

    dw_output = (1 / m) * (A2 @ dZ3.T)

    db_output = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    # Propagar el error hacia la capa oculta
    dA2 = w_output @ dZ3
    dZ2 = dA2 * (Z2 > 0)
    dw_hidden_2 = (1 / m) * (A1 @ dZ2.T)
    db_hidden_2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Propagar el error hacia la capa oculta
    dA1 = w_hidden2 @ dZ2
    dZ1 = dA1 * (Z1 > 0)
    dw_hidden = (1 / m) * (X.T @ dZ1.T)
    db_hidden = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    # Asegurar que las formas coinciden para la actualización
    w_output -= learning_rate * dw_output
    b_output -= learning_rate * db_output
    w_hidden2 -= learning_rate * dw_hidden_2
    b_hidden2 -= learning_rate * db_hidden_2.T
    w_hidden -= learning_rate * dw_hidden
    b_hidden -= learning_rate * db_hidden.T


# Bucle de entrenamiento
epochs = 1000
l = 0.055
for epoch in range(epochs):
    # Forward propagation
    Z1, A1, Z2, A2, Z3, A3 = forward_prop(X_train)

    # Backward propagation
    backward_prop(X_train, Y_train, Z1, A1, A2, A3, learning_rate=l)

    if epoch % 100 == 0:
        train_predictions = (A3 >= 0.5).astype(int)
        train_accuracy = np.mean(train_predictions.flatten() == Y_train)
        print(f"Epoch {epoch}, Training Accuracy: {train_accuracy:.4f}")

# Calculo de precisión
test_predictions = forward_prop(X_test)[-1] # me interesa solo la capa de salida, A3
test_predictions = (test_predictions >= 0.5).astype(int).flatten()
test_comparisons = np.equal(test_predictions, Y_test)
accuracy = np.mean(test_comparisons)
print("ACCURACY: ", accuracy)