import numpy as np

class StreamsNeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, hidden_layer_2_size, output_size):
        # Construir una red neuronal con pesos y sesgos iniciados aleatoriamente

        self.w_hidden = np.random.rand(input_size, hidden_layer_size)
        self.b_hidden = np.random.rand(1, hidden_layer_size)

        self.w_hidden2 = np.random.rand(hidden_layer_size, hidden_layer_2_size)
        self.b_hidden2 = np.random.rand(1, hidden_layer_2_size)

        self.w_output = np.random.rand(hidden_layer_2_size, output_size)
        self.b_output = np.random.rand(1, output_size)

    def relu(self, x):
        return np.maximum(x, 0)

    def logistic(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_prop(self, X):
        Z1 = self.w_hidden.T @ X.T + self.b_hidden.T
        A1 = self.relu(Z1)

        Z2 = self.w_hidden2.T @ A1 + self.b_hidden2.T
        A2 = self.relu(Z2)

        Z3 = self.w_output.T @ A2 + self.b_output.T
        A3 = self.logistic(Z3)

        return Z1, A1, Z2, A2, Z3, A3

    def forward_prop2(self, X):
        # Capa 1
        Z1 = np.dot(X, self.w_hidden) + self.b_hidden
        A1 = self.relu(Z1)

        # Capa 2
        Z2 = np.dot(A1, self.w_hidden2) + self.b_hidden2
        A2 = self.relu(Z2)

        # Capa de salida
        Z3 = np.dot(A2, self.w_output) + self.b_output
        A3 = self.logistic(Z3)

        return Z1, A1, Z2, A2, Z3, A3

    def backward_prop2(self, X, Y, Z1, A1, Z2, A2, A3, learning_rate=0.01):
        m = X.shape[0]  # Número de muestras

        # Error en la capa de salida
        dZ3 = A3 - Y.reshape(-1, 1)
        dw_output = (1 / m) * np.dot(A2.T, dZ3)
        db_output = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)

        # Error en la segunda capa oculta
        dA2 = np.dot(dZ3, self.w_output.T)
        dZ2 = dA2 * (Z2 > 0)  # Derivada de ReLU
        dw_hidden_2 = (1 / m) * np.dot(A1.T, dZ2)
        db_hidden_2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Error en la primera capa oculta
        dA1 = np.dot(dZ2, self.w_hidden2.T)
        dZ1 = dA1 * (Z1 > 0)  # Derivada de ReLU
        dw_hidden = (1 / m) * np.dot(X.T, dZ1)
        db_hidden = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Actualizar pesos y biases
        self.w_output -= learning_rate * dw_output
        self.b_output -= learning_rate * db_output
        self.w_hidden2 -= learning_rate * dw_hidden_2
        self.b_hidden2 -= learning_rate * db_hidden_2
        self.w_hidden -= learning_rate * dw_hidden
        self.b_hidden -= learning_rate * db_hidden

    def backward_prop(self, X, Y, Z1, A1, Z2, A2, A3, learning_rate=0.01):
        m = X.shape[0]  # Número de muestras

        # Calcular el error en la capa de salida
        dZ3 = A3 - Y.reshape(1, -1)
        dw_output = (1 / m) * (A2 @ dZ3.T)
        db_output = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

        # Propagar el error hacia la capa oculta
        dA2 = self.w_output @ dZ3
        dZ2 = dA2 * (Z2 > 0)
        dw_hidden_2 = (1 / m) * (A1 @ dZ2.T)
        db_hidden_2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Propagar el error hacia la capa oculta
        dA1 = self.w_hidden2 @ dZ2
        dZ1 = dA1 * (Z1 > 0)
        dw_hidden = (1 / m) * (X.T @ dZ1.T)
        db_hidden = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Asegurar que las formas coinciden para la actualización
        self.w_output -= learning_rate * dw_output
        self.b_output -= learning_rate * db_output
        self.w_hidden2 -= learning_rate * dw_hidden_2
        self.b_hidden2 -= learning_rate * db_hidden_2.T
        self.w_hidden -= learning_rate * dw_hidden
        self.b_hidden -= learning_rate * db_hidden.T