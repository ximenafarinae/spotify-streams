import numpy as np

np.random.seed(42)

class StreamsNeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size):

        # Construir una red neuronal con pesos y sesgos iniciados aleatoriamente
        self.w_hidden = np.random.rand(input_size, hidden_layer_size) * np.sqrt(2.0 / input_size)
        self.b_hidden = np.random.rand(1, hidden_layer_size)

        self.w_output = np.random.rand(hidden_layer_size, output_size) * np.sqrt(2.0 / input_size)
        self.b_output = np.random.rand(1, output_size)

    def relu(self, x):
        return np.maximum(x, 0)

    def logistic(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward_prop(self, X, dropout_rate, training):
        Z1 = X @ self.w_hidden + self.b_hidden
        A1 = self.relu(Z1)
        if training and dropout_rate > 0:
            dropout_mask = (np.random.rand(*A1.shape) > dropout_rate) / (1 - dropout_rate)  # Normalización
            A1 *= dropout_mask

        Z2 = A1 @ self.w_output + self.b_output
        A2 = self.logistic(Z2)

        return Z1, A1, Z2, A2

    def backward_prop(self, X, Y, Z1, A1, Z2, A2, learning_rate, lambda_l2):

        # Calcular el error en la capa de salida
        dZ2 = A2 - Y.reshape(-1, 1)
        dw_output = A1.T @ dZ2 * self.w_output
        db_output = np.sum(dZ2, axis=0, keepdims=True)

        # Propagar el error hacia la primera capa oculta
        dA1 = dZ2 @ self.w_output.T
        dZ1 = dA1 * (Z1 > 0)
        dw_hidden = X.T @ dZ1 * self.w_hidden
        db_hidden = np.sum(dZ1, axis=0, keepdims=True)

        self.w_output -= learning_rate * dw_output
        self.b_output -= learning_rate * db_output
        self.w_hidden -= learning_rate * dw_hidden
        self.b_hidden -= learning_rate * db_hidden