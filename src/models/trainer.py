def train(np, model, X, Y, epochs, l):
    for epoch in range(epochs):
        # Forward propagation
        Z1, A1, Z2, A2, Z3, A3 = model.forward_prop(X)
        # Backward propagation
        model.backward_prop(X, Y, Z1, A1, Z2, A2, A3, learning_rate=l)

        if epoch % 100 == 0:
            train_predictions = (A3 >= 0.5).astype(int)
            train_accuracy = np.mean(train_predictions.flatten() == Y)
            print(f"Epoch {epoch}, Training Accuracy: {train_accuracy:.4f}")

def train2(np, model, X_train, Y_train, epochs, learning_rate):
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Propagación hacia adelante y cálculo de precisión en entrenamiento
        model.forward_prop(X_train)
        train_accuracy = np.mean((model.A3 >= 0.5) == Y_train.reshape(-1, 1))
        train_accuracies.append(train_accuracy)

        # Verificar si los pesos se están actualizando
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Training Accuracy: {train_accuracy:.4f}")
            print("Peso de w_output:", model.w_output[:5])  # Mostrar algunos pesos de w_output para verificar cambios

        # Retropropagación y actualización de pesos
        Z1, A1, Z2, A2, Z3, A3 = model.forward_prop(X_train)
        model.backward_prop(X_train, Y_train, Z1, A1, Z2, A2, A3, learning_rate)

    return train_accuracies, val_accuracies
