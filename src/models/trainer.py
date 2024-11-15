from src.models.evaluator import evaluate


def train(np, model, X_train, Y_train, X_val, Y_val, epochs, l, n, m):
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(epochs):

        # Seleccionar aleatoriamente uno de los datos de entrenamiento
        idx = np.random.choice(n, 1, replace=False)
        X_sample = X_train[idx].transpose()
        Y_sample = Y_train[idx]

        # Seleccionar aleatoriamente uno de los datos de test
        i = np.random.choice(m, 1, replace=False)
        X_val_sample = X_val[i].transpose()
        Y_val_sample = Y_val[i]

        # Forward propagation
        Z1, A1, Z2, A2 = model.forward_prop(X_sample)

        train_loss = np.mean((A2 - Y_sample) ** 2)
        train_accuracy = np.mean((A2 >= 0.5) == Y_sample)

        l =  0.001 / (1 + 0.0001 * epoch)

        # Backward propagation
        model.backward_prop(X_sample, Y_sample, Z1, A1, Z2, A2, l)

        val_loss, val_accuracy= evaluate(np, model, X_val_sample, Y_val_sample)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
            print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            print("\n")

    return train_accuracies, val_accuracies, train_losses, val_losses