from src.models.evaluator import evaluate


def train(np, model, X_train, Y_train, X_val, Y_val, epochs, l, l2, dropout):
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Forward propagation
        Z1, A1, Z2, A2 = model.forward_prop(X_train, dropout, training=True)

        train_loss = np.mean((A2 - Y_train.reshape(-1, 1)) ** 2)
        train_accuracy = np.mean((A2 >= 0.5) == Y_train.reshape(-1, 1))

        # Backward propagation
        model.backward_prop(X_train, Y_train, Z1, A1, Z2, A2, l, l2)

        val_loss, val_accuracy = evaluate(np, model, X_val, Y_val, dropout)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
            print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            print("\n")

    return train_accuracies, val_accuracies, train_losses, val_losses