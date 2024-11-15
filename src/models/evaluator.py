def evaluate(np, model, X_val, Y_val):
    _, _, _, val_A2 = model.forward_prop(X_val)
    val_loss = np.mean((val_A2 - Y_val) ** 2)
    val_accuracy = np.mean((val_A2 >= 0.5).astype(int).flatten() == Y_val)

    return val_loss, val_accuracy