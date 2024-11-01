def evaluate(np, model, X_val, Y_val, dropout):
    _, _, _, val_A2 = model.forward_prop(X_val, dropout, training=False)
    val_loss = np.mean((val_A2 - Y_val.reshape(-1, 1)) ** 2)
    val_accuracy = np.mean((val_A2 >= 0.5).astype(int).flatten() == Y_val)

    return val_loss, val_accuracy