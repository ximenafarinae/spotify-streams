def evaluate(np, model, X_test, Y_test):
    _, _, _, _, _, predictions = model.forward_prop(X_test)
    accuracy = np.mean((predictions >= 0.5) == Y_test.reshape(-1, 1))
    print("Test Accuracy:", accuracy)
    return accuracy