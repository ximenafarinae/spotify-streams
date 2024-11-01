import matplotlib.pyplot as plt

def plot_accuracy_and_loss(train_accuracies, val_accuracies, train_losses, val_losses):
    plt.figure(figsize=(12, 5))

    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Entrenamiento')
    plt.plot(val_accuracies, label='Validación')
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.title("Precisión a lo largo de las épocas")
    plt.legend()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Entrenamiento')
    plt.plot(val_losses, label='Validación')
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Pérdida a lo largo de las épocas")
    plt.legend()

    plt.tight_layout()
    plt.show()