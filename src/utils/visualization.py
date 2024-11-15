import matplotlib.pyplot as plt
import seaborn as sns

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


def data_distribution(df, column_name):
    # Histograma con Matplotlib
    plt.figure(figsize=(10, 6))

    plt.hist(df[column_name], bins=10, edgecolor='black', alpha=0.7)
    plt.title("Distribución de Datos (Histograma) " + column_name)
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.show()

    # Gráfico de densidad con Seaborn
    plt.figure(figsize=(10, 6))

    sns.kdeplot(df[column_name], fill=True)
    plt.title("Distribución de Datos (Densidad) " + column_name)
    plt.xlabel("Valor")
    plt.ylabel("Densidad")
    plt.show()


def data_box_plot(df, columns):
    plt.figure(figsize=(15, 100))

    for i, column in enumerate(columns, 1):
        plt.subplot(len(columns), 1, i)
        df.boxplot([column])
        plt.title(f"Boxplot of {column}")
        plt.ylabel("Original Values")
        plt.xticks([])

plt.tight_layout()
plt.show()