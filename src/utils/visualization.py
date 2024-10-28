import matplotlib.pyplot as plt

def plot_accuracy(accuracies):
    plt.plot(accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.show()