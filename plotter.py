import matplotlib.pyplot as plt


def plot_history(history):

    plt.plot(range(1, len(history)+1), history)
    plt.xlabel('Iteracja')
    plt.ylabel('Funkcja celu')
    plt.title('Wartość funkcji celu dla kolejnych iteracji')
    plt.show()
