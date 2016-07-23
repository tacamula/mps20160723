from scipy import misc
import numpy as np
from matplotlib import pyplot as plt

class Layer:
    def __init__(self, W, b, f):
        self._W = W
        self._b = b
        self._f = f

    def propagate_forward(self, x):
        return self._f(self._W @ x + self._b)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

if __name__ == '__main__':
    # input
    x = misc.imread('img/9.png', flatten = True).flatten()
    x = x.reshape((len(x), 1))
    print('---- x -----')
    print(x)

    # layer1
    n_output_1 = len(x)
    W1 = np.random.randn(n_output_1, len(x))
    b1 = np.random.randn(n_output_1, 1)
    layer1 = Layer(W1, b1, sigmoid)

    # layer2
    n_output_2 = 10
    W2 = np.random.randn(n_output_2, n_output_1)
    b2 = np.random.randn(n_output_2, 1)
    layer2 = Layer(W2, b2, sigmoid)

    y1 = layer1.propagate_forward(x)
    y2 = layer2.propagate_forward(x)

    hist_W1, bins_W1 = np.histogram(W1.flatten())
    hist_W2, bins_W2 = np.histogram(W2.flatten())

    index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.title('Predication')
    plt.bar(index, y2.flatten(), align='center')
    plt.xticks(index, index)
    plt.show()
