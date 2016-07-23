from scipy import misc
import numpy as np

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
    x = np.random.randn(3, 1)

    # layer1
    W1 = np.random.randn(1, 3)
    b1 = np.random.randn(1, 1)
    layer1 = Layer(W1, b1, sigmoid)

    # layer2
    W2 = np.random.randn(3, 3)
    b2 = np.random.randn(3, 1)
    layer2 = Layer(W2, b2, sigmoid)

    y1 = layer1.propagate_forward(x)
    print(y1)

    y2 = layer2.propagate_forward(x)
    print(y2)
