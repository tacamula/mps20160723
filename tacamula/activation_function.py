import numpy as np

def sigmoid(s):
    return 1/(1 + np.exp(-s))

if __name__ == '__main__':
    x = np.random.randn(3, 1)
    W = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    y = sigmoid(W @ x + b)
    print(y)
