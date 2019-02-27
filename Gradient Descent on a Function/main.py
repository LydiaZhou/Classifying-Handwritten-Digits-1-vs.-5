import numpy as np
import matplotlib.pyplot as plt

# The function is f(x,y) = x^2 + 2y^2 + 2 sin(2πx) sin(2πy).
def function(x, y):
    return np.square(x)+2*np.square(y)+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def step(x, y, learningRate):
    partialx = 2*x+2*np.pi*2*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    partialy = 4*y+2*np.pi*2*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    x_update = x - learningRate * partialx
    y_update = y - learningRate * partialy
    return (x_update, y_update)

if __name__ == '__main__':
    # With learning rate 0.01, start position (0.1, 0.1)
    alpha = 0.01
    (x, y) = (0.1, 0.1)
    plt.title("Learning rate=0.01, start position=(0.1, 0.1)")
    for iter in range(50):
        plt.plot(iter, function(x,y), 'ro')
        (x, y) = step(x, y, alpha)
    plt.show()

    # With learning rate 0.1, start position (0.1, 0.1)
    largeAlpha = 0.1
    (x, y) = (0.1, 0.1)
    plt.title("Learning rate=0.1, start position=(0.1, 0.1)")
    for iter in range(50):
        plt.plot(iter, function(x,y), 'ro')
        (x, y) = step(x, y, largeAlpha)
    plt.show()

    # With learning rate 0.01, start position (1, 1)
    (x, y) = (1, 1)
    plt.title("Learning rate=0.1, start position=(1, 1)")
    for iter in range(50):
        plt.plot(iter, function(x, y), 'ro')
        (x, y) = step(x, y, alpha)
    plt.show()

    # With learning rate 0.01, start position (-1, -1)
    (x, y) = (-1, -1)
    plt.title("Learning rate=0.1, start position=(-1, -1)")
    for iter in range(50):
        plt.plot(iter, function(x, y), 'ro')
        (x, y) = step(x, y, alpha)
    plt.show()

    # With learning rate 0.01, start position (-0.5, -0.5)
    (x, y) = (-0.5, -0.5)
    plt.title("Learning rate=0.1, start position=(-0.5, -0.5)")
    for iter in range(50):
        plt.plot(iter, function(x, y), 'ro')
        (x, y) = step(x, y, alpha)
    plt.show()