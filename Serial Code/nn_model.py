"""
The neural network from the MNIST notebook, pulled out into a normal .py file
so the MPI code and the cluster can use it.

Original code: https://github.com/SwamiKannan/Neural-Network-from-scratch-Numpy
(MNIST_Python_Neural_Network_Final.ipynb). The math in Linear, ReLU, softmax,
CrossEntropy and NN is the same as the notebook. The only things we added are
the helper functions at the bottom for getting weights in and out of a network,
which the parallel version needs so it can send the network to every process.

The network is:  input (784) -> Linear -> ReLU -> Linear -> scores (10)
predict_scores gives the raw scores (no softmax), predict gives the digit.
"""
import numpy as np


class Linear():
    def __init__(self, in_size, out_size):
        self.W = np.random.randn(in_size, out_size) * 0.01
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]
        self.gradW = None
        self.gradB = None
        self.gradInput = None

    def forward(self, X):
        self.X = X
        self.output = np.dot(X, self.W) + self.b
        return self.output

    def backward(self, nextgrad):
        self.gradW = np.dot(self.X.T, nextgrad)
        self.gradB = np.sum(nextgrad, axis=0)
        self.gradInput = np.dot(nextgrad, self.W.T)
        return self.gradInput, [self.gradW, self.gradB]


class ReLU():
    def __init__(self):
        self.params = []
        self.gradInput = None

    def forward(self, X):
        self.output = np.maximum(X, 0)
        return self.output

    def backward(self, nextgrad):
        self.gradInput = nextgrad.copy()
        self.gradInput[self.output <= 0] = 0
        return self.gradInput, []


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class CrossEntropy:
    def forward(self, X, y):
        self.m = y.shape[0]
        self.p = softmax(X)
        cross_entropy = -np.log(self.p[range(self.m), y])
        # The notebook did cross_entropy[0] / m here, which only looks at the
        # first image. We average over all of them instead. This only changes
        # the loss number that gets printed, not the gradients or training.
        return np.mean(cross_entropy)

    def backward(self, X, y):
        grad = softmax(X)
        grad[range(self.m), y] -= 1
        grad /= self.m
        return grad


class NN():
    def __init__(self, lossfunc=None):
        self.params = []
        self.layers = []
        self.loss_func = lossfunc if lossfunc is not None else CrossEntropy()
        self.grads = []

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params.append(layer.params)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, nextgrad):
        self.clear_grad_param()
        for layer in reversed(self.layers):
            nextgrad, grad = layer.backward(nextgrad)
            self.grads.append(grad)
        return self.grads

    def train_step(self, X, y):
        out = self.forward(X)
        loss = self.loss_func.forward(out, y)
        nextgrad = self.loss_func.backward(out, y)
        grads = self.backward(nextgrad)
        return loss, grads

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def predict_scores(self, X):
        return self.forward(X)

    def clear_grad_param(self):
        self.grads = []


def build_net(layer_sizes):
    """Make a new network with random weights.
    layer_sizes = [784, 32, 10] gives Linear(784,32) -> ReLU -> Linear(32,10),
    which is the same network the notebook trains."""
    net = NN()
    n_layers = len(layer_sizes) - 1
    for i in range(n_layers):
        net.add_layer(Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < n_layers - 1:          # ReLU between layers, not after the last one
            net.add_layer(ReLU())
    return net


"""
Helpers for moving weights around. The MPI code uses these to send the
network from rank 0 to everyone else.

Weights are stored as a list of (W, b) pairs, one per Linear layer, in order.
"""

def get_weights(net):
    """Pull the (W, b) out of every Linear layer."""
    return [(layer.W, layer.b) for layer in net.layers if isinstance(layer, Linear)]


def net_from_weights(weights):
    """Build a network that uses the given (W, b) pairs.
    Puts a ReLU between every pair of Linear layers, same as build_net."""
    net = NN()
    for i, (W, b) in enumerate(weights):
        layer = Linear(W.shape[0], W.shape[1])
        layer.W = np.array(W, dtype=np.float64)          # copy so nets don't share arrays
        layer.b = np.array(b, dtype=np.float64).reshape(1, -1)
        layer.params = [layer.W, layer.b]                # params has to point at the new arrays
        net.add_layer(layer)
        if i < len(weights) - 1:
            net.add_layer(ReLU())
    return net


def save_weights(net, path):
    """Save the weights to a .npz file (W0, b0, W1, b1, ...)."""
    arrays = {}
    for i, (W, b) in enumerate(get_weights(net)):
        arrays[f"W{i}"] = W
        arrays[f"b{i}"] = b
    np.savez(path, **arrays)


def load_weights(path):
    """Read a .npz made by save_weights back into a list of (W, b) pairs."""
    data = np.load(path)
    n_layers = len([k for k in data.files if k.startswith("W")])
    return [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_layers)]