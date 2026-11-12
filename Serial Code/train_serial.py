"""
Trains the network exactly like the notebook does, then saves the weights to
weights.npz so the parallel code can load the same trained network.

Same settings as the notebook: 784 -> 32 -> 10, learning rate 0.1,
momentum 0.9, batches of 200, 10 epochs. Should end up around 96-97%
test accuracy.

At the end it also times one serial forward pass over the whole test set,
which is the serial baseline number to compare the MPI version against.

    python train_serial.py
"""
import os
import time
import numpy as np

from nn_model import build_net, save_weights
from mnist_data import load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "weights.npz")


def update_params(velocity, params, grads, learning_rate=0.01, mu=0.9):
    """SGD with momentum, same as the notebook (minus the debug prints)."""
    for v, p, g in zip(velocity, params, reversed(grads)):
        for i in range(len(g)):
            v[i] = mu * v[i] + learning_rate * g[i]
            p[i] -= v[i]          # in place, so the layer's W and b actually change


def minibatch(X, y, minibatch_size):
    """Shuffle once and cut into batches, same as the notebook."""
    perm = np.random.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    return [(X[i:i + minibatch_size], y[i:i + minibatch_size])
            for i in range(0, X.shape[0], minibatch_size)]


def train(net, X_train, y_train, minibatch_size, epochs, learning_rate, mu=0.9,
          X_val=None, y_val=None):
    batches = minibatch(X_train, y_train, minibatch_size)

    for epoch in range(epochs):
        # the notebook resets momentum at the start of every epoch, so we do too
        velocity = [[np.zeros_like(p) for p in layer] for layer in net.params]

        losses = []
        for X_mini, y_mini in batches:
            loss, grads = net.train_step(X_mini, y_mini)
            losses.append(loss)
            update_params(velocity, net.params, grads, learning_rate, mu)

        train_acc = np.mean(net.predict(X_train) == y_train)
        msg = f"epoch {epoch + 1}: loss {np.mean(losses):.4f} | train acc {train_acc:.4f}"
        if X_val is not None:
            msg += f" | test acc {np.mean(net.predict(X_val) == y_val):.4f}"
        print(msg, flush=True)
    return net


if __name__ == "__main__":
    np.random.seed(3)     # same seed as the notebook

    X_train, y_train, X_test, y_test = load_mnist()

    net = build_net([X_train.shape[1], 32, 10])
    net = train(net, X_train, y_train, minibatch_size=200, epochs=10,
                learning_rate=0.1, X_val=X_test, y_val=y_test)

    save_weights(net, WEIGHTS_PATH)
    print(f"saved weights to {WEIGHTS_PATH}")

    # Serial baseline: one forward pass over all 10,000 test images.
    t0 = time.perf_counter()
    pred = net.predict(X_test)
    elapsed = time.perf_counter() - t0
    print(f"serial forward pass: {elapsed:.4f} s for {len(X_test)} images, "
          f"accuracy {np.mean(pred == y_test):.4f}")