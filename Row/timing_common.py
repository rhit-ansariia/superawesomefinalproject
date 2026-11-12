"""
Stuff shared by timing_baseline.py and timing_rows.py so both time
exactly the same network on exactly the same images.
"""
import os
import sys
import numpy as np

# Same path trick as nn_rows_mpi.py so we can import from Serial Code.
SERIAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Serial Code")
sys.path.insert(0, SERIAL_DIR)

from nn_model import net_from_weights
from mnist_data import load_mnist

# How many times to repeat each measurement. We report the median.
REPEATS = 10


def make_net(hidden, seed=0):
    """784 -> hidden -> 10 network with random weights (fixed seed, so every
    run builds the same one). Only rank 0 / the serial script calls this."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((784, hidden)) * 0.01
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, 10)) * 0.01
    b2 = np.zeros(10)
    return net_from_weights([(W1, b1), (W2, b2)])


def load_batch(n):
    """The first n MNIST training images (there are 60,000)."""
    X_train, _, _, _ = load_mnist()
    if n > X_train.shape[0]:
        raise ValueError(f"batch size {n} is bigger than the 60000 MNIST images")
    return np.ascontiguousarray(X_train[:n])
