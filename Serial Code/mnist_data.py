"""
Loads MNIST the same way the notebook does: images flattened to 784 numbers,
scaled from 0-255 down to 0-1, labels as plain digits 0-9.

The notebook used keras to get the data, but we don't want to install
tensorflow on the cluster just for that. So instead:

  1. If data/mnist.npz exists, just load it.
  2. Otherwise download the original MNIST files once, convert them,
     and save data/mnist.npz for next time.


"""
import os
import gzip
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CACHE_PATH = os.path.join(DATA_DIR, "mnist.npz")

# A public mirror of the original MNIST files.
MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_images(path):
    # The image files have a 16 byte header, then one byte per pixel.
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 784)


def _read_labels(path):
    # The label files have an 8 byte header, then one byte per label.
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def _download_and_cache():
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {}
    for key, name in FILES.items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(MIRROR + name, path)
        paths[key] = path

    np.savez_compressed(
        CACHE_PATH,
        train_images=_read_images(paths["train_images"]),
        train_labels=_read_labels(paths["train_labels"]),
        test_images=_read_images(paths["test_images"]),
        test_labels=_read_labels(paths["test_labels"]),
    )


def load_mnist():
    """Returns X_train, y_train, X_test, y_test.
    X arrays are (n, 784) floats between 0 and 1, y arrays are digits 0-9."""
    if not os.path.exists(CACHE_PATH):
        _download_and_cache()
    data = np.load(CACHE_PATH)

    # same normalization as the notebook: divide by 255
    X_train = data["train_images"] / 255.0
    X_test = data["test_images"] / 255.0
    y_train = data["train_labels"].astype(np.int64)
    y_test = data["test_labels"].astype(np.int64)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)