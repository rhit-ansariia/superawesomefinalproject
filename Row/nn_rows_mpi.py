"""
Runs our neural network on many processes at once using MPI.

Each image is one row of X. The network's answer for an image only depends
on that image, not on any other image. So we can split the images up:

  1. Rank 0 has all the images.
  2. Rank 0 sends each process its own chunk of images (Scatterv).
  3. Every process runs the normal network on its chunk.
  4. Every process sends its answers back to rank 0 (Gatherv).
  5. Rank 0 now has the answers for every image, in the right order.

Every process needs the full network, so rank 0 sends a copy of the
weights to everyone once at the start.

The processes only talk to each other at the start and at the end.
They don't need to talk between layers.

If the images don't divide evenly, some processes get one extra image.
If there are fewer images than processes, some processes get zero images
and just sit out. Every process still joins every MPI call, so nothing should get stuck.
"""
import os
import sys
import numpy as np
from mpi4py import MPI

# Add Serial Code folder to Python's search path so we can import from it.
SERIAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Serial Code")
sys.path.insert(0, SERIAL_DIR)
WEIGHTS_PATH = os.path.join(SERIAL_DIR, "weights.npz")   # made by train_serial.py

from nn_model import get_weights, net_from_weights   


"""
Decide how many images each process gets and where each process's images start in the big array

If we had 10 rows and 3 processes:
base, extra = (3,1)

process 0 = 3 + 1 = 4
process 1 = 3 + 0 = 3
process 2: 3 + 0 = 3
counts  [4,3,3]

starts = [4, 7, 10] Each process's start is how many rows come before me
Process 0 starts at row 0
Process 1 starts after process 0's 4 rows, at row 4
Process 2 starts after 4+3 rows at row 7
"""
def row_partition(n_rows, nprocs):
    
    base, extra = divmod(n_rows, nprocs)   
    counts = np.array([base + (1 if r < extra else 0) for r in range(nprocs)], dtype=np.int64)
    starts = np.zeros(nprocs, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]    # each start = total rows before it
    return counts, starts

"""
Give every process a copy of rank 0's network, so that everyone has the exact same weights

Only rank 0 passes a real network; the others pass None.
"""

def broadcast_net(net, comm, root=0):
    
    rank = comm.Get_rank()
    weights = get_weights(net) if rank == root else None

    # Step 1: tell everyone the shape of each weight matrix,
    # so they can make empty arrays of the right size.
    shapes = [(W.shape, b.shape) for W, b in weights] if rank == root else None
    shapes = comm.bcast(shapes, root=root)

    # Step 2: send the actual numbers, one layer at a time.
    received = []
    for i, (w_shape, b_shape) in enumerate(shapes):
        if rank == root:
            W = np.ascontiguousarray(weights[i][0], dtype=np.float64)
            b = np.ascontiguousarray(weights[i][1], dtype=np.float64)
        else:
            W = np.empty(w_shape, dtype=np.float64)   # empty array to receive into
            b = np.empty(b_shape, dtype=np.float64)
        comm.Bcast(W, root=root)   # capital B = sends raw numbers (fast)
        comm.Bcast(b, root=root)
        received.append((W, b))

    # Step 3: everyone except rank 0 builds a network from the weights they got.
    return net if rank == root else net_from_weights(received)


def scatter_rows(X, comm, root=0):
    """Split the images (rows of X) between the processes.

    Only rank 0 passes X; the others pass None.
    Returns this process's chunk, plus the counts/starts for every process.
    """
    rank, size = comm.Get_rank(), comm.Get_size()

    # Everyone needs to know the size of X to work out the split.
    shape = X.shape if rank == root else None
    n_rows, n_cols = comm.bcast(shape, root=root)

    counts, starts = row_partition(n_rows, size)
    X_local = np.empty((counts[rank], n_cols), dtype=np.float64)   # space for my chunk

    if rank == root:
        X = np.ascontiguousarray(X, dtype=np.float64)
        # MPI counts single numbers, not rows, so multiply rows by row length.
        sendbuf = [X, counts * n_cols, starts * n_cols, MPI.DOUBLE]
    else:
        sendbuf = None
    comm.Scatterv(sendbuf, X_local, root=root)
    return X_local, counts, starts


def gather_rows(Y_local, counts, starts, n_out, comm, root=0):
    """Collect every process's answers back on rank 0, in the original order."""
    rank = comm.Get_rank()
    Y_local = np.ascontiguousarray(Y_local, dtype=np.float64)
    if rank == root:
        Y = np.empty((int(counts.sum()), n_out), dtype=np.float64)   # space for all answers
        recvbuf = [Y, counts * n_out, starts * n_out, MPI.DOUBLE]
    else:
        Y, recvbuf = None, None
    comm.Gatherv(Y_local, recvbuf, root=root)
    return Y


def parallel_scores(net, X, comm, root=0, timings=None):
    """Parallel version of net.predict_scores(X).

    Every process must already have the network (use broadcast_net first).
    Only rank 0 needs X. Rank 0 gets back the scores; others get None.
    If you pass a dict as `timings`, it records how long each step took.
    """
    n_out = get_weights(net)[-1][0].shape[1]   # number of outputs (10 digits)

    t0 = MPI.Wtime()
    X_local, counts, starts = scatter_rows(X, comm, root)      # 1. split images
    t1 = MPI.Wtime()

    if X_local.shape[0] > 0:
        Y_local = net.predict_scores(X_local)                  # 2. run our normal network
    else:
        Y_local = np.empty((0, n_out), dtype=np.float64)       #    (no images = no work)
    t2 = MPI.Wtime()

    Y = gather_rows(Y_local, counts, starts, n_out, comm, root)  # 3. collect answers
    t3 = MPI.Wtime()

    if timings is not None:
        timings["scatter"] = t1 - t0
        timings["compute"] = t2 - t1
        timings["gather"] = t3 - t2
    return Y


def parallel_predict(net, X, comm, root=0):
    """Parallel version of net.predict(X): the predicted digit for each image."""
    Y = parallel_scores(net, X, comm, root)
    return np.argmax(Y, axis=1) if comm.Get_rank() == root else None   # highest score wins


if __name__ == "__main__":
    # Demo: classify the MNIST test set in parallel.
    #   mpiexec -n 4 python nn_rows_mpi.py
    from nn_model import load_weights
    from mnist_data import load_mnist

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Only rank 0 loads the network and the data.
    net, X, y = None, None, None
    if rank == 0:
        net = net_from_weights(load_weights(WEIGHTS_PATH))
        _, _, X, y = load_mnist()

    net = broadcast_net(net, comm)   # everyone gets the network
    comm.Barrier()                   # start the clock at the same time
    t0 = MPI.Wtime()
    pred = parallel_predict(net, X, comm)
    elapsed = MPI.Wtime() - t0
    if rank == 0:
        print(f"{comm.Get_size()} procs: test accuracy {np.mean(pred == y):.4f}, "
              f"forward time {elapsed:.4f} s for {len(X)} images")