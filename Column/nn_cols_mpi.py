"""
Runs our neural network on many processes at once using MPI.

Each image is one row of X. The network's answer for an image only depends
on that image, not on any other image. So we can split the images up:

  1. Rank 0 has the entire network.
  2. Rank 0 sends each process its own chunk of the weights and biases of a layer (Scatterv).
  3. Every process runs the every input through its chunk of the network.
  4. Every process sends its answers back to every other processor (Allgatherv).
  5. Steps 3-4 are repeated until we get to the last layer, on which instead of performing an Allgatherv, we do Gatherv to rank 0.
  6. Rank 0 now has the answers for every image, in the right order.

Every process needs the entire input vector, so rank 0 sends a copy of the
inputs to everyone once at the start.

The processes only talk to each other at every layer.

If the weights and biases don't not divide, some processors handle more neurons than others.
If there are fewer neurons than processors, some processors get no neurons.
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
Process 2 starts after 4+3 rows at row 7d
"""

def col_partition(n_cols, nprocs):
    
    base, extra = divmod(n_cols, nprocs)   
    counts = np.array([base + (1 if r < extra else 0) for r in range(nprocs)], dtype=np.int64)
    starts = np.zeros(nprocs, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]    # each start = total rows before it
    return counts, starts

"""
Give every process a copy of rank 0's image, so that everyone has the exact same image

Only rank 0 passes a real image; the others pass None.
"""


def broadcast_inputs(inputs, comm, root=0):
    
    rank = comm.Get_rank()

    shapes = inputs.shape if rank == root else None
    n_rows, n_cols = comm.bcast(shapes, root=root)

    if rank == root:
        X = np.ascontiguousarray(inputs, dtype=np.float64)
    else:
        X = np.empty((n_rows, n_cols), dtype=np.float64)   # empty array to receive into
    comm.Bcast(X, root=root)

    return X


def scatter_cols(net, comm, root=0):
    """Split the network (columns of X) between the processes.

    Only rank 0 passes X; the others pass None.
    Returns this process's chunk, plus the counts/starts for every process.
    """
    rank, size = comm.Get_rank(), comm.Get_size()
    weights = get_weights(net) if rank == root else None

    # Everyone needs to know the size of X to work out the split.
    shapes = [W.shape for W, b in weights] if rank == root else None
    shapes = comm.bcast(shapes, root=root)

    local = []
    for i, (n_in, n_out) in enumerate(shapes):
        counts, starts = col_partition(n_out, size)
        mine = counts[rank]

        if rank == root:
            WT = np.ascontiguousarray(weights[i][0].T, dtype=np.float64)
            b = np.ascontiguousarray(weights[i][1], dtype=np.float64).ravel()

            w_send = [WT, counts * n_in, starts * n_in, MPI.DOUBLE]
            b_send = [b, counts, starts, MPI.DOUBLE]
        else:
            w_send, b_send = None, None

        WT_local = np.empty((mine, n_in), dtype=np.float64)   # space for my rows of W.T
        b_local = np.empty(mine, dtype=np.float64)            # space for my biases
        comm.Scatterv(w_send, WT_local, root=root)
        comm.Scatterv(b_send, b_local, root=root)

        W_local = np.ascontiguousarray(WT_local.T)            # (n_in, my_neurons)
        local.append((W_local, b_local.reshape(1, -1), counts, starts))

    return local

def allgather_cols(Z_local, counts, starts, comm):
    n_rows = Z_local.shape[0]
    n_cols = int(counts.sum())
    sendbuf = np.ascontiguousarray(Z_local.T, dtype = np.float64)
    full_T = np.empty((n_cols, n_rows), dtype=np.float64)

    comm.Allgatherv(sendbuf, [full_T, counts * n_rows, starts * n_rows, MPI.DOUBLE])
    return full_T.T

def gather_cols(Z_local, counts, starts, comm, root=0):
    """Collect every process's neurons' outputs on rank 0, in the original order."""
    rank = comm.Get_rank()
    n_rows = Z_local.shape[0]
    n_cols = int(counts.sum())
    sendbuf = np.ascontiguousarray(Z_local.T, dtype=np.float64)
    if rank == root:
        full_T = np.empty((n_cols, n_rows), dtype=np.float64)     # space for all answers
        recvbuf = [full_T, counts * n_rows, starts * n_rows, MPI.DOUBLE]
    else:
        full_T, recvbuf = None, None
    comm.Gatherv(sendbuf, recvbuf, root=root)
    return np.ascontiguousarray(full_T.T) if rank == root else None




def parallel_scores(local_net, X, comm, root=0, timings=None):
    """Parallel version of net.predict_scores(X).

    Every process must already have the network (use broadcast_net first).
    Only rank 0 needs X. Rank 0 gets back the scores; others get None.
    If you pass a dict as `timings`, it records how long each step took.
    """
    n_layers = len(local_net)

    t0 = MPI.Wtime()
    A = broadcast_inputs(X, comm, root)
    t1 = MPI.Wtime()
    
    compute_time = 0.0
    comm_time = 0.0
    Y = None
    for i, (W_local, b_local, counts, starts) in enumerate(local_net):
        last = (i == n_layers - 1)

        ta = MPI.Wtime()
        Z = np.dot(A, W_local) + b_local
        if not last:
            np.maximum(Z, 0, out=Z) #ReLU
        tb = MPI.Wtime()

        if not last:
            A = allgather_cols(Z, counts, starts, comm)
        else:
            Y = gather_cols(Z, counts, starts, comm) #for final layer activation. handled by root node
        tc = MPI.Wtime()

        compute_time += tb - ta
        comm_time += tc - tb


    if timings is not None:
        timings["scatter"] = t1 - t0
        timings["compute"] = compute_time
        timings["gather"] = comm_time
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

    local_net = scatter_cols(net, comm)
    comm.Barrier()                   # start the clock at the same time
    t0 = MPI.Wtime()
    pred = parallel_predict(local_net, X, comm)
    elapsed = MPI.Wtime() - t0
    if rank == 0:
        print(f"{comm.Get_size()} procs: test accuracy {np.mean(pred == y):.4f}, "
              f"forward time {elapsed:.4f} s for {len(X)} images")