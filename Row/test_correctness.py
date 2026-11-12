"""
Tests to make sure our parallel network gives the same answers as the
serial one.

Put this in the Row folder next to nn_rows_mpi.py and run it with a few
different process counts (odd ones too):

    for n in 1 2 3 4 7; do mpiexec -n $n python test_correctness.py; done

What we check:
  1. row_partition splits the rows right (no rows lost, chunks are the
     same size or off by one).
  2. After broadcast_net, every process has the same weights as rank 0.
  3. parallel_scores gives the same scores as net.predict_scores for a
     bunch of batch sizes, including ones that don't split evenly and
     ones smaller than the number of processes.
  4. On the real MNIST test set, the parallel version predicts the same
     digit for every image and gets the same accuracy as the serial one.

Rank 0 prints PASS or FAIL for each test. If anything fails the script
exits with code 1.
"""
import os
import sys
import numpy as np
from mpi4py import MPI

# nn_rows_mpi adds the Serial Code folder to the path when we import it,
# so the nn_model and mnist_data imports below work.
from nn_rows_mpi import (row_partition, broadcast_net, parallel_scores,
                         parallel_predict, WEIGHTS_PATH)
from nn_model import get_weights, net_from_weights, load_weights
from mnist_data import load_mnist

# How close the scores have to be. Splitting the rows up can change the
# last few decimal places a tiny bit, so we don't check for exact equality.
RTOL, ATOL = 1e-10, 1e-12


def report(name, passed, detail=""):
    """Print a PASS/FAIL line. Only rank 0 calls this."""
    line = f"{'PASS' if passed else 'FAIL'}  {name}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)
    return passed


def random_net(layer_sizes, seed):
    """Make a network with random weights. Only rank 0 calls this."""
    rng = np.random.default_rng(seed)
    weights = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        b = rng.standard_normal(n_out) * 0.1
        weights.append((W, b))
    return net_from_weights(weights)


"""
Test 1: row_partition

This is just math, no MPI, so only rank 0 runs it.
We try a lot of row/process combos and check:
  - the counts add up to the total number of rows
  - the biggest and smallest chunk are off by at most 1
  - each start is right where the last chunk ended
"""
def test_row_partition():
    ok = True
    for n_rows in [0, 1, 2, 7, 10, 97, 10000]:
        for nprocs in [1, 2, 3, 4, 7, 16]:
            counts, starts = row_partition(n_rows, nprocs)
            good = (counts.sum() == n_rows
                    and counts.max() - counts.min() <= 1
                    and starts[0] == 0
                    and np.all(starts[1:] == starts[:-1] + counts[:-1]))
            if not good:
                report("row_partition", False,
                       f"n_rows={n_rows}, nprocs={nprocs}, counts={counts}, starts={starts}")
                ok = False
    if ok:
        report("row_partition", True)
    return ok


"""
Test 2: broadcast_net

Rank 0 makes a random network and sends it out. Then everyone sends their
copy of the weights back to rank 0 so it can check they all match.
"""
def test_broadcast(comm):
    rank = comm.Get_rank()
    net = random_net([784, 128, 64, 10], seed=1) if rank == 0 else None
    net = broadcast_net(net, comm)

    all_weights = comm.gather(get_weights(net), root=0)
    if rank != 0:
        return True

    ref = all_weights[0]
    for r in range(1, len(all_weights)):
        for (W, b), (W0, b0) in zip(all_weights[r], ref):
            if not (np.array_equal(W, W0) and np.array_equal(b, b0)):
                return report("broadcast_net", False, f"rank {r} has different weights")
    return report("broadcast_net", True)


"""
Test 3: parallel scores vs serial scores

Rank 0 makes a random network and some random images, we run them through
the parallel version, and then rank 0 runs the same images through the
serial network to compare.
"""
def test_scores(comm, name, layer_sizes, batch_size, seed=2):
    rank = comm.Get_rank()
    net, X = None, None
    if rank == 0:
        net = random_net(layer_sizes, seed)
        X = np.random.default_rng(seed + 1).standard_normal((batch_size, layer_sizes[0]))
    net = broadcast_net(net, comm)

    Y_par = parallel_scores(net, X, comm)

    if rank != 0:
        return True

    Y_ser = net.predict_scores(X)
    if Y_par is None or Y_par.shape != Y_ser.shape:
        got = None if Y_par is None else Y_par.shape
        return report(name, False, f"got shape {got}, expected {Y_ser.shape}")

    if not np.allclose(Y_par, Y_ser, rtol=RTOL, atol=ATOL):
        # find which rows are wrong, helps figure out which process messed up
        row_ok = np.all(np.isclose(Y_par, Y_ser, rtol=RTOL, atol=ATOL), axis=1)
        bad_rows = np.where(~row_ok)[0]
        return report(name, False, f"{len(bad_rows)} rows wrong, first one is row {bad_rows[0]}")

    return report(name, True, f"batch={batch_size}")


"""
Test 4: real MNIST with our trained weights

This is the main check from our plan. Same images, same weights, and the
parallel version should predict the same digit for every image and get the
same accuracy as serial.

If weights.npz isn't there yet we skip this one. Every rank has to agree
on skipping, otherwise some would wait on an MPI call forever.
"""
def test_mnist(comm):
    rank = comm.Get_rank()

    have_weights = os.path.exists(WEIGHTS_PATH) if rank == 0 else None
    have_weights = comm.bcast(have_weights, root=0)
    if not have_weights:
        if rank == 0:
            print(f"SKIP  MNIST test (can't find {WEIGHTS_PATH}, run train_serial.py first)")
        return True

    net, X, y = None, None, None
    if rank == 0:
        net = net_from_weights(load_weights(WEIGHTS_PATH))
        _, _, X, y = load_mnist()
    net = broadcast_net(net, comm)

    pred_par = parallel_predict(net, X, comm)

    if rank != 0:
        return True

    pred_ser = net.predict(X)
    n_diff = int(np.sum(pred_par != pred_ser))
    acc_par = np.mean(pred_par == y)
    acc_ser = np.mean(pred_ser == y)

    ok1 = report("MNIST: same prediction for every image", n_diff == 0,
                 f"{n_diff} of {len(X)} are different")
    ok2 = report("MNIST: same accuracy", acc_par == acc_ser,
                 f"serial {acc_ser:.4f}, parallel {acc_par:.4f}")
    return ok1 and ok2


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(f"Running tests with {size} process(es)\n", flush=True)

    mnist_shape = [784, 128, 64, 10]

    # (name, layer sizes, batch size)
    # Some of these depend on the number of processes so we hit the
    # even split, uneven split, and more-processes-than-images cases
    # no matter what -n we run with.
    score_cases = [
        ("scores: even split", mnist_shape, 8 * size),
        ("scores: uneven split", mnist_shape, 8 * size + 3),
        ("scores: fewer images than processes", mnist_shape, max(1, size - 1)),
        ("scores: one image", mnist_shape, 1),
        ("scores: prime batch size", mnist_shape, 97),
        ("scores: one layer", [784, 10], 50),
        ("scores: deep network", [64, 64, 64, 64, 64, 10], 40),
        ("scores: wide layer", [784, 1024, 10], 32),
    ]

    all_passed = True
    try:
        if rank == 0:
            all_passed &= test_row_partition()
        all_passed &= test_broadcast(comm)
        for name, layers, batch in score_cases:
            all_passed &= test_scores(comm, name, layers, batch)
        all_passed &= test_mnist(comm)
    except Exception as e:
        # If one process crashes the rest get stuck waiting on it,
        # so kill all of them instead of hanging.
        print(f"[rank {rank}] ERROR: {e!r}", flush=True)
        comm.Abort(1)

    # Only rank 0 actually knows if everything passed, so tell the others
    # and have everyone exit with the same code.
    all_passed = comm.bcast(all_passed, root=0)
    if rank == 0:
        print("\nALL TESTS PASSED" if all_passed else "\nSOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()