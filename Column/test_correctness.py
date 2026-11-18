"""
Tests to make sure our parallel network gives the same answers as the
serial one.

Put this in the Column folder next to nn_cols_mpi.py and run it with a few
different process counts (odd ones too):

    for n in 1 2 3 4 7; do mpiexec -n $n python test_correctness.py; done

What we check:
  1. col_partition splits the neurons right (no rows lost, chunks are the
     same size or off by one).
  2. broadcast_inputs, every process has the same images as rank 0.
  3. scatter_cls gives every process its own columns of each W and b per layer. 
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

# nn_cols_mpi adds the Serial Code folder to the path when we import it,
# so the nn_model and mnist_data imports below work.
from nn_cols_mpi import (col_partition, broadcast_inputs, scatter_cols, parallel_scores,
                         parallel_predict, WEIGHTS_PATH)
from nn_model import get_weights, net_from_weights, load_weights
from mnist_data import load_mnist

# How close the scores have to be. Splitting the cols up can change the
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

def all_ranks_ok(ok, comm):
    """Collect every rank's pass/fail on rank 0.
    Rank 0 gets the list of ranks that failed; others get None."""
    flags = comm.gather(ok, root=0)
    if comm.Get_rank() != 0:
        return None
    return [r for r, f in enumerate(flags) if not f]


"""
Test 1: col_partition

This is just math, no MPI, so only rank 0 runs it.
We try a lot of row/process combos and check:
  - the counts add up to the total number of neurons
  - the biggest and smallest chunk are off by at most 1
  - each start is right where the last chunk ended
"""
def test_col_partition():
    ok = True
    for n_cols in [0, 1, 2, 7, 10, 97, 10000]:
        for nprocs in [1, 2, 3, 4, 7, 16]:
            counts, starts = col_partition(n_cols, nprocs)
            good = (counts.sum() == n_cols
                    and counts.max() - counts.min() <= 1
                    and starts[0] == 0
                    and np.all(starts[1:] == starts[:-1] + counts[:-1]))
            if not good:
                report("col_partition", False,
                       f"n_rows={n_cols}, nprocs={nprocs}, counts={counts}, starts={starts}")
                ok = False
    if ok:
        report("col_partition", True)
    return ok


"""
Test 2: broadcast_inputs

Rank 0 makes a random image and sends it out. Then everyone sends their
copy of the image back to rank 0 so it can check they all match.
"""

def test_broadcast_inputs(comm):
    rank = comm.Get_rank()
    ref = np.random.default_rng(5).standard_normal((37, 784))
    X = broadcast_inputs(ref if rank == 0 else None, comm)

    ok = X.shape == ref.shape and np.array_equal(X, ref)
    bad = all_ranks_ok(ok, comm)
    if rank != 0:
        return True
    if bad:
        return report("broadcast_inputs", False, f"rank(s) {bad} got different images")
    
    return report("broadcast_inputs", True)

"""
Test 3: scatter_cols

Rank 0 creates a random network and scatters it across the other processors.
Every rank builds the same network and verifies its portion.
"""

def test_scatter(comm):
    rank = comm.Get_rank()
    net = random_net([784, 128, 64, 10], seed=1)

    local_net = scatter_cols(net if rank == 0 else None, comm)


    ok = len(local_net) == len(get_weights(net))
    for (W, b), (W_local, b_local, counts, starts) in zip(get_weights(net), local_net):
        first, stop = starts[rank], starts[rank] + counts[rank]
        if not (np.array_equal(W_local, W[:, first:stop])
                and np.array_equal(b_local, b[:, first:stop])):
            ok = False

    bad = all_ranks_ok(ok, comm)
    if rank != 0:
        return True
    if bad:
        return report("scatter_cols", False, f"rank(s) {bad} got the wrong columns")
    return report("scatter_cols", True)


"""
Test 4: parallel scores vs serial scores

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
    local_net = scatter_cols(net, comm)

    Y_par = parallel_scores(local_net, X, comm)

    if rank != 0:
        return True

    Y_ser = net.predict_scores(X)
    if Y_par is None or Y_par.shape != Y_ser.shape:
        got = None if Y_par is None else Y_par.shape
        return report(name, False, f"got shape {got}, expected {Y_ser.shape}")

    if not np.allclose(Y_par, Y_ser, rtol=RTOL, atol=ATOL):
        # find which rows are wrong, helps figure out which process messed up
        col_ok = np.all(np.isclose(Y_par, Y_ser, rtol=RTOL, atol=ATOL), axis=1)
        bad_cols = np.where(~col_ok)[0]
        return report(name, False, f"{len(bad_cols)} rows wrong, first one is row {bad_cols[0]}")

    return report(name, True, f"batch={batch_size}")


"""
Test 5: real MNIST with our trained weights

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
    local_net = scatter_cols(net, comm)

    pred_par = parallel_predict(local_net, X, comm)

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
        ("scores: one image", mnist_shape, 1),
        ("scores: prime batch size", mnist_shape, 97),
        ("scores: one layer", [784, 10], 50),
        ("scores: deep network", [64, 64, 64, 64, 64, 10], 40),
        ("scores: wide layer", [784, 1024, 10], 32),
        ("scores: layers narrower than procs", [20, 3, 5, 2], 16),
    ]

    all_passed = True
    try:
        if rank == 0:
            all_passed &= test_col_partition()
        all_passed &= test_broadcast_inputs(comm)
        all_passed &= test_scatter(comm)
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