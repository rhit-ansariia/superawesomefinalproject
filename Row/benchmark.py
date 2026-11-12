"""
benchmark.py
Times our serial forward pass vs the row-split MPI version and figures out the speedup.

How to run:
    mpiexec -n 4 python benchmark.py --mode strong --images 10000 --hidden 32
    mpiexec -n 4 python benchmark.py --mode weak --images 2500 --hidden 1024

strong scaling = total images stay the same, we just add more processes.
    if it works the time should go down.
weak scaling = each process always gets the same number of images, so the
    total goes up as we add processes. ideally the time stays flat.

We use random weights here instead of the trained ones. the time for a matrix
multiply doesn't care what the numbers are, and this way we can try bigger
hidden layers without having to retrain anything.

"""
import argparse
import os
import time
import numpy as np
from mpi4py import MPI

# nn_rows_mpi has to be imported first since it adds the Serial Code folder to the path
from nn_rows_mpi import broadcast_net, parallel_scores
from nn_model import build_net
from mnist_data import mnist_available, load_mnist

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


# command line options
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["strong", "weak"], default="strong")
ap.add_argument("--images", type=int, default=10000, help="total (strong) or per-process (weak)")
ap.add_argument("--hidden", type=int, default=32, help="hidden layer size (we trained with 32)")
ap.add_argument("--layers", type=int, default=1, help="number of hidden layers")
ap.add_argument("--reps", type=int, default=5, help="how many timed runs to average")
ap.add_argument("--warmup", type=int, default=1, help="untimed runs first")
ap.add_argument("--out", default="results.csv")
args = ap.parse_args()

# weak mode gives every process its own batch, so multiply by the number of processes
total = args.images if args.mode == "strong" else args.images * size
sizes = [784] + [args.hidden] * args.layers + [10]


# setup (rank 0 only)
# only rank 0 makes the network and loads the images.
# everyone else gets theirs later through MPI.
net, X, source = None, None, None
if rank == 0:
    np.random.seed(0)
    net = build_net(sizes)
    if mnist_available():
        _, _, X_test, _ = load_mnist()
        # the test set only has 10000 images, so if we need more we just repeat it
        reps_needed = -(-total // len(X_test))   # this is just ceiling division
        X = np.tile(X_test, (reps_needed, 1))[:total]
        source = "mnist"
    else:
        # no mnist file found, use random pixels so it still runs
        X = np.random.rand(total, 784)
        source = "random"


# send weights to every process
# this only happens once, so we time it on its own and don't add it to the
# parallel time
comm.Barrier()
t0 = MPI.Wtime()
net = broadcast_net(net, comm)
bcast_time = comm.reduce(MPI.Wtime() - t0, op=MPI.MAX, root=0)


# serial baseline
# rank 0 runs all the images by itself. this is what we compare against.
serial_mean, Y_ref = None, None
if rank == 0:
    # warmup run first, the first call is usually slower for no real reason
    for _ in range(args.warmup):
        net.predict_scores(X)

    times = []
    for _ in range(args.reps):
        t = time.perf_counter()
        Y_ref = net.predict_scores(X)
        times.append(time.perf_counter() - t)
    serial_mean = float(np.mean(times))   # average over the runs to cut down on noise


# parallel version
for _ in range(args.warmup):
    parallel_scores(net, X, comm)

keys = ["total", "scatter", "compute", "gather"]
sums = {k: 0.0 for k in keys}

for _ in range(args.reps):
    timings = {}

    # barrier so all the processes start the clock at the same time
    comm.Barrier()
    t = MPI.Wtime()
    Y = parallel_scores(net, X, comm, timings=timings)
    timings["total"] = MPI.Wtime() - t

    # each process has its own times. we take the max because the run
    # isn't actually done until the slowest process finishes.
    for k in keys:
        sums[k] += comm.allreduce(timings[k], op=MPI.MAX)


# results (rank 0 only)
if rank == 0:
    # make sure parallel gave the same answer as serial before we report anything
    assert np.allclose(Y, Y_ref), "parallel result does not match serial!"

    avg = {k: sums[k] / args.reps for k in keys}
    speedup = serial_mean / avg["total"]   # ex. 3.0 means 3x faster
    eff = speedup / size                   # 1.0 (100%) would be perfect

    print(f"mode={args.mode} P={size} images={total} sizes={sizes} inputs={source}")
    print(f"  serial   {serial_mean:.5f} s")
    print(f"  parallel {avg['total']:.5f} s  (scatter {avg['scatter']:.5f}, "
          f"compute {avg['compute']:.5f}, gather {avg['gather']:.5f})")
    print(f"  bcast    {bcast_time:.5f} s (one-time, not in parallel time)")
    print(f"  speedup  {speedup:.2f}x   efficiency {eff:.2%}")

    # add a row to the csv. run_scaling.sh calls this once per process count,
    # so all the runs end up in one file that plot_results.py reads.
    header = ("mode,procs,images,hidden,layers,reps,serial_s,parallel_s,scatter_s,"
              "compute_s,gather_s,bcast_s,speedup,efficiency\n")
    new_file = not os.path.exists(args.out)
    with open(args.out, "a") as f:
        if new_file:
            f.write(header)   # header only goes in the first time
        f.write(f"{args.mode},{size},{total},{args.hidden},{args.layers},{args.reps},"
                f"{serial_mean:.6f},{avg['total']:.6f},{avg['scatter']:.6f},{avg['compute']:.6f},"
                f"{avg['gather']:.6f},{bcast_time:.6f},{speedup:.4f},{eff:.4f}\n")