"""
Parallel (row split) timing. Run it with mpiexec at one process count; the
Makefile calls it once for each count.

    mpiexec -n 4 python3 timing_rows.py <output.csv> <baseline.csv> <hidden_width> <batch sizes...>

For each batch size it times parallel_scores (scatter + compute + gather),
repeats it REPEATS times, and keeps the median. Speedup is
baseline time / parallel time, using the baseline CSV.

Sending the network to every process (broadcast_net) happens once before
timing starts, the same way you'd load a model once and then run it on
lots of batches. So it isn't counted in the times.

It also saves how long scatter, compute and gather took separately (on
rank 0), so we can see how much of the time is communication.
"""
import sys
import csv
import numpy as np
from mpi4py import MPI

from timing_common import make_net, load_batch, REPEATS
from nn_rows_mpi import broadcast_net, parallel_scores


def read_baseline(path, hidden):
    """batch_size -> serial time, from the baseline CSV."""
    base = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["hidden"]) == hidden and int(row["processes"]) == 0:
                base[int(row["batch_size"])] = float(row["time_seconds"])
    return base


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if len(sys.argv) < 5:
        if rank == 0:
            print("Usage: mpiexec -n P python3 timing_rows.py <output.csv> <baseline.csv> <hidden_width> <batch sizes...>")
        sys.exit(1)

    out_file, base_file = sys.argv[1], sys.argv[2]
    hidden = int(sys.argv[3])
    batch_sizes = [int(s) for s in sys.argv[4:]]

    # Rank 0 builds the network and sends it to everyone, once.
    net = make_net(hidden) if rank == 0 else None
    net = broadcast_net(net, comm)

    baseline = read_baseline(base_file, hidden) if rank == 0 else None
    f = open(out_file, "w") if rank == 0 else None
    if rank == 0:
        f.write("batch_size,hidden,processes,time_seconds,speedup,scatter,compute,gather\n")

    for n in batch_sizes:
        X = load_batch(n) if rank == 0 else None
        parallel_scores(net, X, comm)            # warm up once, don't time it

        times, parts = [], []
        for _ in range(REPEATS):
            timings = {}
            comm.Barrier()                       # everyone starts the clock together
            t0 = MPI.Wtime()
            parallel_scores(net, X, comm, timings=timings)
            elapsed = MPI.Wtime() - t0
            # the run is only done when the slowest process is done
            elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)
            if rank == 0:
                times.append(elapsed)
                parts.append((timings["scatter"], timings["compute"], timings["gather"]))

        if rank == 0:
            t = float(np.median(times))
            scatter, compute, gather = np.median(np.array(parts), axis=0)
            speedup = baseline[n] / t
            f.write(f"{n},{hidden},{size},{t:.6e},{speedup:.4f},"
                    f"{scatter:.6e},{compute:.6e},{gather:.6e}\n")
            print(f"{size:>2} procs  batch={n:>6}  hidden={hidden}  time={t:.3e}s  "
                  f"speedup={speedup:.2f}x", flush=True)

    if rank == 0:
        f.close()


if __name__ == "__main__":
    main()
