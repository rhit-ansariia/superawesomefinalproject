"""
Serial baseline timing. Runs the normal (serial) forward pass on batches of
different sizes and writes the times to a CSV.

    python3 timing_baseline.py <output.csv> <hidden_width> <batch sizes...>

The CSV uses processes = 0 to mean "serial baseline", the same way the
linsolve labs used threads = 0.

We use random weights here instead of the trained ones. A forward pass takes
the same time no matter what the numbers in the weights are, and this way we
can try wider hidden layers than the trained 784 -> 32 -> 10 network.
The images are real MNIST images.
"""
import sys
import time
import numpy as np

from timing_common import make_net, load_batch, REPEATS


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 timing_baseline.py <output.csv> <hidden_width> <batch sizes...>")
        sys.exit(1)

    out_file = sys.argv[1]
    hidden = int(sys.argv[2])
    batch_sizes = [int(s) for s in sys.argv[3:]]

    net = make_net(hidden)

    with open(out_file, "w") as f:
        f.write("batch_size,hidden,processes,time_seconds,speedup,scatter,compute,gather\n")
        for n in batch_sizes:
            X = load_batch(n)
            net.predict_scores(X)                 # warm up once, don't time it

            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                net.predict_scores(X)
                times.append(time.perf_counter() - t0)

            t = float(np.median(times))           # median so one slow run doesn't skew it
            # serial has no scatter/gather, it's all compute
            f.write(f"{n},{hidden},0,{t:.6e},1.0,0,{t:.6e},0\n")
            print(f"baseline  batch={n:>6}  hidden={hidden}  time={t:.3e}s", flush=True)


if __name__ == "__main__":
    main()
