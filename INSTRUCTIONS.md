# How to Run

Our code is in Python (NumPy + mpi4py), so there's nothing to compile. Everything runs through the Makefile in "Row/", and the ".pbs" files submit it to the cluster with "qsub", same as the labs.

Run everything from the "Row/" folder or from the Columns/ folder.

## Requirements

You need "python3" with "numpy" and "mpi4py", and "mpiexec". To check:

```bash
python3 -c "import numpy, mpi4py"
```

If mpi4py isn't installed:

```bash
python3 -m pip install --user mpi4py
```

If the cluster uses "mpirun" instead of "mpiexec", pass it to make:

```bash
make test MPIEXEC=mpirun
```

## Running on the cluster

```bash
cd Row
qsub run_tests.pbs    # correctness tests
qsub jobfile.pbs      # timing runs + speedup plots
```

Output goes to "output.txt" and errors go to "error.txt".

## Correctness tests

```bash
make test
```

Runs "test_correctness.py" with 1, 2, 4, 8, and 12 processes and checks that the parallel version gets the same predictions and accuracy as the serial version. It stops at the first failure.

## Timing

```bash
make time         # 784 -> 32 -> 10 (the trained network)
make time_wide    # 784 -> 1024 -> 10 (wider hidden layer)
```

Each one times the serial baseline and then the row-parallel version at 1, 2, 4, 8, and 12 processes, for batch sizes 1, 10, 100, 1000, 10000, and 60000 images. The results are combined into one CSV, and "analyze_rows.py" makes the speedup plot.

- "make time" writes "rows_hidden32_timing_results.csv" and its plot
- "make time_wide" writes "rows_hidden1024_timing_results.csv" and its plot

"jobfile.pbs" runs both on one node with 12 cores.

The Makefile sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS to 1 so NumPy doesn't multithread behind the scenes. Without that, the serial baseline would use every core and the speedup numbers would be off. So run things through make instead of calling the Python scripts directly.

## Other commands

- "make help" lists the targets
- "make clean" deletes all the CSVs and PNGs in "Row/"

You can change the process counts or batch sizes from the command line, for example:

```bash
make time PROCS="1 2 4"
```