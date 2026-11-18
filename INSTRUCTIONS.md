# How to Run

Our code is in Python (NumPy + mpi4py), so there's nothing to compile. Everything runs through the Makefile in "Row/" or "Column/", and the ".pbs" files submit it to the cluster with "qsub", same as the labs.

Run everything from the "Row/" folder or from the "Column/" folder.

## Requirements

You need "python3", "mpiexec", and these Python packages:

- numpy
- mpi4py
- pandas (used by the analysis scripts)
- matplotlib (used for the plots)

On grendel, "python3" is at "/usr/bin/python3" and MPI is at "/opt/openmpi". There's no "module" command and no plain "python", so always use "python3".

### Setup (one time, on the login node)

Put MPI on your path. You can add these two lines to your "~/.bashrc" so you don't have to do it every time:

```bash
export PATH=/opt/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH
```

Install the packages:

```bash
python3 -m pip install --user numpy pandas matplotlib
MPICC=/opt/openmpi/bin/mpicc python3 -m pip install --user mpi4py
```

The "MPICC=..." part tells pip which MPI to build mpi4py against. Without it, the install might fail or link to the wrong MPI.

Check that everything works:

```bash
which python3 mpiexec
python3 -c "import numpy, mpi4py, pandas, matplotlib; print('all good')"
```

### The job files

Compute nodes don't load your ".bashrc" settings automatically, so the ".pbs" files set the MPI paths themselves and print a quick check at the top of "output.txt". If you see "numpy ok" and "mpi4py ok" there, the environment is fine.

### Common problems

- "qsub: script contains cr, lf": the file has Windows line endings (from editing on Windows or VS Code). Fix it with:
  ```bash
  sed -i 's/\r$//' *.pbs *.py Makefile
  ```
  In VS Code, click "CRLF" in the bottom right and switch it to "LF" before saving.
- "python: command not found": use "python3".
- "module: command not found": grendel doesn't have modules. Delete any "module load" lines.

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

For the column version, "cd Column" and use "jobfile.pbs". Uncomment "make test", "make time", or "make time_wide" in the file depending on what you want to run.

Check on the job with "qstat -u $USER". Output goes to "output.txt" and errors go to "error.txt".

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

Each one times the serial baseline and then the row-parallel (or column-parallel, if that was chosen) version at 1, 2, 4, 8, and 12 processes, for batch sizes 1, 10, 100, 1000, 10000, and 60000 images. The results are combined into one CSV, and "analyze_rows.py" (or "analyze_cols.py") makes the speedup plot.

For the row-parallel version:
- "make time" writes "rows_hidden32_timing_results.csv" and its plot
- "make time_wide" writes "rows_hidden1024_timing_results.csv" and its plot

For the column-parallel version:
- "make time" writes "cols_hidden32_timing_results.csv" and its plot
- "make time_wide" writes "cols_hidden1024_timing_results.csv" and its plot

"jobfile.pbs" runs on one node with 12 cores.

The Makefile sets OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS to 1 so NumPy doesn't multithread behind the scenes. Without that, the serial baseline would use every core and the speedup numbers would be off. So run things through make instead of calling the Python scripts directly.

If you only want to remake a plot (like after changing a title), you don't need to rerun the timing. Just run the analysis script on the existing CSV:

```bash
python3 analyze_cols.py cols_hidden1024_timing_results.csv
```

## Other commands

- "make help" lists the targets
- "make clean" deletes all the CSVs and PNGs in "Row/" or "Column/"

You can change the process counts or batch sizes from the command line, for example:

```bash
make time PROCS="1 2 4"
```