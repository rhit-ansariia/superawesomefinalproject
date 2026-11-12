# ============================================================================
# Row Split Neural Network - Makefile
# ============================================================================

# Python and MPI commands. Change these if your cluster uses different names
# (for example MPIEXEC = mpirun).
PYTHON  = python3
MPIEXEC = mpiexec

# Make numpy use 1 thread. Otherwise the "serial" baseline secretly uses
# every core through BLAS, and the MPI runs fight over cores.
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1
export MKL_NUM_THREADS = 1

# Process counts to test. The jobfile asks for 12 cores (ppn=12).
PROCS = 1 2 4 8 12

# Batch sizes (number of images) for timing. MNIST has 60,000 training images.
BATCH_SIZES = 1 10 100 1000 10000 60000

# Hidden layer width for timing. 32 is the size of the trained network.
# time_wide uses a much wider layer so there's more math per image.
HIDDEN      = 32
WIDE_HIDDEN = 1024

# ================
# Targets
# ================

help:
	@echo "============================================================================"
	@echo "Row Split Neural Network - Available make Commands"
	@echo "============================================================================"
	@echo ""
	@echo "  make test       - Correctness tests at every process count in PROCS"
	@echo "  make time       - Timing + speedup plot, hidden width $(HIDDEN)"
	@echo "  make time_wide  - Timing + speedup plot, hidden width $(WIDE_HIDDEN)"
	@echo "  make clean      - Remove CSVs and plots"
	@echo ""

# Correctness tests: run test_correctness.py at every process count.
# Stops at the first failure.
test:
	@for p in $(PROCS); do \
		echo "=== $$p process(es) ==="; \
		$(MPIEXEC) -n $$p $(PYTHON) test_correctness.py || exit 1; \
	done

# Timing for one hidden width. Called by time and time_wide.
#   1. serial baseline
#   2. parallel at each process count
#   3. combine into one CSV and make the plot
define run_timing
	@echo "Running row split timing analysis (hidden width $(1))..."
	@echo "Generating baseline results..."
	$(PYTHON) timing_baseline.py baseline_$(1).csv $(1) $(BATCH_SIZES)
	@echo "Generating parallel results..."
	@for p in $(PROCS); do \
		$(MPIEXEC) -n $$p $(PYTHON) timing_rows.py rows_$(1)_$$p.csv baseline_$(1).csv $(1) $(BATCH_SIZES) || exit 1; \
	done
	@echo "Combining results for analysis..."
	@cat baseline_$(1).csv > rows_hidden$(1)_timing_results.csv
	@for p in $(PROCS); do tail -n +2 rows_$(1)_$$p.csv >> rows_hidden$(1)_timing_results.csv; done
	@rm -f baseline_$(1).csv rows_$(1)_*.csv
	$(PYTHON) analyze_rows.py rows_hidden$(1)_timing_results.csv
endef

time:
	$(call run_timing,$(HIDDEN))

time_wide:
	$(call run_timing,$(WIDE_HIDDEN))

clean:
	rm -f *.csv *.png

.PHONY: help test time time_wide clean