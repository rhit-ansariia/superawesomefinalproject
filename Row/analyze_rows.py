#!/usr/bin/env python3
"""
Row split timing analysis. Reads the timing CSV and makes the same 4-panel
plot as the linsolve labs, plus a summary table.

    python3 analyze_rows.py <input.csv>

Panels:
  1. time vs batch size, one line per process count (plus serial baseline)
  2. speedup vs serial baseline, by batch size
  3. strong scaling at the biggest batch (vs 1-process parallel)
  4. parallel efficiency at the biggest batch (vs 1-process parallel)
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no screen on the cluster
import matplotlib.pyplot as plt

COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']


def create_plots(df, output_prefix):
    hidden = int(df['hidden'].iloc[0])
    baseline = df[df['processes'] == 0]
    parallel = df[df['processes'] > 0]

    sizes = sorted(df['batch_size'].unique())
    proc_counts = sorted(parallel['processes'].unique())

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Row Split Forward Pass Performance (784 → {hidden} → 10 network)',
                 fontsize=16, fontweight='bold')

    # Plot 1: time vs batch size
    ax1.loglog(baseline['batch_size'], baseline['time_seconds'], marker='s', linewidth=3,
               markersize=8, color='black', linestyle='--', label='Baseline (serial)')
    for i, p in enumerate(proc_counts):
        d = parallel[parallel['processes'] == p]
        ax1.loglog(d['batch_size'], d['time_seconds'], marker='o',
                   color=COLORS[i % len(COLORS)], label=f'Row split ({p} process{"es" if p > 1 else ""})')
    ax1.set_xlabel('Batch Size (images)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time vs Batch Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: speedup vs serial baseline
    for i, p in enumerate(proc_counts):
        if p == 1:
            continue
        d = parallel[parallel['processes'] == p]
        ax2.semilogx(d['batch_size'], d['speedup'], marker='s',
                     color=COLORS[i % len(COLORS)], label=f'{p} processes')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xlabel('Batch Size (images)')
    ax2.set_ylabel('Speedup vs Serial Baseline')
    ax2.set_title('Speedup vs Serial Baseline by Batch Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: strong scaling at the biggest batch, vs 1-process parallel
    largest = max(sizes)
    big = parallel[parallel['batch_size'] == largest].sort_values('processes')
    one = big[big['processes'] == 1]['time_seconds']
    one_time = one.iloc[0] if not one.empty else 1.0
    strong = one_time / big['time_seconds']

    ax3.plot(big['processes'], strong, marker='o', linewidth=2, markersize=8, color='red',
             label='Row split')
    ax3.plot(big['processes'], big['processes'], linestyle='--', alpha=0.5, color='black',
             label='Ideal linear speedup')
    ax3.set_xlabel('Number of Processes')
    ax3.set_ylabel('Speedup vs 1-process Parallel')
    ax3.set_title(f'Strong Scaling - Parallel Only (batch of {largest} images)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(proc_counts)

    # Plot 4: efficiency at the biggest batch
    eff = strong / big['processes'] * 100
    ax4.plot(big['processes'], eff, marker='o', linewidth=2, markersize=8, color='green')
    ax4.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Perfect efficiency')
    ax4.set_xlabel('Number of Processes')
    ax4.set_ylabel('Parallel Efficiency (%)')
    ax4.set_title(f'Parallel Efficiency vs 1-process (batch of {largest} images)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(proc_counts)
    ax4.set_ylim(0, max(110, eff.max() * 1.1))

    plt.tight_layout()
    plot_file = f'{output_prefix}_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    plt.close()


def print_summary_table(df):
    baseline = df[df['processes'] == 0]
    parallel = df[df['processes'] > 0]
    sizes = sorted(df['batch_size'].unique())
    proc_counts = sorted(parallel['processes'].unique())

    print("\n" + "=" * 110)
    print(f"ROW SPLIT TIMING SUMMARY (hidden width {int(df['hidden'].iloc[0])})")
    print("=" * 110)
    header = f"{'Batch':<10} {'Baseline':<12}" + "".join(f"{f'{p} procs':<17}" for p in proc_counts)
    print(header)
    print(f"{'':<10} {'Time (s)':<12}" + "".join(f"{'Time/Speedup':<17}" for _ in proc_counts))
    print("-" * 110)
    for n in sizes:
        b = baseline[baseline['batch_size'] == n]
        row = f"{n:<10} {(f'{b.time_seconds.iloc[0]:.1e}' if not b.empty else 'N/A'):<12}"
        for p in proc_counts:
            d = parallel[(parallel['batch_size'] == n) & (parallel['processes'] == p)]
            cell = f"{d.time_seconds.iloc[0]:.1e}/{d.speedup.iloc[0]:.2f}x" if not d.empty else "N/A"
            row += f"{cell:<17}"
        print(row)

    # How much of the time is communication, at the biggest batch
    largest = max(sizes)
    big = parallel[parallel['batch_size'] == largest].sort_values('processes')
    print(f"\nTIME BREAKDOWN AT BATCH {largest} (rank 0, seconds):")
    print("-" * 70)
    print(f"{'Procs':<8} {'Scatter':<14} {'Compute':<14} {'Gather':<14} {'% comm':<10}")
    for _, r in big.iterrows():
        total = r.scatter + r.compute + r.gather
        comm_pct = 100 * (r.scatter + r.gather) / total if total > 0 else 0
        print(f"{int(r.processes):<8} {r.scatter:<14.2e} {r.compute:<14.2e} {r.gather:<14.2e} {comm_pct:<10.1f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_rows.py <input.csv>")
        sys.exit(1)
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    print_summary_table(df)
    create_plots(df, csv_file.replace('.csv', ''))


if __name__ == "__main__":
    main()
