"""
Sweep CG tolerance across all three grid sizes and produce three plots:
  1. Wall-clock time vs cg_tol
  2. SQP iterations vs cg_tol
  3. Total inner CG iterations (summed across all SQP steps) vs cg_tol

Each plot has three lines, one per grid.  X-axis is log-scale cg_tol,
oriented so tighter tolerances are on the right.

Run from the math270c directory:
    python test_cg_sweep.py            # all three grids
    python test_cg_sweep.py --no-64   # skip 64x64x40 (saves ~10-30 min)

Results are saved to output/cg_sweep/results.json so you can re-plot
without re-running the solver.
"""

import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from optimal_transport.test_images import make_images
from optimal_transport.sqp import initialize, sqp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_GRIDS = [(16, 16, 10), (32, 32, 20), (64, 64, 40)]

# CG_MAXITER large enough that the solver is never limited by iteration count.
# Scales with grid size because the Schur system grows as O(n^3).
CG_MAXITER = {16: 300, 32: 750, 64: 2000}

CG_TOLS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

P        = 2
SQP_TOL  = 1e-4
MAX_ITER = 100
CONTRAST = 10

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(grid, cg_tol):
    n1, n2, n3 = grid
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=CONTRAST)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

    t0 = time.perf_counter()
    _, _, _, _, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=P, tol=SQP_TOL, max_iter=MAX_ITER,
        cg_tol=cg_tol, cg_maxiter=CG_MAXITER[n1],
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    sqp_iters = len(stats)
    total_cg  = sum(s['cg_iters'] for s in stats)
    converged = sqp_iters < MAX_ITER

    return dict(
        grid      = list(grid),
        cg_tol    = cg_tol,
        elapsed   = elapsed,
        sqp_iters = sqp_iters,
        total_cg  = total_cg,
        converged = converged,
        kkt_final = stats[-1]['kkt_lam'] if stats else float('nan'),
    )

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(grids):
    all_results = {}

    for grid in grids:
        n1 = grid[0]
        label = f"{grid[0]}x{grid[1]}x{grid[2]}"
        print(f"\n=== Grid {label} (cg_maxiter={CG_MAXITER[n1]}) ===")
        print(f"{'cg_tol':>8}  {'time(s)':>8}  {'SQP':>5}  {'CG_tot':>8}  {'conv':>5}  {'kkt':>10}")
        print("-" * 62)

        results = []
        for i, tol in enumerate(CG_TOLS, 1):
            print(f"  [{i}/{len(CG_TOLS)}] cg_tol={tol:.3f} ...", end=" ", flush=True)
            r = run_one(grid, tol)
            conv = "YES" if r['converged'] else "no"
            print(f"{r['elapsed']:.2f}s  sqp={r['sqp_iters']}  cg={r['total_cg']}  {conv}")
            results.append(r)

        all_results[label] = results

    return all_results

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(all_results, out_dir):
    labels  = list(all_results.keys())
    colors  = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']

    metrics = [
        ('elapsed',    'Wall-clock time (s)',      'Time vs CG tolerance'),
        ('sqp_iters',  'SQP iterations',           'SQP iterations vs CG tolerance'),
        ('total_cg',   'Total inner CG iterations','Total CG work vs CG tolerance'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for label, color, marker in zip(labels, colors, markers):
            results = all_results[label]
            tols = [r['cg_tol'] for r in results]
            vals = [r[key]      for r in results]
            # Mark non-converged points with an open marker
            for t, v, r in zip(tols, vals, results):
                kw = dict(color=color, marker=marker, markersize=7,
                          markerfacecolor=color if r['converged'] else 'white',
                          markeredgecolor=color)
                ax.plot(t, v, **kw)
            ax.plot(tols, vals, color=color, linewidth=1.5, label=label)

        ax.set_xscale('log')
        ax.invert_xaxis()
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('CG tolerance  (log scale, tighter →)')
    fig.suptitle(
        f'CG tolerance sweep  (contrast={CONTRAST}, p={P}, sqp_tol={SQP_TOL})',
        fontsize=12, y=1.005,
    )
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "cg_sweep.png")
    fig.savefig(fig_path, dpi=130, bbox_inches='tight')
    print(f"\nPlot saved to {fig_path}")
    plt.show()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-64", action="store_true",
                        help="Skip the 64x64x40 grid (saves ~10-30 min)")
    args = parser.parse_args()

    grids = ALL_GRIDS[:2] if args.no_64 else ALL_GRIDS

    out_dir = os.path.join("output", "cg_sweep")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Grids: {[f'{g[0]}x{g[1]}x{g[2]}' for g in grids]}")
    print(f"CG tols: {CG_TOLS}")
    print(f"Estimated runs: {len(grids) * len(CG_TOLS)}")

    all_results = run_sweep(grids)

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {json_path}")

    plot(all_results, out_dir)


if __name__ == "__main__":
    main()
