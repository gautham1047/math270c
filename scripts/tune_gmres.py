"""
Golden section search for the optimal GMRES inner tolerance (gmres_tol).

GMRES typically converges in 2-7 iterations for this problem regardless of
tolerance, so the search identifies the looser tolerance at which the outer
SQP loop starts to need more iterations.

Produces a figure with three panels:
  - Wall-clock time vs gmres_tol
  - SQP iterations vs gmres_tol
  - GMRES iterations (mid SQP step) vs gmres_tol

Usage:
    python scripts/tune_gmres.py                  # 16x16x10 (fast)
    python scripts/tune_gmres.py --grid 32        # 32x32x20
    python scripts/tune_gmres.py --all-grids      # 16 + 32  (or 16+32+64 without --no-64)
    python scripts/tune_gmres.py --no-64          # skip 64x64x40
    python scripts/tune_gmres.py --reps 3         # avg timing over 3 reps
"""

import sys, os, argparse, time, math, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimal_transport.test_images import make_images
from optimal_transport.sqp import initialize, sqp

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

ALL_GRIDS     = [(16, 16, 10), (32, 32, 20), (64, 64, 40)]
_GRIDS        = {16: (16, 16, 10), 32: (32, 32, 20), 64: (64, 64, 40)}
GMRES_MAXITER = {16: 50, 32: 50, 64: 50}

P        = 2
SQP_TOL  = 1e-4
MAX_ITER = 50
CONTRAST = 10

LOG_LO  = math.log10(5e-3)   # gmres_tol = 0.005
LOG_HI  = math.log10(0.5)    # gmres_tol = 0.5
GSS_TOL = 0.08
PHI     = (math.sqrt(5) - 1) / 2


# ---------------------------------------------------------------------------
# Single timed run
# ---------------------------------------------------------------------------

def run_one(grid, gmres_tol, gmres_maxiter, reps=1):
    """Return (wall-clock time, sqp_iters, inner_iters_list, converged)."""
    n1, n2, n3 = grid
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=CONTRAST)

    times      = []
    stats_last = []
    for _ in range(reps):
        m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
        t0 = time.perf_counter()
        _, _, _, _, stats = sqp(
            m1, m2, rho_free, lam, mu0, mu1, h,
            p=P, tol=SQP_TOL, max_iter=MAX_ITER,
            method='gmres_amg',
            gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
            verbose=False,
        )
        times.append(time.perf_counter() - t0)
        stats_last = stats

    elapsed     = min(times)
    sqp_iters   = len(stats_last)
    inner_iters = [s['inner_iters'] for s in stats_last]
    converged   = sqp_iters < MAX_ITER
    return elapsed, sqp_iters, inner_iters, converged


# ---------------------------------------------------------------------------
# Golden section search
# ---------------------------------------------------------------------------

def golden_section_search(grid, gmres_maxiter, reps):
    label = f"{grid[0]}x{grid[1]}x{grid[2]}"
    print(f"\n=== GSS [{label}] over gmres_tol ===")
    print(f"  Range: [{10**LOG_LO:.4f}, {10**LOG_HI:.3f}]  "
          f"(stop when interval < {GSS_TOL} log10-units)")
    print(f"  {'step':>4}  {'lo':>7}  {'hi':>7}  {'c':>9}  {'d':>9}  "
          f"{'t(c)':>7}  {'t(d)':>7}  {'better':>6}")
    print("  " + "-" * 68)

    cache = {}

    def evaluate(log_tol):
        if log_tol not in cache:
            tol = 10 ** log_tol
            t, sqp, iters, conv = run_one(grid, tol, gmres_maxiter, reps)
            cache[log_tol] = (t, sqp, iters, conv)
        return cache[log_tol][0]

    lo, hi = LOG_LO, LOG_HI
    c = hi - PHI * (hi - lo)
    d = lo + PHI * (hi - lo)
    step = 0

    while (hi - lo) > GSS_TOL:
        step += 1
        tc = evaluate(c)
        td = evaluate(d)
        better = "c" if tc < td else "d"
        print(f"  {step:>4}  {lo:>7.3f}  {hi:>7.3f}  "
              f"{c:>9.4f}  {d:>9.4f}  "
              f"{tc:>7.3f}s  {td:>7.3f}s  {better:>6}")
        if tc < td:
            hi = d; d, td = c, tc; c = hi - PHI * (hi - lo)
        else:
            lo = c; c, tc = d, td; d = lo + PHI * (hi - lo)

    opt_log = (lo + hi) / 2
    opt_tol = 10 ** opt_log
    opt_t   = evaluate(opt_log)
    print(f"\n  Converged. Optimal: gmres_tol ~ {opt_tol:.4f}  (time ~ {opt_t:.3f}s)")

    print(f"\n  All evaluated points:")
    print(f"  {'gmres_tol':>10}  {'time(s)':>8}  {'SQP':>5}  "
          f"{'GMRES(1,m,n)':>14}  {'conv':>5}")
    print("  " + "-" * 55)
    for lt in sorted(cache):
        t, sqp_n, iters, conv = cache[lt]
        n = len(iters)
        if n > 0:
            i_str = f"({iters[0]},{iters[n//2]},{iters[-1]})"
        else:
            i_str = "(n/a)"
        print(f"  {10**lt:>10.4f}  {t:>8.3f}  {sqp_n:>5d}  {i_str:>14}  "
              f"{'YES' if conv else 'no':>5}")

    return opt_tol, cache


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_sweep(grids, all_caches, out_dir):
    colors  = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    ax_time, ax_sqp, ax_gmres = axes

    for grid, cache, color, marker in zip(grids, all_caches, colors, markers):
        label    = f"{grid[0]}×{grid[1]}×{grid[2]}"
        log_tols = sorted(cache)
        tols     = [10**lt for lt in log_tols]
        times    = [cache[lt][0] for lt in log_tols]
        sqps     = [cache[lt][1] for lt in log_tols]
        convs    = [cache[lt][3] for lt in log_tols]

        # GMRES iters at the median SQP step
        gmres_mid = []
        for lt in log_tols:
            iters = cache[lt][2]
            n = len(iters)
            gmres_mid.append(iters[n // 2] if n > 0 else 0)

        kw = dict(color=color, linewidth=1.5, label=label)
        ax_time.plot(tols, times, marker=marker, **kw)
        ax_sqp.plot(tols, sqps, marker=marker, **kw)
        ax_gmres.plot(tols, gmres_mid, marker=marker, **kw)

        for tol, t, s, g, conv in zip(tols, times, sqps, gmres_mid, convs):
            mfc = color if conv else 'white'
            for ax, val in [(ax_time, t), (ax_sqp, s), (ax_gmres, g)]:
                ax.plot(tol, val, color=color, marker=marker, markersize=8,
                        markerfacecolor=mfc, markeredgecolor=color)

    for ax in axes:
        ax.set_xscale('log')
        ax.invert_xaxis()
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax_time.set_ylabel('Wall-clock time (s)')
    ax_time.set_title('Time vs gmres_tol')
    ax_sqp.set_ylabel('SQP iterations')
    ax_sqp.set_title('Outer SQP iterations vs gmres_tol')
    ax_gmres.set_ylabel('GMRES iters (median SQP step)')
    ax_gmres.set_title('Inner GMRES iterations vs gmres_tol')
    axes[-1].set_xlabel('gmres_tol  (log scale, tighter →)')

    fig.suptitle(
        f'GMRES tolerance sweep  (contrast={CONTRAST}, p={P}, '
        f'sqp_tol={SQP_TOL}, open = did not converge)',
        fontsize=11,
    )
    plt.tight_layout()

    path = os.path.join(out_dir, 'gmres_tol_sweep.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved to {path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--grid', type=int, choices=[16, 32, 64], default=16,
                        help='Single grid to search (overridden by --all-grids)')
    parser.add_argument('--all-grids', action='store_true',
                        help='Run GSS on all grids (respects --no-64)')
    parser.add_argument('--no-64', action='store_true',
                        help='Skip 64x64x40')
    parser.add_argument('--reps', type=int, default=2,
                        help='Timing repetitions per evaluation (min is kept)')
    args = parser.parse_args()

    if args.all_grids:
        grids = [(16, 16, 10), (32, 32, 20)] if args.no_64 else ALL_GRIDS
    else:
        grids = [_GRIDS[args.grid]]

    out_dir = os.path.join(_REPO, 'output', 'tune_gmres')
    os.makedirs(out_dir, exist_ok=True)

    all_caches = []
    for grid in grids:
        gmres_maxiter = GMRES_MAXITER[grid[0]]
        opt_tol, cache = golden_section_search(grid, gmres_maxiter, args.reps)
        all_caches.append(cache)

        # Save raw results
        grid_str = f"{grid[0]}x{grid[1]}x{grid[2]}"
        json_path = os.path.join(out_dir, f'gmres_sweep_{grid_str}.json')
        serializable = {
            str(k): [v[0], v[1], v[2], bool(v[3])]
            for k, v in cache.items()
        }
        with open(json_path, 'w') as fh:
            json.dump(serializable, fh, indent=2)
        print(f'Raw results saved to {json_path}')

    plot_sweep(grids, all_caches, out_dir)


if __name__ == '__main__':
    main()
