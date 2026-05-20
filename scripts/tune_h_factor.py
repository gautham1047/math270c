"""
Sweep and binary-search max_h_factor for CG+SGS and/or GMRES+AMG paths.

Produces a plot of SQP iterations vs max_h_factor (open markers = did not
converge within MAX_ITER steps).

Usage:
    python scripts/tune_h_factor.py                   # 16x16x10, both methods
    python scripts/tune_h_factor.py --grid 32         # 32x32x20
    python scripts/tune_h_factor.py --method cg_sgs   # CG only
    python scripts/tune_h_factor.py --no-binary-search
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimal_transport.test_images import make_images
from optimal_transport.sqp import initialize, sqp

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

_GRIDS   = {16: (16, 16, 10), 32: (32, 32, 20), 64: (64, 64, 40)}
H_FACTORS = [1.0, 1.2, 1.5, 1.8, 2.0, 3.0, 5.0, float('inf')]

P        = 2
SQP_TOL  = 1e-4
MAX_ITER = 100
CONTRAST = 10

BSEARCH_LO  = 1.0
BSEARCH_HI  = 4.0
BSEARCH_EPS = 0.05

_METHOD_LABEL = {'cg_sgs': 'CG+SGS', 'gmres_amg': 'GMRES+AMG'}
_METHOD_COLOR = {'cg_sgs': 'tab:blue', 'gmres_amg': 'tab:orange'}
_METHOD_MARKER = {'cg_sgs': 'o', 'gmres_amg': 's'}


def run_one(grid, method, max_h_factor):
    n1, n2, n3 = grid
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=CONTRAST)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

    t0 = time.perf_counter()
    _, _, _, _, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=P, tol=SQP_TOL, max_iter=MAX_ITER,
        max_h_factor=max_h_factor, method=method,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    n_iter    = len(stats)
    n_acc     = sum(1 for s in stats if s['accepted'])
    reasons   = [s['reject_reason'] for s in stats if not s['accepted']]
    converged = n_iter < MAX_ITER

    return dict(
        max_h_factor = max_h_factor,
        elapsed      = elapsed,
        sqp_iters    = n_iter,
        converged    = converged,
        n_accepted   = n_acc,
        n_rejected   = n_iter - n_acc,
        rej_h_inc    = reasons.count('h_increase'),
        rej_filter   = reasons.count('filter'),
        kkt_final    = stats[-1]['kkt_lam'] if stats else float('nan'),
    )


def sweep(grid, method):
    label = f"{grid[0]}x{grid[1]}x{grid[2]}"
    print(f"\n--- Sweep [{_METHOD_LABEL[method]}] grid {label} ---")
    print(f"{'h_factor':>10}  {'time(s)':>8}  {'SQP':>5}  {'conv':>5}  {'acc':>5}  {'rej':>5}  "
          f"{'h_inc':>6}  {'filt':>5}  {'kkt_lam':>10}")
    print("-" * 80)

    results = []
    for hf in H_FACTORS:
        r = run_one(grid, method, hf)
        hf_str   = f"{hf:.1f}" if hf != float('inf') else "inf"
        conv_str = "YES" if r['converged'] else "no"
        print(f"{hf_str:>10}  {r['elapsed']:>8.2f}  {r['sqp_iters']:>5d}  {conv_str:>5}  "
              f"{r['n_accepted']:>5d}  {r['n_rejected']:>5d}  "
              f"{r['rej_h_inc']:>6d}  {r['rej_filter']:>5d}  "
              f"{r['kkt_final']:>10.3e}")
        results.append(r)
    return results


def binary_search(grid, method):
    label = f"{grid[0]}x{grid[1]}x{grid[2]}"
    lo, hi = BSEARCH_LO, BSEARCH_HI
    print(f"\n--- Binary search [{_METHOD_LABEL[method]}] grid {label} ---")
    print(f"  Range [{lo}, {hi}], eps={BSEARCH_EPS}")
    print(f"  {'h_factor':>10}  {'SQP':>5}  {'conv':>5}  {'kkt_lam':>10}")
    print("  " + "-" * 42)

    while hi - lo > BSEARCH_EPS:
        mid = (lo + hi) / 2
        r   = run_one(grid, method, mid)
        tag = "YES" if r['converged'] else "no"
        print(f"  {mid:>10.4f}  {r['sqp_iters']:>5d}  {tag:>5}  {r['kkt_final']:>10.3e}")
        if r['converged']:
            hi = mid
        else:
            lo = mid

    print(f"\n  Convergence threshold ~ {hi:.4f}  ({_METHOD_LABEL[method]})")
    return hi


def plot_results(sweep_data, grid_label, out_dir):
    methods = list(sweep_data.keys())
    fig, (ax_time, ax_sqp) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    for method in methods:
        results = sweep_data[method]
        color   = _METHOD_COLOR[method]
        marker  = _METHOD_MARKER[method]
        label   = _METHOD_LABEL[method]

        # Replace inf with a sentinel x-value slightly past the last finite point
        hf_finite = [r['max_h_factor'] for r in results if r['max_h_factor'] != float('inf')]
        sentinel  = (max(hf_finite) * 1.4) if hf_finite else 6.0
        hf_vals  = [r['max_h_factor'] if r['max_h_factor'] != float('inf') else sentinel
                    for r in results]
        times    = [r['elapsed']   for r in results]
        sqp_vals = [r['sqp_iters'] for r in results]
        convs    = [r['converged'] for r in results]

        kw = dict(color=color, linewidth=1.5, label=label)
        ax_time.plot(hf_vals, times,    marker=marker, **kw)
        ax_sqp.plot(hf_vals,  sqp_vals, marker=marker, **kw)

        for hf, t, s, conv in zip(hf_vals, times, sqp_vals, convs):
            mfc = color if conv else 'white'
            ax_time.plot(hf, t, color=color, marker=marker, markersize=9,
                         markerfacecolor=mfc, markeredgecolor=color)
            ax_sqp.plot(hf,  s, color=color, marker=marker, markersize=9,
                        markerfacecolor=mfc, markeredgecolor=color)

    for ax in (ax_time, ax_sqp):
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax_time.set_ylabel('Wall-clock time (s)')
    ax_time.set_title('Time vs max_h_factor')
    ax_sqp.set_ylabel('SQP iterations')
    ax_sqp.set_title('SQP iterations vs max_h_factor')
    ax_sqp.set_xlabel(f'max_h_factor  (rightmost point = inf)')

    fig.suptitle(f'max_h_factor sweep — {grid_label}\n'
                 f'(contrast={CONTRAST}, p={P}, open markers = did not converge)',
                 fontsize=11)
    plt.tight_layout()

    path = os.path.join(out_dir, f'h_factor_sweep_{grid_label}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved to {path}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--grid', type=int, choices=[16, 32, 64], default=16)
    parser.add_argument('--method', choices=['cg_sgs', 'gmres_amg', 'both'],
                        default='both')
    parser.add_argument('--no-binary-search', action='store_true',
                        help='Skip binary search (faster, sweep only)')
    args = parser.parse_args()

    grid      = _GRIDS[args.grid]
    methods   = ['cg_sgs', 'gmres_amg'] if args.method == 'both' else [args.method]
    grid_lbl  = f"{grid[0]}x{grid[1]}x{grid[2]}"

    out_dir = os.path.join(_REPO, 'output', 'tune_h_factor')
    os.makedirs(out_dir, exist_ok=True)

    sweep_data = {}
    for method in methods:
        sweep_data[method] = sweep(grid, method)
        if not args.no_binary_search:
            binary_search(grid, method)

    plot_results(sweep_data, grid_lbl, out_dir)


if __name__ == '__main__':
    main()
