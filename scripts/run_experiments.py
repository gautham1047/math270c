"""
Unified experiment runner for Haber & Horesh (2015) + new test problems.

Usage:
    python scripts/run_experiments.py                       # all, CG, all grids
    python scripts/run_experiments.py --no-64               # skip 64x64x40
    python scripts/run_experiments.py --method gmres_amg    # GMRES+AMG path
    python scripts/run_experiments.py --method both         # compare CG vs GMRES
    python scripts/run_experiments.py --exp 1 3 new         # specific subset
    python scripts/run_experiments.py --exp 1 --no-64 --method both
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimal_transport.test_images import (
    make_images, make_asymmetric_gaussian, make_ring_to_disk, make_two_blobs,
)
from optimal_transport.sqp import initialize, sqp
from optimal_transport.experiments import (
    experiment1, experiment2, experiment3, experiment5,
)

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

GRIDS_ALL  = [(16, 16, 10), (32, 32, 20), (64, 64, 40)]
GRIDS_NO64 = [(16, 16, 10), (32, 32, 20)]

NEW_CASES = [
    ('asymmetric_gaussian', 'Asymmetric Gaussian',  make_asymmetric_gaussian),
    ('ring_to_disk',        'Ring to Disk',         make_ring_to_disk),
    ('two_blobs',           'Two Blobs to Centre',  make_two_blobs),
]


# ---------------------------------------------------------------------------
# New test problems (inline, supports grids override)
# ---------------------------------------------------------------------------

def run_new_problems(grids, verbose=False, method='cg_sgs'):
    out_dir = os.path.join(_REPO, 'output', 'new_problems')
    os.makedirs(out_dir, exist_ok=True)

    for case_name, title, make_fn in NEW_CASES:
        print(f'\n=== New problem: {title} ===')
        all_stats = []

        for (n1, n2, n3) in grids:
            h = 1.0 / n1
            mu0, mu1 = make_fn(n1, n2)
            m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
            print(f'  {n1}x{n2}x{n3} ...', end=' ', flush=True)
            m1, m2, rho_free, lam, stats = sqp(
                m1, m2, rho_free, lam, mu0, mu1, h,
                p=2, tol=1e-4, verbose=verbose, method=method
            )
            inner_total = sum(s['inner_iters'] for s in stats)
            print(f'{len(stats)} SQP | {inner_total} inner | '
                  f'|C|={stats[-1]["kkt_lam"]:.2e}')
            all_stats.append(stats)

        colors = ['tab:blue', 'tab:orange', 'tab:green']
        fig, ax = plt.subplots(figsize=(6, 4))
        for (n1, n2, n3), stats, color in zip(grids, all_stats, colors):
            kkt = [s['kkt_lam'] / s['scale_lam'] for s in stats]
            ax.semilogy(range(1, len(kkt) + 1), kkt,
                        color=color, marker='o', markersize=4,
                        label=f'{n1}×{n2}×{n3}')
        ax.set_xlabel('SQP iteration')
        ax.set_ylabel('|C|')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f'{case_name}_convergence.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  saved {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--no-64', action='store_true',
        help='Skip 64x64x40 grids (also skips exp2/3/5 which all use large grids)',
    )
    parser.add_argument(
        '--method', choices=['cg_sgs', 'gmres_amg', 'both'], default='cg_sgs',
        help='Solver method (default: cg_sgs)',
    )
    parser.add_argument(
        '--exp', nargs='+',
        choices=['1', '2', '3', '5', 'new'],
        default=['1', '2', '3', '5', 'new'],
        help='Which experiments to run (default: all)',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    grids   = GRIDS_NO64 if args.no_64 else GRIDS_ALL
    methods = ['cg_sgs', 'gmres_amg'] if args.method == 'both' else [args.method]
    exps    = set(args.exp)

    # ------------------------------------------------------------------
    if '1' in exps:
        for method in methods:
            sep = '=' * 60
            print(f'\n{sep}\nExperiment 1: mesh independence  (method={method})\n{sep}')
            experiment1(verbose=args.verbose, method=method, grids=grids)

    # ------------------------------------------------------------------
    if '2' in exps:
        for method in methods:
            if args.no_64:
                print('\nSkipping experiment 2 (42x42x42) — use without --no-64.')
            else:
                sep = '=' * 60
                print(f'\n{sep}\nExperiment 2: density evolution (42x42x42, contrast=10)\n{sep}')
                experiment2(verbose=args.verbose, method=method)

    # ------------------------------------------------------------------
    if '3' in exps:
        for method in methods:
            if args.no_64:
                print('\nSkipping experiment 3 (multilevel up to 64x64x32) — use without --no-64.')
            else:
                sep = '=' * 60
                print(f'\n{sep}\nExperiment 3: multilevel, contrast=100\n{sep}')
                experiment3(verbose=args.verbose, method=method)

    # ------------------------------------------------------------------
    if '5' in exps:
        for method in methods:
            if args.no_64:
                print('\nSkipping experiment 5 (64x64x32 p-continuation) — use without --no-64.')
            else:
                sep = '=' * 60
                print(f'\n{sep}\nExperiment 5: p-continuation (p=2 → 1.01)\n{sep}')
                experiment5(verbose=args.verbose, method=method)

    # ------------------------------------------------------------------
    if 'new' in exps:
        for method in methods:
            sep = '=' * 60
            print(f'\n{sep}\nNew problems: asymmetric Gaussian, ring→disk, two blobs\n{sep}')
            run_new_problems(grids, verbose=args.verbose, method=method)


if __name__ == '__main__':
    main()
