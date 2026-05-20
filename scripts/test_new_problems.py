"""
Run the three new test problems (asymmetric Gaussian, ring->disk, two blobs)
across three grid sizes to verify mesh independence, and generate density GIFs.

Outputs (written to output/new_problems/):
  <case>_convergence.png   |C| vs SQP iteration for 16x16x10, 32x32x20, 64x64x40
  <case>.gif               Density animation at 32x32x20

Usage:
  python scripts/test_new_problems.py            # full run (slow on 64x64x40)
  python scripts/test_new_problems.py --smoke    # 16x16x10 only, verbose
  python scripts/test_new_problems.py --no-64   # skip 64x64x40
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from optimal_transport.test_images import (
    make_asymmetric_gaussian, make_ring_to_disk, make_two_blobs
)
from optimal_transport.sqp import initialize, sqp

# ---------------------------------------------------------------------------
_REPO    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUTDIR   = os.path.join(_REPO, 'output', 'new_problems')
GRIDS    = [(16, 16, 10), (32, 32, 20), (64, 64, 40)]
GIF_GRID = (16, 16, 10)

CASES = [
    ('asymmetric_gaussian', 'Asymmetric Gaussian',  make_asymmetric_gaussian),
    ('ring_to_disk',        'Ring to Disk',         make_ring_to_disk),
    ('two_blobs',           'Two Blobs to Centre',  make_two_blobs),
]
# ---------------------------------------------------------------------------


def run(make_fn, n1, n2, n3, verbose=False):
    h = 1.0 / n1
    mu0, mu1 = make_fn(n1, n2)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
    m1, m2, rho_free, lam, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=2, tol=1e-4, verbose=verbose, method='gmres_amg'
    )
    rho_full = np.empty((n1, n2, n3 + 1))
    rho_full[:, :, 0]    = mu0
    rho_full[:, :, 1:-1] = rho_free
    rho_full[:, :, -1]   = mu1
    return m1, m2, rho_full, stats, mu0, mu1


def plot_convergence(case_name, title, all_stats, grids):
    """|C| vs SQP iteration for three grids — mesh independence check."""
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    fig, ax = plt.subplots(figsize=(6, 4))
    for (n1, n2, n3), stats, color in zip(grids, all_stats, colors):
        kkt = [s['kkt_lam'] for s in stats]
        ax.semilogy(range(1, len(kkt) + 1), kkt,
                    color=color, marker='o', markersize=4,
                    label=f'{n1}×{n2}×{n3}')
    ax.set_xlabel('SQP iteration')
    ax.set_ylabel('|C| (constraint violation)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, f'{case_name}_convergence.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  saved {path}')


def make_gif(case_name, rho_full, fps=8):
    n3   = rho_full.shape[2] - 1
    vmin = 0.0
    vmax = rho_full.max()

    fig, ax = plt.subplots(figsize=(4, 4))
    im  = ax.imshow(rho_full[:, :, 0].T, origin='lower',
                    vmin=vmin, vmax=vmax, cmap='viridis',
                    extent=[0, 1, 0, 1])
    ttl = ax.set_title('t = 0.00')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(t):
        im.set_data(rho_full[:, :, t].T)
        ttl.set_text(f't = {t / n3:.2f}')
        return [im, ttl]

    ani = animation.FuncAnimation(
        fig, update, frames=n3 + 1, interval=1000 // fps, blit=True
    )
    path = os.path.join(OUTDIR, f'{case_name}.gif')
    ani.save(path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f'  saved {path}')


def smoke_test():
    """Quick 16x16x10 run for all three cases with verbose output."""
    n1, n2, n3 = 16, 16, 10
    for case_name, title, make_fn in CASES:
        print(f'\n=== SMOKE: {title} ({n1}x{n2}x{n3}) ===')
        _, _, _, stats, _, _ = run(make_fn, n1, n2, n3, verbose=True)
        inner_total = sum(s['inner_iters'] for s in stats)
        print(f'  -> {len(stats)} SQP iters | {inner_total} total inner | '
              f'|C|_final={stats[-1]["kkt_lam"]:.2e}')


def full_run(grids=None):
    if grids is None:
        grids = GRIDS
    os.makedirs(OUTDIR, exist_ok=True)

    for case_name, title, make_fn in CASES:
        print(f'\n=== {title} ===')
        all_stats = []

        for (n1, n2, n3) in grids:
            print(f'  {n1}x{n2}x{n3} ...', end=' ', flush=True)
            _, _, _, stats, _, _ = run(make_fn, n1, n2, n3)
            inner = sum(s['inner_iters'] for s in stats)
            print(f'{len(stats)} SQP iters | {inner} inner | '
                  f'|C|={stats[-1]["kkt_lam"]:.2e}')
            all_stats.append(stats)

        # plot_convergence(case_name, title, all_stats, grids)

        if GIF_GRID in grids:
            n1, n2, n3 = GIF_GRID
            print(f'  GIF at {n1}x{n2}x{n3} ...', end=' ', flush=True)
            _, _, rho_full, _, _, _ = run(make_fn, n1, n2, n3)
            make_gif(case_name, rho_full)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-64', action='store_true', help='Skip 64x64x40')
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    else:
        grids = GRIDS[:2] if args.no_64 else GRIDS
        full_run(grids)
