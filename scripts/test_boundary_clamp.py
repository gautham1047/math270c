"""
Diagnose and fix boundary-flux leakage in the asymmetric Gaussian test case.

The divergence operator in grid.py includes boundary momentum faces as free
variables, so the optimizer can route mass through the domain edge.  For the
asymmetric Gaussian (source at bottom-left, target at top-right) this shows
up as the blob wrapping around rather than travelling through the interior.

Fix: multiply mu0/mu1 by a cosine taper that forces density to zero within a
border strip, removing the optimizer's incentive to use boundary flux.

Outputs: output/boundary_clamp/comparison.png
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimal_transport.test_images import make_asymmetric_gaussian
from optimal_transport.sqp import initialize, sqp

_REPO  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUTDIR = os.path.join(_REPO, 'output', 'boundary_clamp')
os.makedirs(OUTDIR, exist_ok=True)


def cosine_taper(n1, n2, border_frac=0.15):
    """
    (n1, n2) window: 1 in the interior, cosine ramp to 0 within border_frac
    of each edge (applied independently per axis, then multiplied).
    """
    x = np.linspace(0.5 / n1, 1.0 - 0.5 / n1, n1)
    y = np.linspace(0.5 / n2, 1.0 - 0.5 / n2, n2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    def ramp1d(t):
        r = np.ones_like(t)
        lo = t < border_frac
        hi = t > 1.0 - border_frac
        r[lo] = 0.5 * (1 - np.cos(np.pi * t[lo] / border_frac))
        r[hi] = 0.5 * (1 - np.cos(np.pi * (1 - t[hi]) / border_frac))
        return r

    return ramp1d(X) * ramp1d(Y)


def solve(mu0, mu1, n3, h):
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
    m1, m2, rho_free, lam, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=2, tol=1e-4, verbose=False
    )
    rho_full = np.empty((mu0.shape[0], mu0.shape[1], n3 + 1))
    rho_full[:, :, 0]    = mu0
    rho_full[:, :, 1:-1] = rho_free
    rho_full[:, :, -1]   = mu1
    return m1, m2, rho_full, stats


if __name__ == '__main__':
    n1, n2, n3 = 16, 16, 10
    h = 1.0 / n1

    # --- original ---
    mu0_orig, mu1_orig = make_asymmetric_gaussian(n1, n2)
    m1_o, m2_o, rho_orig, stats_o = solve(mu0_orig, mu1_orig, n3, h)
    inner_o = sum(s['inner_iters'] for s in stats_o)
    print(f'original : {len(stats_o)} SQP iters | {inner_o} inner | '
          f'|C|={stats_o[-1]["kkt_lam"]:.2e}')

    # --- clamped ---
    w = cosine_taper(n1, n2, border_frac=0.15)
    mu0_c = mu0_orig * w
    mu1_c = mu1_orig * w
    mu1_c = mu1_c * (mu0_c.sum() / mu1_c.sum())   # re-match mass after taper

    m1_c, m2_c, rho_clamp, stats_c = solve(mu0_c, mu1_c, n3, h)
    inner_c = sum(s['inner_iters'] for s in stats_c)
    print(f'clamped  : {len(stats_c)} SQP iters | {inner_c} inner | '
          f'|C|={stats_c[-1]["kkt_lam"]:.2e}')

    # --- boundary flux diagnostic ---
    def mean_bflux(m1, m2):
        return {
            'left':   float(np.mean(np.abs(m1[0,   :, :]))),
            'right':  float(np.mean(np.abs(m1[-1,  :, :]))),
            'bottom': float(np.mean(np.abs(m2[:,  0, :]))),
            'top':    float(np.mean(np.abs(m2[:, -1, :]))),
        }

    bf_o = mean_bflux(m1_o, m2_o)
    bf_c = mean_bflux(m1_c, m2_c)
    print(f'\nMean |boundary flux| (averaged over opposite axis and time):')
    print(f'  {"":8s}  {"left":>10}  {"right":>10}  {"bottom":>10}  {"top":>10}')
    for lbl, bf in [('original', bf_o), ('clamped', bf_c)]:
        print(f'  {lbl:8s}  '
              f'{bf["left"]:10.4f}  {bf["right"]:10.4f}  '
              f'{bf["bottom"]:10.4f}  {bf["top"]:10.4f}')

    # --- comparison image ---
    snap_idx = [0, n3 // 4, n3 // 2, 3 * n3 // 4, n3]
    vmax = max(rho_orig.max(), rho_clamp.max())

    fig, axes = plt.subplots(2, len(snap_idx), figsize=(len(snap_idx) * 3, 6))

    for row, (rho, lbl) in enumerate([(rho_orig,  'Original (no clamp)'),
                                       (rho_clamp, 'Clamped (border 15%)')]):
        for col, t in enumerate(snap_idx):
            ax = axes[row, col]
            im = ax.imshow(rho[:, :, t].T, origin='lower',
                           vmin=0, vmax=vmax, cmap='viridis',
                           extent=[0, 1, 0, 1])
            ax.set_title(f't = {t / n3:.2f}', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(lbl, fontsize=9)

    fig.suptitle('Asymmetric Gaussian: boundary clamp comparison (16x16x10)', y=1.01)
    plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    plt.tight_layout()

    path = os.path.join(OUTDIR, 'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nsaved {path}')
