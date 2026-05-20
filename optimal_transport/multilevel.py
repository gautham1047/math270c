"""
Multilevel warm-start: interpolate a coarse-grid solution to a finer grid.
Each variable is interpolated at its own staggered location.
"""

import numpy as np
from scipy.ndimage import zoom
from .sqp import initialize, sqp
from .test_images import make_images


def prolong_solution(m1_c, m2_c, rho_free_c, lam_c, n1_f, n2_f, n3_f):
    """
    Bilinear (order=1) zoom from coarse to fine grid.

    Coarse shapes:
      m1_c       : (n1_c+1, n2_c,   n3_c  )
      m2_c       : (n1_c,   n2_c+1, n3_c  )
      rho_free_c : (n1_c,   n2_c,   n3_c-1)
      lam_c      : (n1_c,   n2_c,   n3_c  )
    """
    n3_c = lam_c.shape[2]

    m1_f = zoom(m1_c,
                ((n1_f + 1) / m1_c.shape[0],
                 n2_f       / m1_c.shape[1],
                 n3_f       / m1_c.shape[2]),
                order=1)

    m2_f = zoom(m2_c,
                (n1_f       / m2_c.shape[0],
                 (n2_f + 1) / m2_c.shape[1],
                 n3_f       / m2_c.shape[2]),
                order=1)

    rho_f = zoom(rho_free_c,
                 (n1_f       / rho_free_c.shape[0],
                  n2_f       / rho_free_c.shape[1],
                  (n3_f - 1) / rho_free_c.shape[2]),
                 order=1)

    lam_f = zoom(lam_c,
                 (n1_f / lam_c.shape[0],
                  n2_f / lam_c.shape[1],
                  n3_f / lam_c.shape[2]),
                 order=1)

    return m1_f, m2_f, rho_f, lam_f


def multilevel_sqp(levels, contrast, p=2, tol=1e-4,
                   cg_tol=1e-4, cg_maxiter=500,
                   max_iter=100, verbose=True, method='cg_sgs'):
    """
    Run SQP on successive grid levels with warm-starting.

    levels : list of (n1, n2, n3) tuples, coarse to fine
    Returns the fine-grid solution (m1, m2, rho_free, lam, stats_list).
    """
    m1 = m2 = rho_free = lam = None
    all_stats = []

    for idx, (n1, n2, n3) in enumerate(levels):
        h        = 1.0 / n1          # assume n1 == n2 == n3 or use uniform h
        mu0, mu1 = make_images(n1, n2, contrast)

        if idx == 0:
            m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
        else:
            m1, m2, rho_free, lam = prolong_solution(
                m1, m2, rho_free, lam, n1, n2, n3
            )
            # Clamp rho to stay positive after interpolation
            rho_free = np.maximum(rho_free, 1e-8)

        if verbose:
            print(f"\n=== Level {idx+1}: grid ({n1},{n2},{n3}) ===")

        m1, m2, rho_free, lam, stats = sqp(
            m1, m2, rho_free, lam, mu0, mu1, h, p=p,
            tol=tol, max_iter=max_iter,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            verbose=verbose, method=method
        )
        all_stats.append(stats)

    return m1, m2, rho_free, lam, all_stats
