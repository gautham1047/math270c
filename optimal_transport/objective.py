"""
Objective function, gradients, and Gauss-Newton Hessian diagonal.

All functions take rho as the FULL array (n1, n2, n3+1) including fixed
boundary slices rho[:,:,0]=mu0 and rho[:,:,n3]=mu1.
Gradients w.r.t. rho are returned only for the FREE indices 1..n3-1,
shape (n1, n2, n3-1).
"""

import numpy as np
from .grid import avg_x, avg_y, avg_t, avg_x_adj, avg_y_adj, avg_t_adj, grad_st


def objective(m1, m2, rho, h, p=2, eps=1e-8):
    """
    f(m, rho) = h^3 * sum [ As(|m|_eps^p) * At(1/rho^{p-1}) ]

    First square/power at face location, then average (stability rule).
    First invert/power at rho location, then average.
    """
    m1_norm_p = (m1**2 + eps**2) ** (p / 2)
    m2_norm_p = (m2**2 + eps**2) ** (p / 2)

    ms_avg = avg_x(m1_norm_p) + avg_y(m2_norm_p)   # (n1, n2, n3)

    inv_rho     = 1.0 / rho ** (p - 1)              # (n1, n2, n3+1)
    inv_rho_avg = avg_t(inv_rho)                     # (n1, n2, n3)

    return h**3 * np.sum(ms_avg * inv_rho_avg)


def grad_m(m1, m2, rho, lam, h, p=2, eps=1e-8):
    """
    Gradient of L w.r.t. m1 and m2 (face fields, same shapes as m1/m2).
    """
    inv_rho_avg = avg_t(1.0 / rho ** (p - 1))       # (n1, n2, n3)

    m1_factor = p * (m1**2 + eps**2) ** ((p - 2) / 2) * m1
    g_m1 = h**3 * m1_factor * avg_x_adj(inv_rho_avg)

    m2_factor = p * (m2**2 + eps**2) ** ((p - 2) / 2) * m2
    g_m2 = h**3 * m2_factor * avg_y_adj(inv_rho_avg)

    D1T_lam, D2T_lam, _ = grad_st(lam, h)
    g_m1 += D1T_lam
    g_m2 += D2T_lam

    return g_m1, g_m2


def grad_rho(m1, m2, rho, lam, h, p=2, eps=1e-8):
    """
    Gradient of L w.r.t. FREE rho unknowns, shape (n1, n2, n3-1).
    """
    ms_avg       = avg_x((m1**2 + eps**2) ** (p / 2)) + avg_y((m2**2 + eps**2) ** (p / 2))
    ms_at_tfaces = avg_t_adj(ms_avg)                  # (n1, n2, n3+1)

    rho_factor = -(p - 1) / rho[:, :, 1:-1] ** p     # (n1, n2, n3-1)
    g_rho = h**3 * rho_factor * ms_at_tfaces[:, :, 1:-1]

    _, _, D3T_lam = grad_st(lam, h)
    g_rho += D3T_lam

    return g_rho


def hessian_diag(m1, m2, rho, h, p=2, eps=1e-8):
    """
    Gauss-Newton Hessian diagonal blocks (all strictly positive).

    Returns:
      A_hat_m1   : (n1+1, n2,   n3  )
      A_hat_m2   : (n1,   n2+1, n3  )
      A_hat_rho  : (n1,   n2,   n3-1)  (free rho only)
    """
    inv_rho_avg = avg_t(1.0 / rho ** (p - 1))        # (n1, n2, n3)

    if p == 2:
        A_hat_m1 = h**3 * 2.0 * avg_x_adj(inv_rho_avg)
        A_hat_m2 = h**3 * 2.0 * avg_y_adj(inv_rho_avg)
    else:
        m1_curv  = p * (p - 1) * (m1**2 + eps**2) ** ((p - 2) / 2)
        m2_curv  = p * (p - 1) * (m2**2 + eps**2) ** ((p - 2) / 2)
        A_hat_m1 = h**3 * m1_curv * avg_x_adj(inv_rho_avg)
        A_hat_m2 = h**3 * m2_curv * avg_y_adj(inv_rho_avg)

    ms_avg       = avg_x(m1**2) + avg_y(m2**2)
    ms_at_tfaces = avg_t_adj(ms_avg)                  # (n1, n2, n3+1)

    if p == 2:
        A_hat_rho = h**3 * 2.0 * ms_at_tfaces[:, :, 1:-1] / rho[:, :, 1:-1] ** 3
    else:
        A_hat_rho = h**3 * p * (p - 1) * ms_at_tfaces[:, :, 1:-1] / rho[:, :, 1:-1] ** (p + 1)

    # Floor m blocks at a small absolute value; floor rho block at h^3 so that
    # the rho/m Schur contributions stay comparable even when |m|≈0 and A_hat_rho
    # would otherwise collapse to near-zero, causing CG to produce directions that
    # increase the constraint violation.
    floor_m   = 1e-12
    floor_rho = h**3
    A_hat_m1  = np.maximum(A_hat_m1,  floor_m)
    A_hat_m2  = np.maximum(A_hat_m2,  floor_m)
    A_hat_rho = np.maximum(A_hat_rho, floor_rho)

    return A_hat_m1, A_hat_m2, A_hat_rho
