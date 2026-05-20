"""
Constraint residual for the space-time continuity equation.

The constraint is  D·[m; rho] = 0  (enforced as rho includes fixed boundary slices).
"""

from .grid import div_st


def constraint_residual(m1, m2, rho_full, h):
    """
    Returns C = div_st(m1, m2, rho_full, h), shape (n1, n2, n3).
    Satisfied when C = 0 everywhere.

    rho_full : (n1, n2, n3+1), with boundary slices already set to mu0/mu1.
    """
    return div_st(m1, m2, rho_full, h)
