"""
Staggered grid operators for space-time optimal transport.

Grid convention (n1 x n2 spatial cells, n3 time cells, uniform h):
  m1   : x-faces  (n1+1, n2,   n3  )
  m2   : y-faces  (n1,   n2+1, n3  )
  rho  : t-faces  (n1,   n2,   n3+1)   rho[:,:,0]=mu0, rho[:,:,n3]=mu1 fixed
  lam  : centers  (n1,   n2,   n3  )
"""

import numpy as np


# ---------------------------------------------------------------------------
# Divergence and gradient (adjoint pair)
# ---------------------------------------------------------------------------

def div_st(m1, m2, rho, h):
    """Space-time divergence. Returns (n1, n2, n3) array."""
    dm1  = (m1[1:, :, :]  - m1[:-1, :, :])  / h
    dm2  = (m2[:, 1:, :]  - m2[:, :-1, :])  / h
    drho = (rho[:, :, 1:] - rho[:, :, :-1]) / h
    return dm1 + dm2 + drho


def grad_st(lam, h):
    """
    Adjoint of div_st. Maps (n1,n2,n3) cell-center field to face fields.
    Returns (g_m1, g_m2, g_rho_free) with shapes
      g_m1       : (n1+1, n2,   n3  )
      g_m2       : (n1,   n2+1, n3  )
      g_rho_free : (n1,   n2,   n3-1)   (free rho indices 1..n3-1)
    """
    n1, n2, n3 = lam.shape

    # D1^T: x-gradient onto x-faces (no-flux BC: spatial boundary faces fixed to zero)
    g_m1 = np.zeros((n1 + 1, n2, n3))
    g_m1[1:-1, :, :] = (lam[:-1, :, :] - lam[1:, :, :]) / h

    # D2^T: y-gradient onto y-faces (no-flux BC: spatial boundary faces fixed to zero)
    g_m2 = np.zeros((n1, n2 + 1, n3))
    g_m2[:, 1:-1, :] = (lam[:, :-1, :] - lam[:, 1:, :]) / h

    # D3^T: t-gradient onto free t-faces (indices 1..n3-1)
    # (D3^T lam)[k] = (lam[:,:,k-1] - lam[:,:,k]) / h  for k=1..n3-1
    g_rho_free = (lam[:, :, :-1] - lam[:, :, 1:]) / h   # (n1, n2, n3-1)

    return g_m1, g_m2, g_rho_free


# ---------------------------------------------------------------------------
# Averaging operators and their adjoints
# ---------------------------------------------------------------------------

def avg_x(f):
    """(n1+1, n2, n3) -> (n1, n2, n3): average in x."""
    return 0.5 * (f[1:, :, :] + f[:-1, :, :])


def avg_y(f):
    """(n1, n2+1, n3) -> (n1, n2, n3): average in y."""
    return 0.5 * (f[:, 1:, :] + f[:, :-1, :])


def avg_t(f):
    """(n1, n2, n3+1) -> (n1, n2, n3): average in t."""
    return 0.5 * (f[:, :, 1:] + f[:, :, :-1])


def avg_x_adj(f):
    """Adjoint of avg_x: (n1, n2, n3) -> (n1+1, n2, n3)."""
    out = np.zeros((f.shape[0] + 1, f.shape[1], f.shape[2]))
    out[:-1, :, :] += 0.5 * f
    out[1:,  :, :] += 0.5 * f
    return out


def avg_y_adj(f):
    """Adjoint of avg_y: (n1, n2, n3) -> (n1, n2+1, n3)."""
    out = np.zeros((f.shape[0], f.shape[1] + 1, f.shape[2]))
    out[:, :-1, :] += 0.5 * f
    out[:, 1:,  :] += 0.5 * f
    return out


def avg_t_adj(f):
    """Adjoint of avg_t: (n1, n2, n3) -> (n1, n2, n3+1)."""
    out = np.zeros((f.shape[0], f.shape[1], f.shape[2] + 1))
    out[:, :, :-1] += 0.5 * f
    out[:, :, 1:]  += 0.5 * f
    return out
