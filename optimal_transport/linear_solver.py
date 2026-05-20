"""
Linear solver for the Schur complement system arising from KKT elimination.

Saddle-point system:
  [ A_hat   D^T ] [dw ]   = -[grad_w L]
  [ D       0   ] [dlam]     [grad_lam L]

Eliminate dw -> solve  S dlam = rhs  where S = D A_hat^{-1} D^T (SPD).
Preconditioner: symmetric Gauss-Seidel (SGS) on the sparse S matrix.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .grid import div_st, grad_st


# ---------------------------------------------------------------------------
# Matrix-free Schur matvec  S x = D A_hat^{-1} D^T x
# ---------------------------------------------------------------------------

def schur_matvec(x_flat, A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3):
    x = x_flat.reshape(n1, n2, n3)

    g_m1, g_m2, g_rho_free = grad_st(x, h)

    g_m1      /= A_hat_m1
    g_m2      /= A_hat_m2

    rho_full = np.zeros((n1, n2, n3 + 1))
    rho_full[:, :, 1:-1] = g_rho_free / A_hat_rho

    y = div_st(g_m1, g_m2, rho_full, h)
    return y.ravel()


# ---------------------------------------------------------------------------
# Sparse Schur matrix (for preconditioner only)
# ---------------------------------------------------------------------------

def _build_derivative_matrices(n1, n2, n3, h):
    """
    Build sparse derivative matrices D1, D2, D3_free.
    Each maps from face DOFs to cell-center DOFs.
    Returns (D1, D2, D3f) in CSR format.
    """
    N  = n1 * n2 * n3      # cell centers

    def cell_idx(i, j, k):
        return i * n2 * n3 + j * n3 + k

    # ----- D1: (n1+1)*n2*n3 x-face DOFs -> N cells -----
    N_m1 = (n1 + 1) * n2 * n3
    rows, cols, vals = [], [], []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                c = cell_idx(i, j, k)
                # face (i+1,j,k) gets +1/h, face (i,j,k) gets -1/h
                f_plus  = (i + 1) * n2 * n3 + j * n3 + k
                f_minus = i       * n2 * n3 + j * n3 + k
                rows += [c, c]; cols += [f_plus, f_minus]; vals += [1/h, -1/h]
    D1 = sp.csr_matrix((vals, (rows, cols)), shape=(N, N_m1))

    # ----- D2: n1*(n2+1)*n3 y-face DOFs -> N cells -----
    N_m2 = n1 * (n2 + 1) * n3
    rows, cols, vals = [], [], []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                c = cell_idx(i, j, k)
                f_plus  = i * (n2 + 1) * n3 + (j + 1) * n3 + k
                f_minus = i * (n2 + 1) * n3 + j       * n3 + k
                rows += [c, c]; cols += [f_plus, f_minus]; vals += [1/h, -1/h]
    D2 = sp.csr_matrix((vals, (rows, cols)), shape=(N, N_m2))

    # ----- D3_free: n1*n2*(n3-1) free t-face DOFs -> N cells -----
    # free t-faces are at time indices 1..n3-1 (interior)
    N_rho_free = n1 * n2 * (n3 - 1)

    def rho_free_idx(i, j, k):
        # k here is the FREE index (0-based), corresponding to t-face k+1
        return i * n2 * (n3 - 1) + j * (n3 - 1) + k

    rows, cols, vals = [], [], []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                c = cell_idx(i, j, k)
                # t-face at k+1 (free index k) contributes +1/h
                if k < n3 - 1:
                    f_plus = rho_free_idx(i, j, k)
                    rows.append(c); cols.append(f_plus); vals.append(1/h)
                # t-face at k (free index k-1) contributes -1/h
                if k > 0:
                    f_minus = rho_free_idx(i, j, k - 1)
                    rows.append(c); cols.append(f_minus); vals.append(-1/h)
    D3f = sp.csr_matrix((vals, (rows, cols)), shape=(N, N_rho_free))

    return D1, D2, D3f


def build_schur_sparse(A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3):
    """
    Build S = D A_hat^{-1} D^T as a sparse CSC matrix (for SGS preconditioner).
    """
    D1, D2, D3f = _build_derivative_matrices(n1, n2, n3, h)

    inv_m1  = sp.diags(1.0 / A_hat_m1.ravel())
    inv_m2  = sp.diags(1.0 / A_hat_m2.ravel())
    inv_rho = sp.diags(1.0 / A_hat_rho.ravel())

    S = (D1 @ inv_m1 @ D1.T
       + D2 @ inv_m2 @ D2.T
       + D3f @ inv_rho @ D3f.T)

    return S.tocsc()


# ---------------------------------------------------------------------------
# SGS preconditioner
# ---------------------------------------------------------------------------

def build_sgs_preconditioner(S_sparse):
    """Symmetric Gauss-Seidel preconditioner for SPD matrix S."""
    L = sp.tril(S_sparse, format='csc')
    U = sp.triu(S_sparse, format='csc')
    D_diag = S_sparse.diagonal()

    def sgs_solve(r):
        z = spla.spsolve_triangular(L, r,       lower=True)
        z = D_diag * z
        z = spla.spsolve_triangular(U, z,       lower=False)
        return z

    n = S_sparse.shape[0]
    return spla.LinearOperator((n, n), matvec=sgs_solve)


# ---------------------------------------------------------------------------
# Main solver: CG on  S dlam = -rhs
# ---------------------------------------------------------------------------

def solve_schur_system(rhs, A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3,
                       tol=1e-4, maxiter=500):
    """
    Solve S dlam = rhs using preconditioned CG.
    Returns (dlam as (n1,n2,n3) array, CG info code).
    """
    N = n1 * n2 * n3

    S_op = spla.LinearOperator(
        (N, N),
        matvec=lambda x: schur_matvec(x, A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3)
    )

    S_sparse = build_schur_sparse(A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3)
    M        = build_sgs_preconditioner(S_sparse)

    iters = [0]
    delta_lam_flat, _ = spla.cg(S_op, rhs.ravel(), rtol=tol, maxiter=maxiter, M=M,
                                 callback=lambda _: iters.__setitem__(0, iters[0] + 1))

    return delta_lam_flat.reshape(n1, n2, n3), iters[0]


# ---------------------------------------------------------------------------
# Recover dw after solving for dlam
# ---------------------------------------------------------------------------

def recover_dw(delta_lam, g_m1, g_m2, g_rho, A_hat_m1, A_hat_m2, A_hat_rho, h):
    """
    dw = -A_hat^{-1} (D^T dlam + grad_w L)
    Returns (dm1, dm2, drho_free).
    """
    D1T_dl, D2T_dl, D3T_dl = grad_st(delta_lam, h)

    dm1  = -(D1T_dl + g_m1) / A_hat_m1
    dm2  = -(D2T_dl + g_m2) / A_hat_m2
    drho = -(D3T_dl + g_rho) / A_hat_rho

    return dm1, dm2, drho
