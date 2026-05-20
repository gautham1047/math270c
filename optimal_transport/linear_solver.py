"""
Linear solver for the saddle-point system (Haber & Horesh 2015, Sec. 4).

Two solver paths share the same staggered-grid operators:

CG+SGS path  (default):
  Eliminate dw -> solve  S dlam = rhs  where S = D A_hat^{-1} D^T (SPD).
  Preconditioner: symmetric Gauss-Seidel (SGS) on the sparse S matrix.

GMRES+AMG path:
  Solve the full saddle-point system K [dw; dlam] = -[grad_w L; glam]
  matrix-free, with a block-triangular AMG preconditioner.
  Preconditioner: one Ruge-Stüben AMG V-cycle on S (Schur complement).
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


# ---------------------------------------------------------------------------
# GMRES + AMG path  (Ruge-Stüben AMG on Schur complement)
# ---------------------------------------------------------------------------

def _compute_sizes(n1, n2, n3):
    N_m1  = (n1 + 1) * n2 * n3
    N_m2  = n1 * (n2 + 1) * n3
    N_rho = n1 * n2 * (n3 - 1)   # free interior faces only
    N_w   = N_m1 + N_m2 + N_rho
    N_lam = n1 * n2 * n3
    return N_m1, N_m2, N_rho, N_w, N_lam


def _pack_saddle(m1, m2, rho_free, lam):
    return np.concatenate([m1.ravel(), m2.ravel(), rho_free.ravel(), lam.ravel()])


def _unpack_saddle(x, n1, n2, n3):
    N_m1, N_m2, N_rho, N_w, N_lam = _compute_sizes(n1, n2, n3)
    x_m1  = x[:N_m1].reshape(n1 + 1, n2, n3)
    x_m2  = x[N_m1:N_m1 + N_m2].reshape(n1, n2 + 1, n3)
    x_rho = x[N_m1 + N_m2:N_w].reshape(n1, n2, n3 - 1)
    x_lam = x[N_w:N_w + N_lam].reshape(n1, n2, n3)
    return x_m1, x_m2, x_rho, x_lam


def _saddle_matvec(x, A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3):
    """Matrix-free K · x where K = [Â, D^T; D, 0]."""
    x_m1, x_m2, x_rho, x_lam = _unpack_saddle(x, n1, n2, n3)

    # Top block: Â·xw + D^T·xlam
    D1T_xl, D2T_xl, D3T_xl = grad_st(x_lam, h)
    y_m1  = A_hat_m1 * x_m1 + D1T_xl
    y_m2  = A_hat_m2 * x_m2 + D2T_xl
    y_rho = A_hat_rho * x_rho + D3T_xl   # grad_st returns (n1,n2,n3-1) for rho

    # Bottom block: D·xw  (embed free rho into full array, zero BCs)
    x_rho_full = np.zeros((n1, n2, n3 + 1))
    x_rho_full[:, :, 1:-1] = x_rho
    y_lam = div_st(x_m1, x_m2, x_rho_full, h)

    return _pack_saddle(y_m1, y_m2, y_rho, y_lam)


def _precond_apply(r, A_hat_m1, A_hat_m2, A_hat_rho, amg_solver, h, n1, n2, n3):
    """
    Apply block-triangular preconditioner P^{-1} where P = [Â, D^T; 0, S].

    Back-substitution:
      Step 1: S z_lam = r_lam  (one AMG V-cycle on S = D Â^{-1} D^T)
      Step 2: z_w = Â^{-1}(r_w - D^T z_lam)
    """
    r_m1, r_m2, r_rho, r_lam = _unpack_saddle(r, n1, n2, n3)

    z_lam = amg_solver.solve(r_lam.ravel(), tol=1e-10, maxiter=1)
    z_lam = z_lam.reshape(n1, n2, n3)

    D1T_zl, D2T_zl, D3T_zl = grad_st(z_lam, h)
    z_m1  = (r_m1  - D1T_zl) / A_hat_m1
    z_m2  = (r_m2  - D2T_zl) / A_hat_m2
    z_rho = (r_rho - D3T_zl) / A_hat_rho

    return _pack_saddle(z_m1, z_m2, z_rho, z_lam)


def solve_saddle_system_gmres(rhs_m1, rhs_m2, rhs_rho, rhs_lam,
                               A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3,
                               tol=0.1, maxiter=50):
    """
    Solve K [dw; dlam] = -[grad_w L; glam] with GMRES + Ruge-Stüben AMG.

    Two-phase approach (plan §4.4):
      Phase 1: GMRES to tol (default 0.1, relative preconditioned residual).
      Phase 2: Filter backtracking on the step (handled in sqp.py).

    Requires: pip install pyamg

    Returns (dm1, dm2, drho_free, dlam, gmres_iter_count).
    """
    try:
        import pyamg
    except ImportError as exc:
        raise ImportError(
            "PyAMG is required for method='gmres_amg': pip install pyamg"
        ) from exc

    N_m1, N_m2, N_rho, N_w, N_lam = _compute_sizes(n1, n2, n3)
    N = N_w + N_lam

    b = -_pack_saddle(rhs_m1, rhs_m2, rhs_rho, rhs_lam)

    # Build AMG hierarchy once per SQP iteration (S is rebuilt from current Â).
    # PyAMG's RS solver needs CSR; build_schur_sparse returns CSC, so convert.
    S_sparse = build_schur_sparse(A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3).tocsr()
    amg_solver = pyamg.ruge_stuben_solver(S_sparse)

    K_op = spla.LinearOperator(
        (N, N), dtype=float,
        matvec=lambda x: _saddle_matvec(x, A_hat_m1, A_hat_m2, A_hat_rho, h, n1, n2, n3),
    )
    P_op = spla.LinearOperator(
        (N, N), dtype=float,
        matvec=lambda r: _precond_apply(r, A_hat_m1, A_hat_m2, A_hat_rho,
                                        amg_solver, h, n1, n2, n3),
    )

    iters = [0]
    x_flat, _ = spla.gmres(
        K_op, b, M=P_op,
        rtol=tol, maxiter=maxiter,
        restart=min(50, maxiter),
        callback=lambda _: iters.__setitem__(0, iters[0] + 1),
    )

    dm1, dm2, drho, dlam = _unpack_saddle(x_flat, n1, n2, n3)
    return dm1, dm2, drho, dlam, iters[0]
