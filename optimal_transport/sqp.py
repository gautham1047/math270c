"""
Outer SQP loop for time-dependent optimal transport (Haber & Horesh 2015, Sec. 4).
"""

import numpy as np
from .grid import div_st
from .objective import objective, grad_m, grad_rho, hessian_diag
from .linear_solver import solve_schur_system, recover_dw
from .filter import Filter


def initialize(mu0, mu1, n3, h):
    """
    Default initialization: m=0, rho = linear interp, lam=0.
    Returns (m1, m2, rho_free, lam).
    """
    n1, n2 = mu0.shape

    m1 = np.zeros((n1 + 1, n2, n3))
    m2 = np.zeros((n1, n2 + 1, n3))

    rho_full = np.zeros((n1, n2, n3 + 1))
    for k in range(n3 + 1):
        t = k / n3
        rho_full[:, :, k] = (1.0 - t) * mu0 + t * mu1
    rho_free = rho_full[:, :, 1:-1].copy()

    lam = np.zeros((n1, n2, n3))

    return m1, m2, rho_free, lam


def _make_rho_full(mu0, mu1, rho_free):
    n1, n2 = mu0.shape
    n3 = rho_free.shape[2] + 1
    rho_full = np.empty((n1, n2, n3 + 1))
    rho_full[:, :, 0]    = mu0
    rho_full[:, :, 1:-1] = rho_free
    rho_full[:, :, -1]   = mu1
    return rho_full


def sqp(m1, m2, rho_free, lam, mu0, mu1, h, p=2,
        tol=1e-4, max_iter=100, cg_tol=0.01, cg_maxiter=200,
        max_backtracks=20, max_h_factor=2.0, verbose=True):
    """
    SQP solver. Returns (m1, m2, rho_free, lam, stats).

    stats : list of dicts with keys
              iter, cg_iters, f, h_viol, alpha, kkt_w, kkt_lam
    """
    n1, n2 = mu0.shape
    n3     = rho_free.shape[2] + 1

    rho_full = _make_rho_full(mu0, mu1, rho_free)
    f0       = objective(m1, m2, rho_full, h, p)
    h0       = np.sum(np.abs(div_st(m1, m2, rho_full, h)))

    filt = Filter(max_h_factor=max_h_factor)
    filt.initialize(f0, h0)

    # Reference scales for convergence check
    scale_lam = max(h0, 1e-10)

    stats = []

    for it in range(1, max_iter + 1):
        rho_full = _make_rho_full(mu0, mu1, rho_free)

        # 1. Gradients
        gm1, gm2 = grad_m(m1, m2, rho_full, lam, h, p)
        grho     = grad_rho(m1, m2, rho_full, lam, h, p)
        glam     = div_st(m1, m2, rho_full, h)

        kkt_w   = np.linalg.norm(gm1) + np.linalg.norm(gm2) + np.linalg.norm(grho)
        kkt_lam = np.sum(np.abs(glam))

        f_cur = objective(m1, m2, rho_full, h, p)
        if verbose:
            print(f"  SQP {it:3d}: f={f_cur:.6e}  |C|={kkt_lam:.3e}  |grad_w|={kkt_w:.3e}"
                  f"  filt={len(filt.entries)}")

        # 2. Convergence check (primal feasibility only — kkt_w is un-normalised)
        if kkt_lam / scale_lam < tol:
            if verbose:
                print(f"  Converged at iter {it}.")
            break

        # 3. Hessian diagonal
        Ah_m1, Ah_m2, Ah_rho = hessian_diag(m1, m2, rho_full, h, p)

        # 4. Schur complement RHS:  glam - D A_hat^{-1} grad_w L
        inv_gm1 = gm1 / Ah_m1
        inv_gm2 = gm2 / Ah_m2
        inv_grho_full = np.zeros((n1, n2, n3 + 1))
        inv_grho_full[:, :, 1:-1] = grho / Ah_rho

        rhs = glam - div_st(inv_gm1, inv_gm2, inv_grho_full, h)

        # 5. Solve for dlam
        delta_lam, cg_info = solve_schur_system(
            rhs, Ah_m1, Ah_m2, Ah_rho, h, n1, n2, n3,
            tol=cg_tol, maxiter=cg_maxiter
        )

        # 6. Recover dw
        dm1, dm2, drho = recover_dw(delta_lam, gm1, gm2, grho,
                                    Ah_m1, Ah_m2, Ah_rho, h)

        # 7. Line search with filter
        alpha     = 1.0
        accepted  = False
        cg_iters  = cg_info  # scipy returns iteration count or 0 on success

        reject_reason = None
        for _ in range(max_backtracks):
            m1_t  = m1       + alpha * dm1
            m2_t  = m2       + alpha * dm2
            rho_t = rho_free + alpha * drho

            if rho_t.min() <= 0:
                reject_reason = 'positivity'
                alpha *= 0.5
                continue

            rho_full_t = _make_rho_full(mu0, mu1, rho_t)
            f_trial    = objective(m1_t, m2_t, rho_full_t, h, p)
            h_trial    = np.sum(np.abs(div_st(m1_t, m2_t, rho_full_t, h)))

            # Check h-increase cap first to get a finer rejection reason.
            if (filt.h_current is not None
                    and h_trial > filt.max_h_factor * filt.h_current):
                reject_reason = 'h_increase'
                alpha *= 0.5
                continue

            if filt.is_acceptable(f_trial, h_trial):
                # Only add to filter when f genuinely increases (not just floating-
                # point noise). Prevents staircase accumulation of near-duplicate
                # entries that block subsequent steps.
                if f_trial > f_cur * (1.0 + 1e-4):
                    filt.add(f_trial, h_trial)
                filt.accept(h_trial)
                m1, m2, rho_free = m1_t, m2_t, rho_t
                lam = lam + alpha * delta_lam
                accepted = True
                break
            else:
                reject_reason = 'filter'
                alpha *= 0.5

        if not accepted and verbose:
            print(f"  Warning: line search failed at iter {it} ({reject_reason}), step rejected.")

        stats.append({
            'iter':          it,
            'cg_iters':      cg_iters,
            'f':             objective(m1, m2, _make_rho_full(mu0, mu1, rho_free), h, p),
            'h_viol':        kkt_lam,
            'alpha':         alpha,
            'kkt_w':         kkt_w,
            'kkt_lam':       kkt_lam,
            'accepted':      accepted,
            'reject_reason': reject_reason if not accepted else None,
        })

    return m1, m2, rho_free, lam, stats
