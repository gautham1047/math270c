"""
Outer SQP loop for time-dependent optimal transport (Haber & Horesh 2015, Sec. 4).
"""

import numpy as np
from .grid import div_st
from .objective import objective, grad_m, grad_rho, hessian_diag
from .linear_solver import solve_schur_system, recover_dw, solve_saddle_system_gmres
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
        tol=1e-4, max_iter=100, cg_tol=5e-5, cg_maxiter=500,
        max_backtracks=20, max_h_factor=1.3, hessian_eps=1e-8,
        method='cg_sgs', gmres_tol=0.025, gmres_maxiter=50,
        verbose=True):
    """
    SQP solver. Returns (m1, m2, rho_free, lam, stats).

    method : 'cg_sgs'   — CG on Schur complement with SGS preconditioner (default)
             'gmres_amg' — GMRES on full saddle-point system with AMG preconditioner

    stats : list of dicts with keys
              iter, inner_iters, f, h_viol, alpha, kkt_w, kkt_lam
    """
    n1, n2 = mu0.shape
    n3     = rho_free.shape[2] + 1

    rho_full = _make_rho_full(mu0, mu1, rho_free)
    h0       = np.sum(np.abs(div_st(m1, m2, rho_full, h)))

    filt = Filter(max_h_factor=max_h_factor)
    filt.initialize(h0)

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
        Ah_m1, Ah_m2, Ah_rho = hessian_diag(m1, m2, rho_full, h, p, eps=hessian_eps)

        # 4-6. Inner linear solve (dispatch on method)
        if method == 'cg_sgs':
            # Reduced Schur system: S dlam = glam - D Â^{-1} grad_w L
            inv_gm1 = gm1 / Ah_m1
            inv_gm2 = gm2 / Ah_m2
            inv_grho_full = np.zeros((n1, n2, n3 + 1))
            inv_grho_full[:, :, 1:-1] = grho / Ah_rho
            rhs = glam - div_st(inv_gm1, inv_gm2, inv_grho_full, h)
            delta_lam, inner_iters = solve_schur_system(
                rhs, Ah_m1, Ah_m2, Ah_rho, h, n1, n2, n3,
                tol=cg_tol, maxiter=cg_maxiter
            )
            dm1, dm2, drho = recover_dw(delta_lam, gm1, gm2, grho,
                                        Ah_m1, Ah_m2, Ah_rho, h)
        elif method == 'gmres_amg':
            # Full saddle-point system with Ruge-Stüben AMG preconditioner
            dm1, dm2, drho, delta_lam, inner_iters = solve_saddle_system_gmres(
                gm1, gm2, grho, glam,
                Ah_m1, Ah_m2, Ah_rho, h, n1, n2, n3,
                tol=gmres_tol, maxiter=gmres_maxiter,
            )
        else:
            raise ValueError(f"Unknown method {method!r}. Use 'cg_sgs' or 'gmres_amg'.")

        # 7. Line search with filter
        alpha    = 1.0
        accepted = False

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

        # Restoration phase: if line search failed completely, try a pure
        # h-reduction step (ignore f).  Guarantees progress whenever the Newton
        # direction has any feasibility-improving component.
        restored = False
        if not accepted:
            if verbose:
                print(f"  Warning: line search failed at iter {it} ({reject_reason}), "
                      f"attempting restoration.")
            alpha_r = 1.0
            for _ in range(max_backtracks):
                m1_r  = m1       + alpha_r * dm1
                m2_r  = m2       + alpha_r * dm2
                rho_r = rho_free + alpha_r * drho

                if rho_r.min() <= 0:
                    alpha_r *= 0.5
                    continue

                rho_full_r = _make_rho_full(mu0, mu1, rho_r)
                h_r = np.sum(np.abs(div_st(m1_r, m2_r, rho_full_r, h)))

                if h_r < filt.h_current:
                    f_r = objective(m1_r, m2_r, rho_full_r, h, p)
                    filt.add(f_r, h_r)
                    filt.accept(h_r)
                    m1, m2, rho_free = m1_r, m2_r, rho_r
                    lam = lam + alpha_r * delta_lam
                    alpha    = alpha_r
                    restored = True
                    if verbose:
                        print(f"  Restored at alpha={alpha_r:.3e}  h={h_r:.3e}")
                    break

                alpha_r *= 0.5

            if not accepted and not restored and verbose:
                print(f"  Warning: restoration also failed at iter {it}, step skipped.")

        stats.append({
            'iter':          it,
            'inner_iters':   inner_iters,
            'f':             objective(m1, m2, _make_rho_full(mu0, mu1, rho_free), h, p),
            'h_viol':        kkt_lam,
            'alpha':         alpha,
            'kkt_w':         kkt_w,
            'kkt_lam':       kkt_lam,
            'scale_lam':     scale_lam,
            'accepted':      accepted or restored,
            'reject_reason': ('restored' if restored else reject_reason) if not accepted else None,
        })

    return m1, m2, rho_free, lam, stats
