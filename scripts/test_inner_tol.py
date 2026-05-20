"""
Test the SQP solver on the smallest grid (16x16x10) across a range of CG
inner tolerances.  Run from the math270c directory:

    python scripts/test_inner_tol.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from optimal_transport.test_images import make_images
from optimal_transport.sqp import initialize, sqp


GRID   = (16, 16, 10)
P      = 2
SQP_TOL     = 1e-4
MAX_ITER    = 50
CG_MAXITER  = 750
CONTRAST    = 10

CG_TOLS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]


def run(cg_tol):
    n1, n2, n3 = GRID
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=CONTRAST)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

    m1, m2, rho_free, lam, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=P, tol=SQP_TOL, max_iter=MAX_ITER,
        cg_tol=cg_tol, cg_maxiter=CG_MAXITER,
        verbose=False,
    )

    n_iter      = len(stats)
    n_accepted  = sum(1 for s in stats if s['accepted'])
    n_rejected  = n_iter - n_accepted
    reasons     = [s['reject_reason'] for s in stats if not s['accepted']]
    filter_blocks = reasons.count('filter')
    strict_f      = reasons.count('strict_f')
    positivity    = reasons.count('positivity')

    last         = stats[-1]
    converged    = last['kkt_w'] < SQP_TOL and last['kkt_lam'] / max(stats[0]['kkt_lam'], 1e-10) < SQP_TOL

    # CG iterations: first, middle, last accepted step
    accepted_stats = [s for s in stats if s['accepted']]
    cg_first = accepted_stats[0]['inner_iters']  if accepted_stats else -1
    cg_mid   = accepted_stats[len(accepted_stats)//2]['inner_iters'] if accepted_stats else -1
    cg_last  = accepted_stats[-1]['inner_iters'] if accepted_stats else -1

    return dict(
        cg_tol       = cg_tol,
        sqp_iters    = n_iter,
        converged    = converged,
        n_accepted   = n_accepted,
        n_rejected   = n_rejected,
        filter_blocks= filter_blocks,
        strict_f     = strict_f,
        positivity   = positivity,
        kkt_lam_final= last['kkt_lam'],
        kkt_w_final  = last['kkt_w'],
        cg_iters     = (cg_first, cg_mid, cg_last),
    )


def main():
    print(f"Grid: {GRID}  contrast={CONTRAST}  p={P}  sqp_tol={SQP_TOL}  max_iter={MAX_ITER}\n")
    print(f"{'cg_tol':>8}  {'SQP':>5}  {'conv':>5}  {'acc':>5}  {'rej':>5}  "
          f"{'filt':>6}  {'strF':>5}  {'pos':>4}  "
          f"{'kkt_lam':>10}  {'kkt_w':>10}  {'CG(1,m,n)':>14}")
    print("-" * 100)

    for tol in CG_TOLS:
        r = run(tol)
        conv_str = "YES" if r['converged'] else "no"
        cg_str   = f"({r['cg_iters'][0]},{r['cg_iters'][1]},{r['cg_iters'][2]})"
        print(f"{r['cg_tol']:>8.3f}  {r['sqp_iters']:>5d}  {conv_str:>5}  "
              f"{r['n_accepted']:>5d}  {r['n_rejected']:>5d}  "
              f"{r['filter_blocks']:>6d}  {r['strict_f']:>5d}  {r['positivity']:>4d}  "
              f"{r['kkt_lam_final']:>10.3e}  {r['kkt_w_final']:>10.3e}  {cg_str:>14}")

    print("\nColumns: cg_tol | SQP iters | converged | accepted steps | rejected steps |")
    print("         filter-blocked | strict_f-blocked | positivity-blocked |")
    print("         final kkt_lam | final kkt_w | CG iters (first, mid, last accepted)")


if __name__ == "__main__":
    main()
