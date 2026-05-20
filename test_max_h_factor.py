"""
Sweep max_h_factor on 16x16x10 to find how tight the per-step h-increase
cap should be.  Run from the math270c directory:

    python test_max_h_factor.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from optimal_transport.test_images import make_images
from optimal_transport.sqp import initialize, sqp

GRID        = (16, 16, 10)
P           = 2
SQP_TOL     = 1e-4
MAX_ITER    = 100
CG_TOL      = 0.01
CG_MAXITER  = 200
CONTRAST    = 10

H_FACTORS  = [1.0, 1.5, 2.0, 3.0, 5.0, float('inf')]
BSEARCH_LO = 1.0
BSEARCH_HI = 3.0
BSEARCH_EPS = 0.02   # stop when interval width < this


def run(max_h_factor):
    n1, n2, n3 = GRID
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=CONTRAST)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

    m1, m2, rho_free, lam, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=P, tol=SQP_TOL, max_iter=MAX_ITER,
        cg_tol=CG_TOL, cg_maxiter=CG_MAXITER,
        max_h_factor=max_h_factor,
        verbose=False,
    )

    n_iter     = len(stats)
    n_accepted = sum(1 for s in stats if s['accepted'])
    reasons    = [s['reject_reason'] for s in stats if not s['accepted']]

    last = stats[-1]
    kkt_lam_final = last['kkt_lam']
    # sqp() breaks before appending to stats on convergence, so n_iter < MAX_ITER
    # means the loop exited via the convergence criterion, not the iteration limit.
    converged = n_iter < MAX_ITER

    return dict(
        max_h_factor = max_h_factor,
        sqp_iters    = n_iter,
        converged    = converged,
        n_accepted   = n_accepted,
        n_rejected   = n_iter - n_accepted,
        rej_h_inc    = reasons.count('h_increase'),
        rej_filter   = reasons.count('filter'),
        rej_pos      = reasons.count('positivity'),
        kkt_lam_final= kkt_lam_final,
    )


def main():
    print(f"Grid: {GRID}  contrast={CONTRAST}  p={P}  cg_tol={CG_TOL}  sqp_tol={SQP_TOL}\n")
    print(f"{'h_factor':>10}  {'SQP':>5}  {'conv':>5}  {'acc':>5}  {'rej':>5}"
          f"  {'h_inc':>6}  {'filt':>5}  {'pos':>4}  {'kkt_lam':>10}")
    print("-" * 80)

    for hf in H_FACTORS:
        r = run(hf)
        hf_str   = f"{hf:.1f}" if hf != float('inf') else "inf"
        conv_str = "YES" if r['converged'] else "no"
        print(f"{hf_str:>10}  {r['sqp_iters']:>5d}  {conv_str:>5}  "
              f"{r['n_accepted']:>5d}  {r['n_rejected']:>5d}  "
              f"{r['rej_h_inc']:>6d}  {r['rej_filter']:>5d}  {r['rej_pos']:>4d}  "
              f"{r['kkt_lam_final']:>10.3e}")

    print("\nColumns: h_factor | SQP iters | converged | accepted | rejected |")
    print("         h_increase blocks | filter blocks | positivity blocks | final kkt_lam")


if __name__ == "__main__":
    main()
