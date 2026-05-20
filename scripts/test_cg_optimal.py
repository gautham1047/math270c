"""
Golden section search for the CG tolerance that minimises wall-clock time.

We search in log-space because cg_tol spans several orders of magnitude.
Wall-clock time is unimodal: too loose → many SQP iters; too tight → each
CG solve becomes expensive.  The optimum is somewhere in between.

Starting bounds: [LO, HI] = [1e-5, 0.01] (the 16x16x10 sweep showed
time still falling at 0.001, so we extend the search below that).

Run from the math270c directory:
    python scripts/test_cg_optimal.py                # 16x16x10 (fast, ~2 min)
    python scripts/test_cg_optimal.py --grid 32      # 32x32x20 (slower, ~10-20 min)
    python scripts/test_cg_optimal.py --grid 64      # 64x64x40 (slow)
    python scripts/test_cg_optimal.py --reps 3       # average over 3 timing reps to reduce noise
"""

import sys, os, time, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from optimal_transport.test_images import make_images
from optimal_transport.sqp import initialize, sqp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_GRIDS = {16: (16, 16, 10), 32: (32, 32, 20), 64: (64, 64, 40)}
_CG_MAXITER = {16: 300, 32: 750, 64: 2000}

P        = 2
SQP_TOL  = 1e-4
MAX_ITER = 100
CONTRAST = 10

# Search bounds in log10 space
LOG_LO = math.log10(1e-5)   # tightest (1e-5)
LOG_HI = math.log10(1e-2)   # loosest  (0.01)

GSS_TOL = 0.05    # stop when log10-interval width < this (~12% ratio)
PHI     = (math.sqrt(5) - 1) / 2   # golden ratio ≈ 0.618

# ---------------------------------------------------------------------------
# Single timed run
# ---------------------------------------------------------------------------

def run_one(grid, cg_tol, cg_maxiter, reps=1):
    """Return (mean wall-clock time, sqp_iters, total_cg, converged)."""
    n1, n2, n3 = grid
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=CONTRAST)

    times = []
    for _ in range(reps):
        m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
        t0 = time.perf_counter()
        _, _, _, _, stats = sqp(
            m1, m2, rho_free, lam, mu0, mu1, h,
            p=P, tol=SQP_TOL, max_iter=MAX_ITER,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            verbose=False,
        )
        times.append(time.perf_counter() - t0)

    elapsed   = min(times)           # min over reps reduces scheduling noise
    sqp_iters = len(stats)
    total_cg  = sum(s['inner_iters'] for s in stats)
    converged = sqp_iters < MAX_ITER
    return elapsed, sqp_iters, total_cg, converged


# ---------------------------------------------------------------------------
# Golden section search
# ---------------------------------------------------------------------------

def golden_section_search(grid, cg_maxiter, reps, lo=LOG_LO, hi=LOG_HI):
    """
    Minimise wall-clock time over cg_tol in [10^lo, 10^hi] (log scale).
    Returns (optimal_cg_tol, optimal_time).
    """
    label = f"{grid[0]}x{grid[1]}x{grid[2]}"
    print(f"\n=== Golden section search - {label} "
          f"(reps={reps}, cg_maxiter={cg_maxiter}) ===")
    print(f"  Search range: cg_tol in [1e{lo:.1f}, 1e{hi:.1f}]  (log10)")
    print(f"  Stop when interval < {GSS_TOL} log10-units\n")
    print(f"  {'step':>4}  {'lo':>8}  {'hi':>8}  {'c':>10}  {'d':>10}"
          f"  {'t(c)':>8}  {'t(d)':>8}  {'better':>6}")
    print("  " + "-" * 72)

    cache = {}

    def evaluate(log_tol):
        if log_tol not in cache:
            tol = 10 ** log_tol
            t, sqp, cg, conv = run_one(grid, tol, cg_maxiter, reps)
            conv_str = "YES" if conv else "no"
            cache[log_tol] = (t, sqp, cg, conv_str)
        return cache[log_tol][0]

    c = hi - PHI * (hi - lo)
    d = lo + PHI * (hi - lo)
    step = 0

    while (hi - lo) > GSS_TOL:
        step += 1
        tc = evaluate(c)
        td = evaluate(d)
        better = "c" if tc < td else "d"
        print(f"  {step:>4}  {lo:>8.3f}  {hi:>8.3f}  "
              f"{c:>10.4f}  {d:>10.4f}  "
              f"{tc:>8.3f}s  {td:>8.3f}s  {better:>6}")

        if tc < td:
            hi = d
            d, td = c, tc
            c = hi - PHI * (hi - lo)
        else:
            lo = c
            c, tc = d, td
            d = lo + PHI * (hi - lo)

    opt_log = (lo + hi) / 2
    opt_tol = 10 ** opt_log
    opt_t   = evaluate(opt_log)

    print(f"\n  Converged. Optimal interval: [1e{lo:.3f}, 1e{hi:.3f}]")
    print(f"  Optimal cg_tol ~= {opt_tol:.2e}  (time ~= {opt_t:.3f}s)")

    # Print all evaluated points sorted by cg_tol
    print(f"\n  All evaluated points:")
    print(f"  {'cg_tol':>10}  {'time(s)':>8}  {'SQP':>5}  {'CG_tot':>8}  {'conv':>5}")
    print("  " + "-" * 45)
    for log_tol in sorted(cache):
        t, sqp, cg, conv = cache[log_tol]
        print(f"  {10**log_tol:>10.2e}  {t:>8.3f}  {sqp:>5d}  {cg:>8d}  {conv:>5}")

    return opt_tol, opt_t


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, choices=[16, 32, 64], default=16)
    parser.add_argument("--contrast", type=int, default=10)
    parser.add_argument("--reps", type=int, default=2,
                        help="Timing repetitions per point (min is kept)")
    args = parser.parse_args()

    global CONTRAST
    CONTRAST = args.contrast

    grid       = _GRIDS[args.grid]
    cg_maxiter = _CG_MAXITER[args.grid]

    golden_section_search(grid, cg_maxiter, args.reps)


if __name__ == "__main__":
    main()
