# SQP Solver Handoff — math270c

## What this project is

Implementing the matrix-free time-dependent optimal transport solver from
**Haber & Horesh (2015)**, Section 5.  The space-time OT problem is:

    min  h^3 * sum[ A_s(|m|^p) * A_t(1/rho^{p-1}) ]
    s.t. div(m1, m2, rho) = 0   (continuity equation on space-time grid)
         rho(:,:,0) = mu0,  rho(:,:,end) = mu1  (boundary conditions)

Variables: momentum fields m1 (x-faces), m2 (y-faces), density rho (t-faces),
multipliers lam (cell centres).  Solved with an outer SQP loop and an inner
preconditioned Krylov solve for the Schur complement system.

## File map

```
optimal_transport/
  grid.py          staggered-grid div/grad operators and averaging ops
  objective.py     f(m,rho), grad_m, grad_rho, hessian_diag
  sqp.py           outer SQP loop: initialize(), sqp()
  linear_solver.py Schur complement matvec, sparse SGS preconditioner, CG solve
  filter.py        Fletcher-Leyffer filter for the line search
  multilevel.py    multilevel warm-start
  test_images.py   make_images() — Gaussian blobs, contrast=10 or 100
  experiments.py   experiment1..5 from the paper

test_inner_tol.py  sweeps cg_tol on 16x16x10 grid, reports
                   SQP iters, accepted/rejected counts, rejection reasons
HANDOFF.md         this file
```

## Current state of the solver (as of May 2026)

All five key bugs have been fixed. The solver runs without filter blocking or
positivity failures on all tested grids.

```
Grid         SQP  acc  rej   kkt_rel    f_final
(16,16,10)    23   23    0   <1e-4     0.3792   (cg_tol=0.01)
(32,32,20)    --   --    --  --        --       (needs more CG iters, see below)
(64,64,40)    --   --    --  --        --
```

Multilevel (experiment3, contrast=100, levels 16→32→64):
- Level 1 (16×16×8):  1 SQP iter (near-immediate convergence at cg_tol=1e-4)
- Level 2 (32×32×16): a few iters
- Level 3 (64×64×32): fine-grid polish

## Bug fixes applied (all resolved)

### 1. `grad_st` sign reversal  (grid.py)
`grad_st` had the subtraction reversed on all interior faces, making every
Newton direction wrong.  Fixed to `(lam[j-1] - lam[j]) / h`.

### 2. Filter seeded with (f0=0, h0)  (filter.py)
At m=0, f0=0.  The seeded entry could never be dominated, permanently blocking
all steps on coarser grids.  Fixed: initialize filter empty; only set
`h_max = beta * h_0`.

### 3. `A_hat_rho` floor too low  (objective.py)
At m≈0, `A_hat_rho` collapses to ~1e-9, making the Schur complement 10^4×
dominated by the rho block.  CG then gives directions that *increase* h.
Fixed: floor at `h^3` (same scale as m block).

### 4. `strict_f` near-feasibility heuristic  (sqp.py)
A previous session added a block requiring `f_trial < f_cur` when
`kkt_lam/scale_lam < 10*tol`.  This caused 29–33 step rejections per run.
Removed entirely.

### 5. Convergence criterion scaling  (sqp.py)
`kkt_w` is an un-normalised L2 norm that scales as `sqrt(n) * h^3` and never
reaches `tol=1e-4` at mesh scale.  Changed to primal-only:
`kkt_lam / scale_lam < tol`.

### 6. Filter staircase accumulation  (sqp.py)
Near-duplicate filter entries `(f+eps, h)` accumulated and blocked CG steps
near the optimum.  Fixed: only add to filter when `f_trial > f_cur * (1+1e-4)`.

---

## Open problem 1: CG tolerance tuning

**Goal:** minimize wall-clock time for a given grid and convergence target.

### What we know

Timing benchmark on 16×16×10 (16 outer, cg_maxiter=200, all grids converge):

```
cg_tol   SQP iters   wall-clock    notes
0.1         75         2.58 s       paper's recommended value
0.05        51         1.99 s
0.02        38         1.77 s
0.01        23         1.23 s       current sqp.py default
0.001        6         0.29 s
1e-4         1         0.05 s       premature exit (see below)
```

Tighter CG is faster in total on this grid because each near-exact Newton step
reduces the outer iteration count enough to more than offset the extra inner work.
`cg_tol=0.001` is ~4× faster than `0.01` with similar convergence quality.

**Spurious convergence at cg_tol=1e-4:** After 1 SQP iter the solver exits
because `kkt_lam < scale_lam * tol`, but the pre-step kkt_lam recorded in
stats equals scale_lam, so the display shows `kkt_rel = 1.0`.  The actual
convergence appears genuine (one near-exact Newton step gets h very small on
this grid), but it should be verified on larger grids.

### What to try next

1. **Run the timing sweep on 32×32×20.**  The 32×32×20 grid needs many more CG
   iterations per step than 16×16×10 (the SGS preconditioner quality degrades at
   larger n).  It is possible the crossover point (tight vs loose CG) shifts —
   check whether `cg_tol=0.1` or `0.01` wins on the medium grid.

2. **Adaptive CG tolerance (inexact Newton scheduling).**  Start with loose
   `cg_tol=0.1` (fast cheap steps that reduce h quickly) and tighten as
   `kkt_lam` decreases, e.g.:

       cg_tol_k = min(0.1, max(1e-4, (kkt_lam_k / h0) ** 0.5))

   This mimics the standard inexact-Newton forcing sequence and avoids wasting
   CG work during the early "feasibility-seeking" phase.

3. **Jacobi preconditioner for the Schur system.**  The Schur diagonal can be
   extracted from `build_schur_sparse` via `.diagonal()`.  It is SPD (unlike
   SGS), so CG's convergence guarantee holds.  Might reduce per-step CG count
   considerably on the medium/fine grids.

---

## Open problem 2: Filter restoration phase

**Goal:** Prevent constraint violation from growing between accepted steps while
still allowing the algorithm to make objective progress.

### Current filter state

The filter now includes a per-step h-increase cap:

```python
# filter.py — new guard in is_acceptable()
if h_trial > self.max_h_factor * self.h_current:
    return False          # h_max_factor default = 2.0
```

`filt.accept(h_trial)` is called in sqp.py on every accepted step so
`h_current` tracks the most recently accepted constraint violation.

### What the restoration phase is

In standard SQP / interior-point methods, when a step is rejected by the filter
on f-grounds (not h-grounds), a **restoration phase** is triggered: temporarily
ignore the objective and solve a sub-problem that purely reduces h.  A simple
version is:

    Find alpha s.t.  div(m + alpha*dm, rho + alpha*drho) is minimised

which reduces to a line search on h alone (ignoring f).  The restored point is
then added to the filter, and the main SQP iteration resumes from the restored
iterate.

Without restoration, a filter rejection at max_backtracks causes the step to be
silently skipped, which can stall progress entirely.

### What to try

1. **Tune `max_h_factor`.**  Default is 2.0.  Try 1.5 (tighter) and 3.0
   (looser) on the 16×16×10 grid.  If the solver rejects many steps with
   `reason='h_increase'`, the factor is too tight and is acting like a monotone
   constraint on h — not desirable.

2. **Simple restoration step.**  When `max_backtracks` is exhausted without
   acceptance, instead of skipping the step, perform a pure h-reduction line
   search:

       for beta in [1, 0.5, 0.25, ...]:
           h_test = h(m + beta*dm, rho + beta*drho)
           if h_test < h_current:
               accept this (m_r, rho_r) as the restored iterate
               add (f(m_r), h_test) to filter
               update lam += beta * dlam
               break

   This guarantees the algorithm never stalls completely — if the Newton
   direction has any h-decreasing component, restoration will find it.

3. **Feasibility-restoration sub-problem.**  A more principled approach:
   when the main step fails, solve a small QP (or a few gradient steps) that
   drives h toward zero while keeping the iterate near the constraint manifold.
   This matches the restoration strategy in Wächter & Biegler (IPOPT, 2006).
   Likely overkill for this project but worth noting for the report.

4. **Log restoration events.**  Add `reason='restored'` to the stats dict when
   a restoration step is taken, so convergence plots distinguish normal
   accepted steps from restoration steps.

---

## Current default hyperparameters

| Parameter        | Value   | Paper        | Notes                            |
|------------------|---------|--------------|----------------------------------|
| `tol`            | 1e-4    | 1e-4         | outer KKT tolerance              |
| `max_iter`       | 100     | ~20          |                                  |
| `cg_tol`         | 0.01    | 0.1          | tighter than paper; faster here  |
| `cg_maxiter`     | 200     | N/A          |                                  |
| `max_backtracks` | 20      | not stated   |                                  |
| Filter gamma_f   | 1e-5    | not stated   |                                  |
| Filter gamma_h   | 1e-5    | not stated   |                                  |
| Filter beta      | 10      | not stated   |                                  |
| Filter max_h_factor | 2.0  | not stated   | new; caps per-step h increase    |
