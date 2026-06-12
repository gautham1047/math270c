"""
§6 Extended experiments.

All cases use gmres_amg.  Asymmetric Gaussian runs twice: before-fix (free
BCs, monkey-patched grad_st) and after-fix (no-flux BCs, current code).
Ring and blobs run after-fix only.

Figures produced:
  asym_before_convergence.png   |C|_1 vs SQP iter, all grids, free BCs
  asym_after_convergence.png    |C|_1 vs SQP iter, all grids, no-flux BCs
  asym_before_stills.png        rho snapshots (32×32×20), before fix
  asym_after_stills.png         rho snapshots (32×32×20), after fix
  asym_error_comparison.png     both |C|_1 histories on one axes (32×32×20)
  ring_to_disk_convergence.png  }
  ring_to_disk_stills.png       } after-fix, gmres_amg
  two_blobs_convergence.png     }
  two_blobs_stills.png          }
  scaling.png                   wall-clock and inner-iter vs DOFs, log-log
  scaling.csv

Usage:
    python -m optimal_transport.figures.extension
    python -m optimal_transport.figures.extension --no-64
    python -m optimal_transport.figures.extension --cases asym ring
    python -m optimal_transport.figures.extension --no-scaling
"""

import os
# Pin threads for reproducible wall-clock timing in the scaling figure.
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')

import argparse
import contextlib
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

from ._common import (
    GRIDS, GMRES_TOL, SQP_TOL,
    grid_style, apply_style, out_dir, save_fig,
)
from ..test_images import make_asymmetric_gaussian, make_ring_to_disk, make_two_blobs
from ..sqp import initialize, sqp

# ── registries ────────────────────────────────────────────────────────────────
CASES = {
    'asym':  ('asymmetric_gaussian', 'Asymmetric Gaussian',  make_asymmetric_gaussian),
    'ring':  ('ring_to_disk',        'Ring to Disk',         make_ring_to_disk),
    'blobs': ('two_blobs',           'Two Blobs to Centre',  make_two_blobs),
}

STILLS_GRID   = (32, 32, 20)   # grid for the GIF-replacement stills
GMRES_MAXITER = 50

# Five-point grid sequence for the scaling figure (same aspect ratio n3/n1≈0.625)
SCALING_GRIDS = [(16, 16, 10), (24, 24, 15), (32, 32, 20), (48, 48, 30), (64, 64, 40)]


# ── single solve ──────────────────────────────────────────────────────────────

def _run_gmres(make_fn, grid, verbose=False):
    """Solve one case on one grid with gmres_amg.
    Returns (m1, m2, rho_full, stats, mu0, mu1)."""
    n1, n2, n3 = grid
    h = 1.0 / n1
    mu0, mu1 = make_fn(n1, n2)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
    m1, m2, rho_free, lam, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=2, tol=SQP_TOL,
        method='gmres_amg', gmres_tol=GMRES_TOL, gmres_maxiter=GMRES_MAXITER,
        verbose=verbose,
    )
    rho_full = np.empty((n1, n2, n3 + 1))
    rho_full[:, :, 0]    = mu0
    rho_full[:, :, 1:-1] = rho_free
    rho_full[:, :, -1]   = mu1
    return m1, m2, rho_full, stats, mu0, mu1


def _run_cgsgs(make_fn, grid, verbose=False):
    """Solve one case on one grid with cg_sgs.
    Returns (m1, m2, rho_full, stats, mu0, mu1)."""
    n1, n2, n3 = grid
    h = 1.0 / n1
    mu0, mu1 = make_fn(n1, n2)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
    m1, m2, rho_free, lam, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=2, tol=SQP_TOL,
        method='cg_sgs', verbose=verbose,
    )
    rho_full = np.empty((n1, n2, n3 + 1))
    rho_full[:, :, 0]    = mu0
    rho_full[:, :, 1:-1] = rho_free
    rho_full[:, :, -1]   = mu1
    return m1, m2, rho_full, stats, mu0, mu1


def _run_auto(make_fn, grid, verbose=False):
    return _run_cgsgs(make_fn, grid, verbose=verbose)


# ── convergence figure ────────────────────────────────────────────────────────

def plot_convergence(case_name, all_stats, grids, directory):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for grid, stats in zip(grids, all_stats):
        color, marker, label = grid_style(grid)
        kkt = [s['kkt_lam'] for s in stats]
        ax.semilogy(range(1, len(kkt) + 1), kkt,
                    color=color, marker=marker, linewidth=1.5, markersize=5,
                    label=label)
    ax.set_xlabel('SQP iteration')
    ax.set_ylabel(r'$\|C\|_1$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    save_fig(fig, directory, f'{case_name}_convergence')


# ── stills figure ─────────────────────────────────────────────────────────────

def plot_stills(case_name, rho_full, n3, directory):
    """Six ρ snapshots at t = 0, 0.2, 0.4, 0.6, 0.8, 1.0."""
    apply_style()
    t_fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    indices = [round(t * n3) for t in t_fracs]
    vmin, vmax = rho_full.min(), rho_full.max()

    fig, axes = plt.subplots(1, 6, figsize=(6.5, 1.4), constrained_layout=True)
    ims = []
    for ax, k, t in zip(axes, indices, t_fracs):
        im = ax.imshow(rho_full[:, :, k].T, origin='lower',
                       vmin=vmin, vmax=vmax, cmap='viridis',
                       interpolation='nearest', extent=[0, 1, 0, 1])
        ims.append(im)
        ax.set_title(f't={t:.1f}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(ims[-1], ax=axes, orientation='horizontal',
                 fraction=0.05, pad=0.02, shrink=0.8)
    save_fig(fig, directory, f'{case_name}_stills')


# ── boundary flux: before / after ────────────────────────────────────────────

@contextlib.contextmanager
def _free_boundary_context():
    """
    Temporarily restore pre-fix free-boundary behaviour in grad_st.

    The June 2026 fix zeroed the four spatial boundary face entries in grad_st
    (grid.py).  This context manager reverts that for a diagnostic run on the
    cg_sgs path (the SGS preconditioner's interior-only D matrices are a mild
    inconsistency but don't prevent the 'before' solution from showing leakage).
    """
    import optimal_transport.grid          as _grid_mod
    import optimal_transport.linear_solver as _ls_mod

    orig = _grid_mod.grad_st

    def _free_grad_st(lam, h):
        n1, n2, n3 = lam.shape
        g_m1 = np.zeros((n1 + 1, n2, n3))
        g_m1[1:-1, :, :] = (lam[:-1, :, :] - lam[1:, :, :]) / h
        g_m1[0,  :, :] = -lam[0,  :, :] / h   # old: boundary faces free
        g_m1[-1, :, :] =  lam[-1, :, :] / h
        g_m2 = np.zeros((n1, n2 + 1, n3))
        g_m2[:, 1:-1, :] = (lam[:, :-1, :] - lam[:, 1:, :]) / h
        g_m2[:,  0,  :] = -lam[:,  0,  :] / h
        g_m2[:, -1,  :] =  lam[:, -1,  :] / h
        g_rho_free = (lam[:, :, :-1] - lam[:, :, 1:]) / h
        return g_m1, g_m2, g_rho_free

    _grid_mod.grad_st = _free_grad_st
    _ls_mod.grad_st   = _free_grad_st
    try:
        yield
    finally:
        _grid_mod.grad_st = orig
        _ls_mod.grad_st   = orig


def run_asym_both(grids, verbose=False):
    """
    Run asymmetric Gaussian with cg_sgs for both before-fix and after-fix BCs
    on every grid in *grids*, plus stills at STILLS_GRID.

    Returns (before_stats_list, after_stats_list, before_rho_stills, after_rho_stills).
    """
    stills_grid = STILLS_GRID if STILLS_GRID in grids else grids[-1]
    n1s, n2s, n3s = stills_grid
    h_s = 1.0 / n1s

    after_stats_list  = []
    before_stats_list = []

    print('  After-fix (cg_sgs):')
    for grid in grids:
        n1, n2, n3 = grid
        print(f'    {n1}×{n2}×{n3} …', end=' ', flush=True)
        _, _, _, stats, _, _ = _run_auto(make_asymmetric_gaussian, grid,
                                         verbose=verbose)
        n_inner = sum(s['inner_iters'] for s in stats)
        print(f'{len(stats)} SQP  {n_inner} inner  |C|={stats[-1]["kkt_lam"]:.2e}')
        after_stats_list.append(stats)

    _, _, after_rho_stills, _, _, _ = _run_auto(
        make_asymmetric_gaussian, stills_grid, verbose=False)

    print('  Before-fix (cg_sgs, free BCs):')
    for grid in grids:
        n1, n2, n3 = grid
        h = 1.0 / n1
        mu0, mu1 = make_asymmetric_gaussian(n1, n2)
        m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
        print(f'    {n1}×{n2}×{n3} …', end=' ', flush=True)
        with _free_boundary_context():
            m1, m2, rho_free, lam, stats_bf = sqp(
                m1, m2, rho_free, lam, mu0, mu1, h,
                p=2, tol=SQP_TOL, max_iter=60,
                method='cg_sgs', verbose=verbose,
            )
        n_inner = sum(s['inner_iters'] for s in stats_bf)
        print(f'{len(stats_bf)} SQP  {n_inner} inner  |C|={stats_bf[-1]["kkt_lam"]:.2e}')
        before_stats_list.append(stats_bf)

    # Stills always at STILLS_GRID = 32×32×20, so cg_sgs.
    mu0, mu1 = make_asymmetric_gaussian(n1s, n2s)
    m1b, m2b, rho_free_b, lam_b = initialize(mu0, mu1, n3s, h_s)
    with _free_boundary_context():
        m1b, m2b, rho_free_b, lam_b, _ = sqp(
            m1b, m2b, rho_free_b, lam_b, mu0, mu1, h_s,
            p=2, tol=SQP_TOL, max_iter=60,
            method='cg_sgs', verbose=False,
        )
    before_rho_stills = np.empty((n1s, n2s, n3s + 1))
    before_rho_stills[:, :, 0]    = mu0
    before_rho_stills[:, :, 1:-1] = rho_free_b
    before_rho_stills[:, :, -1]   = mu1

    return before_stats_list, after_stats_list, before_rho_stills, after_rho_stills


def plot_asym_error_comparison(before_stats, after_stats, directory):
    """
    ||C||_1 vs SQP iteration for before-fix and after-fix on one axes.
    Intended for the STILLS_GRID run (single grid).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)

    kkt_before = [s['kkt_lam'] for s in before_stats]
    kkt_after  = [s['kkt_lam'] for s in after_stats]

    ax.semilogy(range(1, len(kkt_before) + 1), kkt_before,
                color='tab:red', linestyle='--', linewidth=1.5,
                marker='o', markersize=4, label='Before fix (free BCs)')
    ax.semilogy(range(1, len(kkt_after) + 1), kkt_after,
                color='tab:blue', linewidth=1.5,
                marker='s', markersize=4, label='After fix (no-flux BCs)')

    ax.set_xlabel('SQP iteration')
    ax.set_ylabel(r'$\|C\|_1$')
    ax.set_title('Asymmetric Gaussian: BC fix effect (gmres_amg, 32×32×20)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    save_fig(fig, directory, 'asym_error_comparison')


# ── scaling figure ────────────────────────────────────────────────────────────

def _total_dofs(n1, n2, n3):
    """Total saddle-point system DOFs (interior m-faces + free ρ + λ)."""
    return (n1 - 1) * n2 * n3 + n1 * (n2 - 1) * n3 + n1 * n2 * (n3 - 1) + n1 * n2 * n3


def run_scaling(grids, repeats):
    """
    Time the asymmetric_gaussian case on each grid *repeats* times with cg_sgs.
    Returns list of dicts: {grid, dofs, inner, times, median, min, max}.
    """
    data = []
    for grid in grids:
        n1, n2, n3 = grid
        dofs = _total_dofs(n1, n2, n3)
        _, _, lbl = grid_style(grid)
        print(f'  {lbl}  dofs={dofs}', end='  ', flush=True)
        times   = []
        n_inner = 0
        for _ in range(repeats):
            h = 1.0 / n1
            mu0, mu1 = make_asymmetric_gaussian(n1, n2)
            m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
            t0 = time.perf_counter()
            _, _, _, _, stats = sqp(
                m1, m2, rho_free, lam, mu0, mu1, h,
                p=2, tol=SQP_TOL,
                method='cg_sgs', verbose=False,
            )
            times.append(time.perf_counter() - t0)
            n_inner = sum(s['inner_iters'] for s in stats)
        med = float(np.median(times))
        print(f'median={med:.2f}s  inner={n_inner}')
        data.append({
            'grid':   grid, 'dofs': dofs, 'inner': n_inner,
            'times':  times,
            'median': med,
            'min':    float(np.min(times)),
            'max':    float(np.max(times)),
        })
    return data


def plot_scaling(data, directory):
    """
    Two-panel log-log: (a) inner iterations vs DOFs, (b) wall-clock vs DOFs.
    Fitted power law α overlaid; reference lines for α=1 and α=4/3.
    """
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                      constrained_layout=True)

    dofs  = np.array([d['dofs']   for d in data], dtype=float)
    inner = np.array([d['inner']  for d in data], dtype=float)
    med   = np.array([d['median'] for d in data], dtype=float)
    lo    = np.array([d['min']    for d in data], dtype=float)
    hi    = np.array([d['max']    for d in data], dtype=float)

    for ax, vals, band_lo, band_hi, ylabel in [
        (ax_a, inner, None, None, 'Total inner iterations'),
        (ax_b, med,   lo,   hi,   'Wall-clock time (s)'),
    ]:
        # Scatter points coloured by grid
        for d, v in zip(data, vals):
            color, marker, label = grid_style(d['grid'])
            ax.loglog(d['dofs'], v, color=color, marker=marker,
                      markersize=7, linewidth=0, label=label)

        # Connecting line
        ax.loglog(dofs, vals, color='k', linewidth=1.0, alpha=0.35, zorder=0)

        # Min-max band (wall-clock panel)
        if band_lo is not None and len(dofs) > 1:
            ax.fill_between(dofs, band_lo, band_hi, color='gray', alpha=0.15)

        # Power-law fit
        if len(dofs) >= 3:
            alpha_fit, log_C = np.polyfit(np.log(dofs), np.log(vals), 1)
            C_fit     = np.exp(log_C)
            dof_range = np.array([dofs.min() * 0.8, dofs.max() * 1.2])
            ax.loglog(dof_range, C_fit * dof_range ** alpha_fit,
                      color='tab:red', linestyle='--', linewidth=1.5,
                      label=f'fit α={alpha_fit:.2f}', zorder=5)

            # Reference slopes through geometric mean of data
            g_dof = np.exp(np.mean(np.log(dofs)))
            g_val = np.exp(np.mean(np.log(vals)))
            for a_ref, ls_ref, lbl_ref in [(1.0, ':', 'α=1'), (4/3, '-.', 'α=4/3')]:
                C_ref = g_val / g_dof ** a_ref
                ax.loglog(dof_range, C_ref * dof_range ** a_ref,
                          color='gray', linestyle=ls_ref, linewidth=1.0,
                          label=lbl_ref, zorder=4)

        ax.set_xlabel('DOFs')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)

    ax_a.set_title('(a)')
    ax_b.set_title('(b)')
    save_fig(fig, directory, 'scaling')


def dump_scaling_csv(data, directory):
    path = os.path.join(directory, 'scaling.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['grid', 'dofs', 'total_inner_iters',
                    'walltime_s_median', 'walltime_min', 'walltime_max'])
        for d in data:
            g = d['grid']
            w.writerow([f'{g[0]}x{g[1]}x{g[2]}', d['dofs'], d['inner'],
                        d['median'], d['min'], d['max']])
    print(f'  → {path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--no-64', action='store_true',
                        help='Skip 64x64x40 in per-case runs and scaling')
    parser.add_argument(
        '--cases', nargs='+', default=['asym', 'ring', 'blobs'],
        choices=['asym', 'ring', 'blobs'],
        help='New cases to run (default: all)',
    )
    parser.add_argument('--no-scaling',  action='store_true',
                        help='Skip scaling / complexity figure')
    parser.add_argument('--scaling-repeats', type=int, default=3, metavar='N',
                        help='Timing repeats for scaling figure (default: 3)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    grids     = GRIDS[:2] if args.no_64 else GRIDS
    directory = out_dir('extension')

    print(f'Output:  {directory}')
    print(f'Method:  gmres_amg  gmres_tol={GMRES_TOL}')
    print(f'Cases:   {args.cases}')
    print(f'Grids:   {[f"{g[0]}x{g[1]}x{g[2]}" for g in grids]}')

    # ── Per-case convergence + stills ─────────────────────────────────────────
    for key in args.cases:
        case_name, title, make_fn = CASES[key]
        stills_grid = STILLS_GRID if STILLS_GRID in grids else grids[-1]
        n3s = stills_grid[2]

        if key == 'asym':
            # Asymmetric Gaussian: run cg_sgs before-fix AND after-fix.
            print(f'\n=== {title} (before-fix and after-fix) ===')
            (before_stats_list, after_stats_list,
             before_rho, after_rho) = run_asym_both(grids, verbose=args.verbose)

            plot_convergence('asym_before', before_stats_list, grids, directory)
            plot_convergence('asym_after',  after_stats_list,  grids, directory)
            plot_stills('asym_before', before_rho, n3s, directory)
            plot_stills('asym_after',  after_rho,  n3s, directory)

            cmp_idx = grids.index(stills_grid) if stills_grid in grids else -1
            plot_asym_error_comparison(
                before_stats_list[cmp_idx], after_stats_list[cmp_idx], directory)

        else:
            # Ring and blobs: after-fix only (cg_sgs ≤32, gmres_amg ≥64).
            print(f'\n=== {title} ===')
            all_stats = []
            for grid in grids:
                n1, n2, n3 = grid
                print(f'  {n1}×{n2}×{n3} …', end=' ', flush=True)
                _, _, _, stats, _, _ = _run_auto(make_fn, grid, verbose=args.verbose)
                n_inner = sum(s['inner_iters'] for s in stats)
                print(f'{len(stats)} SQP  {n_inner} inner  |C|={stats[-1]["kkt_lam"]:.2e}')
                all_stats.append(stats)

            plot_convergence(case_name, all_stats, grids, directory)

            n1s, n2s, _ = stills_grid
            print(f'  stills at {n1s}×{n2s}×{n3s} …', end=' ', flush=True)
            _, _, rho_stills, _, _, _ = _run_auto(make_fn, stills_grid, verbose=False)
            print('done')
            plot_stills(case_name, rho_stills, n3s, directory)

    # ── Scaling ────────────────────────────────────────────────────────────────
    if not args.no_scaling:
        sc_grids = ([g for g in SCALING_GRIDS if g[0] <= 48]
                    if args.no_64 else SCALING_GRIDS)
        print(f'\n=== Scaling ({len(sc_grids)} grids, {args.scaling_repeats} repeats) ===')
        sc_data = run_scaling(sc_grids, args.scaling_repeats)
        plot_scaling(sc_data, directory)
        dump_scaling_csv(sc_data, directory)

    print('\nDone.')


if __name__ == '__main__':
    main()
