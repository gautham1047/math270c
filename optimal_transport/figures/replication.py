"""
§5 Replication figures (Haber & Horesh 2015, Exps 1–3 and 5).

All computations use literature-standard tolerances (cg_tol=1e-4, sqp_tol=1e-4)
so iteration counts are comparable to the paper's Tables 1–4.

Figures produced
  density_evolution_c10.{pdf,png}   — Exp 2: rho slices, contrast=10, 42×42×42
  density_evolution_c100.{pdf,png}  — Exp 3: rho slices, contrast=100, multilevel
  multilevel_conv.{pdf,png}         -- Exp 3: |C|_1 vs cumulative SQP iteration
  pcont.{pdf,png}                   — Exp 5: primal/dual feasibility vs p

CSVs
  exp1.csv    grid, sqp_iters, inner_first, inner_mid, inner_last
  exp3.csv    grid, sqp_iters

Usage:
    python -m optimal_transport.figures.replication
    python -m optimal_transport.figures.replication --no-64   # skip exp3/exp5
    python -m optimal_transport.figures.replication --exp 1 2 # subset
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from ._common import (
    GRIDS, GRID_COLORS, GRID_MARKERS,
    CG_TOL, SQP_TOL,
    apply_style, out_dir, save_fig,
)
from ..test_images import make_images
from ..sqp import initialize, sqp
from ..multilevel import multilevel_sqp
from ..grid import div_st
from ..objective import grad_m, grad_rho

def _cg_maxiter(n1, n2, n3):
    """Scale CG budget with system size so larger grids get sufficient iterations."""
    return max(500, n1 * n2 * n3 // 50)


# ── Experiment 1: mesh independence ──────────────────────────────────────────

def run_exp1(grids, verbose=False):
    """
    Run paper test case (contrast=10, p=2) on *grids*.
    Returns list of dicts: {grid, sqp_iters, inner_first, inner_mid, inner_last}.
    """
    results = []
    for grid in grids:
        n1, n2, n3 = grid
        h = 1.0 / n1
        mu0, mu1 = make_images(n1, n2, contrast=10)
        m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
        print(f'  Exp 1  {n1}×{n2}×{n3} …', end=' ', flush=True)
        _, _, _, _, stats = sqp(
            m1, m2, rho_free, lam, mu0, mu1, h,
            p=2, tol=SQP_TOL, cg_tol=CG_TOL, cg_maxiter=_cg_maxiter(n1, n2, n3),
            method='cg_sgs', verbose=verbose,
        )
        n_iter   = len(stats)
        ic       = [s['inner_iters'] for s in stats]
        i_first  = ic[0]           if n_iter > 0 else 0
        i_mid    = ic[n_iter // 2] if n_iter > 1 else i_first
        i_last   = ic[-1]          if n_iter > 0 else 0
        print(f'{n_iter} SQP  inner=[{i_first},{i_mid},{i_last}]')
        results.append({
            'grid': grid, 'sqp_iters': n_iter,
            'inner_first': i_first, 'inner_mid': i_mid, 'inner_last': i_last,
        })
    return results


# ── Experiment 2: density evolution (contrast=10, 42×42×42) ──────────────────

def run_exp2(verbose=False):
    """Solve 42×42×42, contrast=10.  Returns rho_full (43 time slices)."""
    n1, n2, n3 = 42, 42, 42
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=10)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
    print(f'  Exp 2  42×42×42 …', end=' ', flush=True)
    m1_out, m2_out, rho_free, _, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=2, tol=SQP_TOL, cg_tol=CG_TOL, cg_maxiter=_cg_maxiter(n1, n2, n3),
        method='cg_sgs', max_iter=200, verbose=verbose,
    )
    rho_full = np.empty((n1, n2, n3 + 1))
    rho_full[:, :, 0]    = mu0
    rho_full[:, :, 1:-1] = rho_free
    rho_full[:, :, -1]   = mu1
    final_kkt = np.sum(np.abs(div_st(m1_out, m2_out, rho_full, h)))
    print(f'{len(stats)} SQP  |C|_init={stats[0]["kkt_lam"]:.2e}  |C|_final={final_kkt:.2e}')
    return rho_full, n3


# ── Experiment 3: multilevel, contrast=100 ────────────────────────────────────

def run_exp3(verbose=False, no_64=False):
    """
    Multilevel solve: (16,16,8)→(32,32,16)→(64,64,32) (or just first two levels).
    Returns (rho_full_finest, all_stats, levels).
    """
    levels = [(16, 16, 8), (32, 32, 16)] if no_64 else [(16, 16, 8), (32, 32, 16), (64, 64, 32)]
    print(f'  Exp 3  multilevel contrast=100  levels={[f"{l[0]}×{l[1]}×{l[2]}" for l in levels]}')
    _, _, rho_free, _, all_stats = multilevel_sqp(
        levels, contrast=100, p=2, tol=SQP_TOL,
        cg_tol=CG_TOL, cg_maxiter=_cg_maxiter(*levels[-1]),
        verbose=verbose, method='cg_sgs',
    )
    n1, n2, n3 = levels[-1]
    mu0, mu1   = make_images(n1, n2, contrast=100)
    rho_full   = np.empty((n1, n2, n3 + 1))
    rho_full[:, :, 0]    = mu0
    rho_full[:, :, 1:-1] = rho_free
    rho_full[:, :, -1]   = mu1
    for lv, st in zip(levels, all_stats):
        print(f'    {lv[0]}×{lv[1]}×{lv[2]}: {len(st)} SQP iters')
    return rho_full, all_stats, levels


# ── Experiment 5: p-continuation ──────────────────────────────────────────────

def run_exp5(verbose=False, n_p=100, max_inner=5, tol_factor=5.0):
    """
    p-continuation from p=2 to p=1.01 on 64×64×32.
    Returns (p_values, primal_1step, dual_1step).
    """
    n1, n2, n3 = 64, 64, 32
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=10)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

    n_cells  = n1 * n2 * n3
    n_dof_w  = (n1 + 1) * n2 * n3 + n1 * (n2 + 1) * n3 + n1 * n2 * (n3 - 1)

    def kkt_norms(m1, m2, rho_free, lam, p):
        rf = np.empty((n1, n2, n3 + 1))
        rf[:, :, 0]    = mu0
        rf[:, :, -1]   = mu1
        rf[:, :, 1:-1] = rho_free
        glam     = div_st(m1, m2, rf, h)
        gm1, gm2 = grad_m(m1, m2, rf, lam, h, p)
        grho     = grad_rho(m1, m2, rf, lam, h, p)
        primal = np.sum(np.abs(glam)) * h / n_cells
        dual   = (np.sum(np.abs(gm1)) + np.sum(np.abs(gm2))
                  + np.sum(np.abs(grho))) / n_dof_w
        return primal, dual

    # Converge fully at p=2
    print('  Exp 5  converging at p=2 …', end=' ', flush=True)
    m1, m2, rho_free, lam, _ = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=2, tol=SQP_TOL, cg_tol=CG_TOL, cg_maxiter=_cg_maxiter(n1, n2, n3),
        method='cg_sgs', verbose=verbose,
    )
    pr_ref, du_ref = kkt_norms(m1, m2, rho_free, lam, p=2)
    primal_tol = tol_factor * pr_ref
    dual_tol   = tol_factor * du_ref
    print(f'done  primal_ref={pr_ref:.2e}  dual_ref={du_ref:.2e}')

    p_values     = np.linspace(2.0, 1.01, n_p)
    primal_1step = []
    dual_1step   = []

    print(f'  Exp 5  continuation  ({n_p} p values) …')
    for i, p in enumerate(p_values, 1):
        pr1 = du1 = None
        for inner in range(1, max_inner + 1):
            m1, m2, rho_free, lam, _ = sqp(
                m1, m2, rho_free, lam, mu0, mu1, h,
                p=p, tol=1e-10, max_iter=1, hessian_eps=1e-4,
                cg_tol=CG_TOL, cg_maxiter=_cg_maxiter(n1, n2, n3),
                method='cg_sgs', verbose=False,
            )
            pr, du = kkt_norms(m1, m2, rho_free, lam, p)
            if inner == 1:
                pr1, du1 = pr, du
            if pr < primal_tol and du < dual_tol:
                break
        primal_1step.append(pr1)
        dual_1step.append(du1)
        if i % 20 == 0 or i == n_p:
            print(f'    p={p:.3f}  primal={pr1:.2e}  dual={du1:.2e}')

    return p_values, primal_1step, dual_1step


# ── Density-evolution strip ───────────────────────────────────────────────────

def plot_density_strip(rho_full, n3, n_slices, directory, stem):
    """
    Single row of *n_slices* ρ heatmaps at evenly-spaced t values.
    Shared colour scale; one horizontal colorbar beneath.
    """
    apply_style()
    indices = np.round(np.linspace(0, n3, n_slices)).astype(int)
    vmin    = rho_full.min()
    vmax    = rho_full.max()

    fig, axes = plt.subplots(1, n_slices,
                             figsize=(6.5, 1.4),
                             constrained_layout=True)

    ims = []
    for ax, k in zip(axes, indices):
        im = ax.imshow(rho_full[:, :, k].T, origin='lower',
                       vmin=vmin, vmax=vmax, cmap='viridis',
                       interpolation='nearest',
                       extent=[0, 1, 0, 1])
        ims.append(im)
        ax.set_title(f't={k/n3:.2f}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(ims[-1], ax=axes, orientation='horizontal',
                 fraction=0.05, pad=0.02, shrink=0.8)
    save_fig(fig, directory, stem)


# ── Multilevel convergence ────────────────────────────────────────────────────

def plot_multilevel_conv(all_stats, levels, directory):
    """
    |C|₁ vs cumulative SQP iteration; one coloured segment per level;
    dotted vertical line at each level transition.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    offset = 0
    for idx, (level, stats) in enumerate(zip(levels, all_stats)):
        if not stats:
            continue
        color  = GRID_COLORS[idx] if idx < len(GRID_COLORS) else 'gray'
        marker = GRID_MARKERS[idx] if idx < len(GRID_MARKERS) else 'o'
        iters  = [offset + s['iter'] for s in stats]
        kkt    = [s['kkt_lam'] for s in stats]
        n1, n2, n3 = level
        ax.semilogy(iters, kkt, color=color, marker=marker,
                    linewidth=1.5, markersize=5,
                    label=f'{n1}×{n2}×{n3}')

        # Dotted transition line and label (not at the last level)
        if idx < len(levels) - 1:
            ax.axvline(iters[-1], color='gray', linestyle=':', linewidth=1.0)
            n1_next = levels[idx + 1][0]
            n3_next = levels[idx + 1][2]
            ax.text(iters[-1] + 0.3, kkt[-1],
                    f'→{n1_next}²×{n3_next}',
                    fontsize=8, va='center')
        offset = iters[-1]

    ax.set_xlabel('Cumulative SQP iteration')
    ax.set_ylabel(r'$\|C\|_1$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    save_fig(fig, directory, 'multilevel_conv')


# ── p-continuation ────────────────────────────────────────────────────────────

def plot_pcont(p_values, primal, dual, directory):
    """
    Two-panel figure: primal feasibility (left) and dual feasibility (right),
    both vs p with inverted x-axis (p decreases left→right, approaching 1).
    Log y-scale.  Shaded region marks where feasibility starts rising.
    """
    apply_style()
    fig, (ax_pr, ax_du) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                        constrained_layout=True)

    def _find_divergence(vals):
        """Index of first point where the series exceeds 1.5× its running min."""
        running_min = vals[0]
        for i in range(1, len(vals)):
            if vals[i] > 1.5 * running_min:
                return i
            running_min = min(running_min, vals[i])
        return None

    for ax, vals, ylabel in [
        (ax_pr, primal, r'$\|\nabla_\lambda L\|$ per cell'),
        (ax_du, dual,   r'$\|\nabla_w L\|$ per DOF'),
    ]:
        ax.semilogy(p_values, vals, color='tab:blue', linewidth=1.5)
        ax.invert_xaxis()   # p decreases left→right
        ax.set_xlabel('p')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.3)

        # Shade the diverging region (feasibility climbing toward p→1)
        div_idx = _find_divergence(vals)
        if div_idx is not None:
            p_div = p_values[div_idx]
            ax.axvspan(p_values[-1], p_div, color='red', alpha=0.08,
                       label='diverging')
            ax.axvline(p_div, color='red', linestyle=':', linewidth=1.0)

        # Annotation pointing at paper's expected behavior
        ax.text(0.97, 0.95, 'paper: monotone ↓\n(Fig. 4)',
                transform=ax.transAxes, fontsize=7, ha='right', va='top',
                color='gray',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.7))

    ax_pr.set_title('(a) Primal')
    ax_du.set_title('(b) Dual')
    save_fig(fig, directory, 'pcont')


# ── CSVs ──────────────────────────────────────────────────────────────────────

def dump_csvs(exp1_results, exp3_data, directory):
    # exp1.csv
    if exp1_results:
        path = os.path.join(directory, 'exp1.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['grid', 'sqp_iters', 'inner_first', 'inner_mid', 'inner_last'])
            for r in exp1_results:
                g = r['grid']
                w.writerow([f'{g[0]}x{g[1]}x{g[2]}', r['sqp_iters'],
                            r['inner_first'], r['inner_mid'], r['inner_last']])
        print(f'  → {path}')

    # exp3.csv
    if exp3_data is not None:
        _, all_stats, levels = exp3_data
        path = os.path.join(directory, 'exp3.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['grid', 'sqp_iters'])
            for level, stats in zip(levels, all_stats):
                w.writerow([f'{level[0]}x{level[1]}x{level[2]}', len(stats)])
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
                        help='Skip exp3 finest level and exp5 (both use 64x64 grids)')
    parser.add_argument(
        '--exp', nargs='+', default=['1', '2', '3', '5'],
        choices=['1', '2', '3', '5'],
        help='Which experiments to run (default: all)',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    exps      = set(args.exp)
    directory = out_dir('replication')
    grids_e1  = GRIDS[:2] if args.no_64 else GRIDS

    print(f'Output: {directory}')
    print(f'Experiments: {sorted(exps)}')
    print(f'Literature tolerances: cg_tol={CG_TOL}, sqp_tol={SQP_TOL}')

    exp1_results = None
    exp3_data    = None

    # ── Exp 1 ─────────────────────────────────────────────────────────────────
    if '1' in exps:
        print('\n=== Experiment 1: mesh independence ===')
        exp1_results = run_exp1(grids_e1, verbose=args.verbose)

    # ── Exp 2 ─────────────────────────────────────────────────────────────────
    if '2' in exps:
        print('\n=== Experiment 2: density evolution (contrast=10) ===')
        rho_full_c10, n3_c10 = run_exp2(verbose=args.verbose)
        print('  Plotting density strip …')
        plot_density_strip(rho_full_c10, n3_c10, n_slices=8,
                           directory=directory, stem='density_evolution_c10')

    # ── Exp 3 ─────────────────────────────────────────────────────────────────
    if '3' in exps:
        print('\n=== Experiment 3: multilevel, contrast=100 ===')
        exp3_data = run_exp3(verbose=args.verbose, no_64=args.no_64)
        rho_full_c100, all_stats_c100, levels_c100 = exp3_data
        n3_c100 = levels_c100[-1][2]
        print('  Plotting density strip …')
        plot_density_strip(rho_full_c100, n3_c100, n_slices=8,
                           directory=directory, stem='density_evolution_c100')
        print('  Plotting multilevel convergence …')
        plot_multilevel_conv(all_stats_c100, levels_c100, directory)

    # ── Exp 5 ─────────────────────────────────────────────────────────────────
    if '5' in exps:
        if args.no_64:
            print('\n=== Experiment 5: skipped (uses 64×64×32, pass without --no-64) ===')
        else:
            print('\n=== Experiment 5: p-continuation ===')
            p_values, primal, dual = run_exp5(verbose=args.verbose)
            print('  Plotting p-continuation …')
            plot_pcont(p_values, primal, dual, directory)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    print('\nWriting CSVs …')
    dump_csvs(exp1_results, exp3_data, directory)

    print('\nDone.')


if __name__ == '__main__':
    main()
