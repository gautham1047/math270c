"""
§4 Tuning Study — wall-clock time sweeps for cg_tol, max_h_factor, and gmres_tol.

Figures produced:
  tuning.[pdf|png]       — (a) cg_tol sweep (cg_sgs) / (b) max_h_factor sweep (cg_sgs)
  tuning_gmres.[pdf|png] — gmres_tol sweep (gmres_amg), single panel

One curve per grid size; each configuration run REPEATS times; median
plotted, min–max shaded.  OMP_NUM_THREADS=1 keeps BLAS/AMG setup stable.

Usage:
    python -m optimal_transport.figures.tuning
    python -m optimal_transport.figures.tuning --no-64
    python -m optimal_transport.figures.tuning --repeats 1    # quick test
    python -m optimal_transport.figures.tuning --no-maxh      # panel (a) only
    python -m optimal_transport.figures.tuning --no-gmres     # skip gmres_tol sweep
"""

# Pin to a single BLAS/OpenMP thread BEFORE numpy is imported so timing
# reflects solver work rather than thread-spawning overhead.
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')

import argparse
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

from ._common import (
    GRIDS, GRID_LABELS, GRID_COLORS, GRID_MARKERS,
    GMRES_TOL, SQP_TOL,
    apply_style, out_dir, save_fig,
)
from ..test_images import make_images
from ..sqp import initialize, sqp

# ── sweep parameters ──────────────────────────────────────────────────────────
CG_TOLS    = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
H_FACTORS  = [1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 5.0]
GMRES_TOLS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

LIT_CG_TOL   = 1e-4   # reference line in panel (a)
LIT_H_FACT   = 1.3    # reference line in panel (b)
LIT_GMRES_TOL = GMRES_TOL  # 0.1 — reference line in gmres figure

CG_MAXITER   = {16: 500, 32: 1000, 64: 2000}
GMRES_MAXITER = 100
P          = 2
CONTRAST   = 10
MAX_ITER   = 100


# ── timed single run ──────────────────────────────────────────────────────────

def _run(grid, cg_tol, max_h_factor):
    """Run one SQP solve and return (walltime_s, sqp_iters, total_inner_iters)."""
    n1, n2, n3 = grid
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=CONTRAST)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
    t0 = time.perf_counter()
    _, _, _, _, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=P, tol=SQP_TOL, max_iter=MAX_ITER,
        cg_tol=cg_tol, cg_maxiter=CG_MAXITER[n1],
        max_h_factor=max_h_factor,
        method='cg_sgs', verbose=False,
    )
    elapsed = time.perf_counter() - t0
    n_sqp   = len(stats)
    n_inner = sum(s['inner_iters'] for s in stats)
    return elapsed, n_sqp, n_inner


def _run_gmres(grid, gmres_tol, max_h_factor):
    """Run one gmres_amg SQP solve; return (walltime_s, sqp_iters, total_inner_iters)."""
    n1, n2, n3 = grid
    h = 1.0 / n1
    mu0, mu1 = make_images(n1, n2, contrast=CONTRAST)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)
    t0 = time.perf_counter()
    _, _, _, _, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=P, tol=SQP_TOL, max_iter=MAX_ITER,
        gmres_tol=gmres_tol, gmres_maxiter=GMRES_MAXITER,
        max_h_factor=max_h_factor,
        method='gmres_amg', verbose=False,
    )
    elapsed = time.perf_counter() - t0
    n_sqp   = len(stats)
    n_inner = sum(s['inner_iters'] for s in stats)
    return elapsed, n_sqp, n_inner


# ── sweep helpers ─────────────────────────────────────────────────────────────

def _run_repeated(grid, cg_tol, max_h_factor, repeats):
    """Run *repeats* times; return list of elapsed times and the last iter counts."""
    times = []
    n_sqp = n_inner = 0
    for _ in range(repeats):
        t, n_sqp, n_inner = _run(grid, cg_tol, max_h_factor)
        times.append(t)
    return times, n_sqp, n_inner


def sweep_cgtol(grids, repeats):
    """
    Returns dict {grid_label: [row, ...]}.
    Each row: {val, times, median, min, max, outer, inner}.
    """
    data = {}
    for grid, label in zip(grids, [GRID_LABELS[GRIDS.index(g)] for g in grids]):
        print(f'\n  [cg_tol] {label}')
        rows = []
        for tol in CG_TOLS:
            times, outer, inner = _run_repeated(grid, tol, LIT_H_FACT, repeats)
            med = float(np.median(times))
            print(f'    {tol:.1e}  median={med:.2f}s  sqp={outer}', flush=True)
            rows.append({
                'val': tol, 'times': times,
                'median': med,
                'min':    float(np.min(times)),
                'max':    float(np.max(times)),
                'outer':  outer, 'inner': inner,
            })
        data[label] = rows
    return data


def sweep_maxh(grids, repeats):
    """
    Returns dict {grid_label: [row, ...]}.
    Each row: {val, times, median, min, max, outer, inner}.
    """
    data = {}
    for grid, label in zip(grids, [GRID_LABELS[GRIDS.index(g)] for g in grids]):
        print(f'\n  [max_h] {label}')
        rows = []
        for hf in H_FACTORS:
            times, outer, inner = _run_repeated(grid, LIT_CG_TOL, hf, repeats)
            med = float(np.median(times))
            print(f'    {hf:.2f}  median={med:.2f}s  sqp={outer}', flush=True)
            rows.append({
                'val': hf, 'times': times,
                'median': med,
                'min':    float(np.min(times)),
                'max':    float(np.max(times)),
                'outer':  outer, 'inner': inner,
            })
        data[label] = rows
    return data


def sweep_gmres(grids, repeats):
    """
    Sweep gmres_tol for the gmres_amg solver path.
    Returns dict {grid_label: [row, ...]}.
    Each row: {val, times, median, min, max, outer, inner}.
    """
    data = {}
    for grid, label in zip(grids, [GRID_LABELS[GRIDS.index(g)] for g in grids]):
        print(f'\n  [gmres_tol] {label}')
        rows = []
        for tol in GMRES_TOLS:
            times = []
            n_sqp = n_inner = 0
            for _ in range(repeats):
                t, n_sqp, n_inner = _run_gmres(grid, tol, LIT_H_FACT)
                times.append(t)
            med = float(np.median(times))
            print(f'    {tol:.3g}  median={med:.2f}s  sqp={n_sqp}', flush=True)
            rows.append({
                'val': tol, 'times': times,
                'median': med,
                'min':    float(np.min(times)),
                'max':    float(np.max(times)),
                'outer':  n_sqp, 'inner': n_inner,
            })
        data[label] = rows
    return data


# ── plot ──────────────────────────────────────────────────────────────────────

def _draw_panel(ax, data, grids_used, x_log, ref_x, ref_label, x_label):
    """Draw one tuning panel: median lines + min–max bands + reference vline."""
    for label, color, marker in zip(GRID_LABELS, GRID_COLORS, GRID_MARKERS):
        if label not in data:
            continue
        rows = data[label]
        xs  = [r['val']    for r in rows]
        med = [r['median'] for r in rows]
        lo  = [r['min']    for r in rows]
        hi  = [r['max']    for r in rows]

        ax.plot(xs, med, color=color, marker=marker,
                linewidth=1.5, markersize=5, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)

        # Star at empirical minimum
        i_min = int(np.argmin(med))
        ax.plot(xs[i_min], med[i_min], '*', color=color,
                markersize=11, zorder=5, linestyle='none')

    # Vertical reference line
    ax.axvline(ref_x, color='gray', linestyle='--', linewidth=1.0, alpha=0.8,
               label=f'{ref_label} ({ref_x})')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Wall-clock time (s)')
    if x_log:
        ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)


def plot_tuning(cgtol_data, maxh_data, directory, grids_used, repeats):
    apply_style()
    has_a = bool(cgtol_data)
    has_b = bool(maxh_data)

    if not has_a and not has_b:
        print('  (nothing to plot)')
        return

    ncols = (1 if has_a else 0) + (1 if has_b else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(7.0 if ncols == 2 else 6.5, 3.2),
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    col = 0
    if has_a:
        _draw_panel(axes[col], cgtol_data, grids_used,
                    x_log=True, ref_x=LIT_CG_TOL, ref_label='literature std.',
                    x_label='Inner CG tolerance')
        axes[col].set_title('(a)')
        col += 1
    if has_b:
        _draw_panel(axes[col], maxh_data, grids_used,
                    x_log=False, ref_x=LIT_H_FACT, ref_label='literature std.',
                    x_label='max_h_factor')
        axes[col].set_title('(b)')

    save_fig(fig, directory, 'tuning')


def plot_gmres_tol(gmres_data, directory, grids_used):
    """Single-panel figure: wall-clock vs gmres_tol for gmres_amg solver."""
    if not gmres_data:
        return
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2), constrained_layout=True)
    _draw_panel(ax, gmres_data, grids_used,
                x_log=True, ref_x=LIT_GMRES_TOL, ref_label='literature std.',
                x_label='GMRES tolerance')
    ax.set_title('GMRES tolerance sweep (gmres_amg)')
    save_fig(fig, directory, 'tuning_gmres')


# ── CSV ───────────────────────────────────────────────────────────────────────

def dump_csvs(cgtol_data, maxh_data, gmres_data, directory):
    def _write(fname, col, data):
        path = os.path.join(directory, fname)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['grid', col, 'outer_iters', 'total_inner_iters',
                        'walltime_s', 'walltime_min', 'walltime_max'])
            for label, rows in data.items():
                for r in rows:
                    w.writerow([label, r['val'], r['outer'], r['inner'],
                                r['median'], r['min'], r['max']])
        print(f'  → {path}')

    if cgtol_data:
        _write('tuning_cgtol.csv', 'cg_tol', cgtol_data)
    if maxh_data:
        _write('tuning_maxh.csv', 'max_h_factor', maxh_data)
    if gmres_data:
        _write('tuning_gmres.csv', 'gmres_tol', gmres_data)


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
                        help='Skip 64x64x40 (saves hours on full sweeps)')
    parser.add_argument('--repeats', type=int, default=3, metavar='N',
                        help='Runs per configuration for median/band (default: 3)')
    parser.add_argument('--no-cgtol', action='store_true',
                        help='Skip cg_tol sweep (panel a)')
    parser.add_argument('--no-maxh', action='store_true',
                        help='Skip max_h_factor sweep (panel b)')
    parser.add_argument('--no-gmres', action='store_true',
                        help='Skip gmres_tol sweep (tuning_gmres figure)')
    args = parser.parse_args()

    grids = GRIDS[:2] if args.no_64 else GRIDS
    directory = out_dir('tuning')

    print(f'Grids: {[f"{g[0]}x{g[1]}x{g[2]}" for g in grids]}')
    print(f'Repeats: {args.repeats}')
    print(f'Output: {directory}')

    cgtol_data = {}
    maxh_data  = {}
    gmres_data = {}

    if not args.no_cgtol:
        print('\n=== Panel (a): cg_tol sweep (cg_sgs) ===')
        cgtol_data = sweep_cgtol(grids, args.repeats)

    if not args.no_maxh:
        print('\n=== Panel (b): max_h_factor sweep (cg_sgs) ===')
        maxh_data = sweep_maxh(grids, args.repeats)

    if not args.no_gmres:
        print('\n=== gmres_tol sweep (gmres_amg) ===')
        gmres_data = sweep_gmres(grids, args.repeats)

    print('\nPlotting…')
    if cgtol_data or maxh_data:
        plot_tuning(cgtol_data, maxh_data, directory, grids, args.repeats)
    if gmres_data:
        plot_gmres_tol(gmres_data, directory, grids)
    if cgtol_data or maxh_data or gmres_data:
        dump_csvs(cgtol_data, maxh_data, gmres_data, directory)
    print('\nDone.')


if __name__ == '__main__':
    main()
