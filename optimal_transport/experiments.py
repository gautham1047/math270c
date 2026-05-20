"""
Experiments 1, 2, 3, and 5 from Haber & Horesh 2015, Section 5.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from optimal_transport.test_images import make_images
from optimal_transport.sqp import initialize, sqp
from optimal_transport.multilevel import multilevel_sqp
from optimal_transport.objective import objective
from optimal_transport.grid import div_st
from optimal_transport.utils import print_stats_table


# ---------------------------------------------------------------------------
# Experiment 1: L2, contrast 10, mesh independence  (Tables 1 & 2 analog)
# ---------------------------------------------------------------------------

def experiment1(verbose=True):
    """
    Solve on three grids with contrast=10, p=2.
    Report: SQP iteration count, CG count at first/middle/last SQP step.
    """
    grids   = [(16, 16, 10), (32, 32, 20), (64, 64, 40)]
    results = []

    for (n1, n2, n3) in grids:
        h        = 1.0 / n1
        mu0, mu1 = make_images(n1, n2, contrast=10)
        m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

        if verbose:
            print(f"\n--- Exp 1: grid ({n1},{n2},{n3}) ---")

        m1, m2, rho_free, lam, stats = sqp(
            m1, m2, rho_free, lam, mu0, mu1, h,
            p=2, tol=1e-4, verbose=verbose
        )

        cg_counts = [s['cg_iters'] for s in stats]
        n_iter    = len(stats)
        cg_first  = cg_counts[0]              if n_iter > 0 else 0
        cg_mid    = cg_counts[n_iter // 2]    if n_iter > 1 else cg_first
        cg_last   = cg_counts[-1]             if n_iter > 0 else 0

        results.append({
            'grid':      (n1, n2, n3),
            'sqp_iters': n_iter,
            'cg_first':  cg_first,
            'cg_mid':    cg_mid,
            'cg_last':   cg_last,
        })

    print("\n=== Experiment 1 Summary ===")
    print(f"{'Grid':>20}  {'SQP':>5}  {'CG_1':>6}  {'CG_mid':>6}  {'CG_n':>6}")
    for r in results:
        g = r['grid']
        print(f"  ({g[0]:2d},{g[1]:2d},{g[2]:2d})          "
              f"{r['sqp_iters']:5d}  {r['cg_first']:6d}  {r['cg_mid']:6d}  {r['cg_last']:6d}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: L2, contrast 10, density evolution  (Figure 3)
# ---------------------------------------------------------------------------

def experiment2(save_fig=True, verbose=True):
    """
    Solve 42x42x42, contrast=10, p=2. Plot rho at each time step.
    """
    n1, n2, n3 = 42, 42, 42
    h           = 1.0 / n1
    mu0, mu1    = make_images(n1, n2, contrast=10)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

    if verbose:
        print("\n--- Exp 2: 42x42x42, contrast=10 ---")

    m1, m2, rho_free, lam, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=2, tol=1e-4, verbose=verbose
    )

    # Assemble full rho including boundary slices
    rho_full        = np.empty((n1, n2, n3 + 1))
    rho_full[:,:,0] = mu0
    rho_full[:,:,1:-1] = rho_free
    rho_full[:,:,-1]   = mu1

    # Plot n3+1 time slices in a grid
    n_plots  = n3 + 1
    ncols    = 7
    nrows    = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes_flat = axes.ravel()

    for k in range(n_plots):
        ax = axes_flat[k]
        ax.imshow(rho_full[:, :, k].T, origin='lower',
                  vmin=rho_full.min(), vmax=rho_full.max(), cmap='viridis')
        ax.set_title(f"t={k/n3:.2f}", fontsize=7)
        ax.axis('off')
    for k in range(n_plots, len(axes_flat)):
        axes_flat[k].axis('off')

    plt.suptitle("Experiment 2: density evolution (contrast=10, p=2)", y=1.01)
    plt.tight_layout()

    if save_fig:
        out_dir = os.path.join("output", "exp2")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "density_evolution.png")
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
        print(f"Saved {out_path}")
    plt.show()

    return m1, m2, rho_free, lam, stats


# ---------------------------------------------------------------------------
# Experiment 3: L2, contrast 100, multilevel  (Table 3)
# ---------------------------------------------------------------------------

def experiment3(save_fig=True, verbose=True):
    """
    Multilevel solve with contrast=100, p=2, levels (16,16,8)->(32,32,16)->(64,64,32).
    Plots: (a) density evolution at finest grid, (b) kkt_lam convergence across levels.
    """
    levels = [(16, 16, 8), (32, 32, 16), (64, 64, 32)]

    if verbose:
        print("\n--- Exp 3: multilevel, contrast=100 ---")

    m1, m2, rho_free, lam, all_stats = multilevel_sqp(
        levels, contrast=100, p=2, tol=1e-4, verbose=verbose
    )

    print("\n=== Experiment 3 Summary ===")
    for idx, (lv, stats) in enumerate(zip(levels, all_stats)):
        print(f"  Level {idx+1} grid {lv}: {len(stats)} SQP iters")

    # --- Assemble rho_full at finest grid ---
    n1, n2, n3 = levels[-1]
    mu0, mu1   = make_images(n1, n2, contrast=100)
    rho_full   = np.empty((n1, n2, n3 + 1))
    rho_full[:, :, 0]    = mu0
    rho_full[:, :, 1:-1] = rho_free
    rho_full[:, :, -1]   = mu1

    if save_fig:
        out_dir = os.path.join("output", "exp3")
        os.makedirs(out_dir, exist_ok=True)

    # --- Plot 1: density evolution (finest grid) ---
    n_plots = n3 + 1
    ncols   = 7
    nrows   = int(np.ceil(n_plots / ncols))
    fig1, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes_flat  = axes.ravel()

    vmin, vmax = rho_full.min(), rho_full.max()
    for k in range(n_plots):
        ax = axes_flat[k]
        ax.imshow(rho_full[:, :, k].T, origin='lower',
                  vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f"t={k/n3:.2f}", fontsize=7)
        ax.axis('off')
    for k in range(n_plots, len(axes_flat)):
        axes_flat[k].axis('off')

    plt.suptitle("Experiment 3: density evolution (contrast=100, p=2, multilevel)", y=1.01)
    plt.tight_layout()

    if save_fig:
        path1 = os.path.join(out_dir, "density_evolution.png")
        fig1.savefig(path1, dpi=100, bbox_inches='tight')
        print(f"Saved {path1}")
    plt.show()

    # --- Plot 2: kkt_lam convergence across levels ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    offset = 0
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for idx, (lv, stats) in enumerate(zip(levels, all_stats)):
        if stats:
            iters    = [offset + s['iter']    for s in stats]
            kkt_vals = [s['kkt_lam'] for s in stats]
            ax2.semilogy(iters, kkt_vals, color=colors[idx],
                         marker='o', markersize=3,
                         label=f"Level {idx+1} {lv}")
            offset = iters[-1]

    ax2.set_xlabel("Cumulative SQP iteration")
    ax2.set_ylabel("|C| (constraint violation)")
    ax2.set_title("Experiment 3: multilevel convergence (contrast=100)")
    ax2.legend()
    plt.tight_layout()

    if save_fig:
        path2 = os.path.join(out_dir, "convergence.png")
        fig2.savefig(path2, dpi=100, bbox_inches='tight')
        print(f"Saved {path2}")
    plt.show()

    return m1, m2, rho_free, lam, all_stats


# ---------------------------------------------------------------------------
# Experiment 5: p-continuation from p=2 to p=1.01  (Figure 4)
# ---------------------------------------------------------------------------

def experiment5(save_fig=True, verbose=True):
    """
    One SQP step per p value, tracking primal/dual feasibility as p -> 1.
    """
    n1, n2, n3 = 64, 64, 32
    h           = 1.0 / n1
    mu0, mu1    = make_images(n1, n2, contrast=10)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

    p_values     = np.linspace(2.0, 1.01, 100)
    primal_norms = []
    dual_norms   = []

    if verbose:
        print("\n--- Exp 5: p-continuation ---")

    for p in p_values:
        m1, m2, rho_free, lam, stats = sqp(
            m1, m2, rho_free, lam, mu0, mu1, h,
            p=p, tol=1e-10, max_iter=1, verbose=False
        )
        if stats:
            primal_norms.append(stats[-1]['kkt_lam'])
            dual_norms.append(stats[-1]['kkt_w'])
        else:
            # Already converged at init — record zero
            rho_full = np.empty((n1, n2, n3 + 1))
            rho_full[:,:,0] = mu0; rho_full[:,:,-1] = mu1
            rho_full[:,:,1:-1] = rho_free
            primal_norms.append(np.sum(np.abs(div_st(m1, m2, rho_full, h))))
            dual_norms.append(0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(p_values, primal_norms, label='Primal (|C|)')
    ax.semilogy(p_values, dual_norms,   label='Dual (|grad_w L|)')
    ax.set_xlabel('p')
    ax.set_ylabel('Feasibility norm')
    ax.set_title('Experiment 5: p-continuation from p=2 to p=1.01')
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()

    if save_fig:
        out_dir = os.path.join("output", "exp5")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "p_continuation.png")
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
        print(f"Saved {out_path}")
    plt.show()

    return p_values, primal_norms, dual_norms


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 5], default=1,
                        help="Experiment number to run")
    args = parser.parse_args()

    if args.exp == 1:
        experiment1()
    elif args.exp == 2:
        experiment2()
    elif args.exp == 3:
        experiment3()
    elif args.exp == 5:
        experiment5()
