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
from optimal_transport.objective import objective, grad_m, grad_rho
from optimal_transport.grid import div_st
from optimal_transport.utils import print_stats_table


# ---------------------------------------------------------------------------
# Experiment 1: L2, contrast 10, mesh independence  (Tables 1 & 2 analog)
# ---------------------------------------------------------------------------

def experiment1(verbose=True, method='cg_sgs', grids=None):
    """
    Solve on three grids with contrast=10, p=2.
    Report: SQP iteration count, inner-solver iteration count at first/mid/last SQP step.

    method : 'cg_sgs' (default) or 'gmres_amg'
    grids  : list of (n1, n2, n3) tuples; defaults to [(16,16,10),(32,32,20),(64,64,40)]
    """
    if grids is None:
        grids = [(16, 16, 10), (32, 32, 20), (64, 64, 40)]
    results = []
    label   = 'GMRES' if method == 'gmres_amg' else 'CG'

    for (n1, n2, n3) in grids:
        h        = 1.0 / n1
        mu0, mu1 = make_images(n1, n2, contrast=10)
        m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

        if verbose:
            print(f"\n--- Exp 1 [{method}]: grid ({n1},{n2},{n3}) ---")

        m1, m2, rho_free, lam, stats = sqp(
            m1, m2, rho_free, lam, mu0, mu1, h,
            p=2, tol=1e-4, verbose=verbose, method=method
        )

        inner_counts = [s['inner_iters'] for s in stats]
        n_iter       = len(stats)
        i_first      = inner_counts[0]           if n_iter > 0 else 0
        i_mid        = inner_counts[n_iter // 2] if n_iter > 1 else i_first
        i_last       = inner_counts[-1]          if n_iter > 0 else 0

        results.append({
            'grid':      (n1, n2, n3),
            'sqp_iters': n_iter,
            'i_first':   i_first,
            'i_mid':     i_mid,
            'i_last':    i_last,
        })

    col = f'{label}_1'
    print(f"\n=== Experiment 1 Summary [{method}] ===")
    print(f"{'Grid':>20}  {'SQP':>5}  {col:>7}  {label+'_mid':>7}  {label+'_n':>7}")
    for r in results:
        g = r['grid']
        print(f"  ({g[0]:2d},{g[1]:2d},{g[2]:2d})          "
              f"{r['sqp_iters']:5d}  {r['i_first']:7d}  {r['i_mid']:7d}  {r['i_last']:7d}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: L2, contrast 10, density evolution  (Figure 3)
# ---------------------------------------------------------------------------

def experiment2(save_fig=True, method='cg_sgs', verbose=True):
    """
    Solve 42x42x42, contrast=10, p=2. Plot rho at each time step.
    
    method : 'cg_sgs' (default) or 'gmres_amg'
    grids  : list of (n1, n2, n3) tuples; defaults to [(16,16,10),(32,32,20),(64,64,40)]
    """
    n1, n2, n3 = 42, 42, 42
    h           = 1.0 / n1
    mu0, mu1    = make_images(n1, n2, contrast=10)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

    if verbose:
        print("\n--- Exp 2: 42x42x42, contrast=10 ---")

    m1, m2, rho_free, lam, stats = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=2, tol=1e-4, verbose=verbose, method=method
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

def experiment3(save_fig=True, method='cg_sgs', verbose=True):
    """
    Multilevel solve with contrast=100, p=2, levels (16,16,8)->(32,32,16)->(64,64,32).
    Plots: (a) density evolution at finest grid, (b) kkt_lam convergence across levels.

    method : 'cg_sgs' (default) or 'gmres_amg'
    grids  : list of (n1, n2, n3) tuples; defaults to [(16,16,10),(32,32,20),(64,64,40)]
    """
    levels = [(16, 16, 8), (32, 32, 16), (64, 64, 32)]

    if verbose:
        print("\n--- Exp 3: multilevel, contrast=100 ---")

    m1, m2, rho_free, lam, all_stats = multilevel_sqp(
        levels, contrast=100, p=2, tol=1e-4, verbose=verbose, method=method
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

def experiment5(save_fig=True, method='cg_sgs', verbose=True, max_inner=5, tol_factor=5.0):
    """
    p-continuation from p=2 to p=1.01 (100 steps).

    Faithful to Haber & Horesh (2015) Sec 5:
      - Fully converge at p=2 before starting continuation.
      - Adaptive inner loop: up to max_inner SQP steps per p value, stopping
        early once both norms drop below tol_factor * (reference norms at p=2).
      - Saves two figures: one after exactly 1 SQP step per p, one after the
        adaptive loop, to show the effect of the extra inner iterations.

    method : 'cg_sgs' (default) or 'gmres_amg'
    grids  : list of (n1, n2, n3) tuples; defaults to [(16,16,10),(32,32,20),(64,64,40)]
    """
    n1, n2, n3 = 64, 64, 32
    h           = 1.0 / n1
    mu0, mu1    = make_images(n1, n2, contrast=10)
    m1, m2, rho_free, lam = initialize(mu0, mu1, n3, h)

    n_cells = n1 * n2 * n3
    n_dof_w = (n1 + 1) * n2 * n3 + n1 * (n2 + 1) * n3 + n1 * n2 * (n3 - 1)

    def kkt_norms(m1, m2, rho_free, lam, p):
        rho_full = np.empty((n1, n2, n3 + 1))
        rho_full[:, :, 0]    = mu0
        rho_full[:, :, -1]   = mu1
        rho_full[:, :, 1:-1] = rho_free
        glam     = div_st(m1, m2, rho_full, h)
        gm1, gm2 = grad_m(m1, m2, rho_full, lam, h, p)
        grho     = grad_rho(m1, m2, rho_full, lam, h, p)
        primal = np.sum(np.abs(glam)) * h / n_cells
        dual   = (np.sum(np.abs(gm1)) + np.sum(np.abs(gm2))
                  + np.sum(np.abs(grho))) / n_dof_w
        return primal, dual

    # Step 1: converge fully at p=2.
    if verbose:
        print("\n--- Exp 5: converging at p=2 ---")
    m1, m2, rho_free, lam, _ = sqp(
        m1, m2, rho_free, lam, mu0, mu1, h,
        p=2, verbose=verbose, method=method
    )

    # Use the post-convergence norms as the threshold reference.
    pr_ref, du_ref = kkt_norms(m1, m2, rho_free, lam, p=2)
    primal_tol = tol_factor * pr_ref
    dual_tol   = tol_factor * du_ref
    if verbose:
        print(f"\n  Reference norms after p=2 convergence:")
        print(f"    primal={pr_ref:.3e}  dual={du_ref:.3e}")
        print(f"  Inner-loop thresholds (x{tol_factor}):")
        print(f"    primal_tol={primal_tol:.3e}  dual_tol={dual_tol:.3e}")

    # Step 2: adaptive continuation.
    p_values = np.linspace(2.0, 1.01, 100)

    # one-step norms: recorded after exactly the first inner SQP step
    primal_1step = []
    dual_1step   = []
    # adaptive norms: recorded after the inner loop exits
    primal_adapt = []
    dual_adapt   = []

    if verbose:
        print(f"\n--- Exp 5: p-continuation (adaptive, max_inner={max_inner}) ---")
        print(f"  {'step':>4}  {'p':>6}  {'pr_1':>10}  {'du_1':>10}  "
              f"{'pr_fin':>10}  {'du_fin':>10}  {'inner':>6}  {'cg_tot':>7}")
        print("  " + "-" * 78)

    for i, p in enumerate(p_values, 1):
        total_cg = 0
        pr1 = du1 = None

        for inner in range(1, max_inner + 1):
            m1, m2, rho_free, lam, stats = sqp(
                m1, m2, rho_free, lam, mu0, mu1, h,
                p=p, tol=1e-10, max_iter=1, hessian_eps=1e-4, 
                verbose=False, method=method
            )
            if stats:
                total_cg += stats[-1]['inner_iters']

            pr, du = kkt_norms(m1, m2, rho_free, lam, p)

            if inner == 1:
                pr1, du1 = pr, du   # save the 1-step result

            if pr < primal_tol and du < dual_tol:
                break

        primal_1step.append(pr1)
        dual_1step.append(du1)
        primal_adapt.append(pr)
        dual_adapt.append(du)

        if verbose:
            print(f"  {i:4d}  {p:6.4f}  {pr1:10.3e}  {du1:10.3e}  "
                  f"{pr:10.3e}  {du:10.3e}  {inner:6d}  {total_cg:7d}")

    # Step 3: save two figures.
    out_dir = os.path.join("output", "exp5")
    if save_fig:
        os.makedirs(out_dir, exist_ok=True)

    def save_panel_fig(primal, dual, title, filename):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(p_values, primal, color='tab:blue')
        ax1.set_xlabel('p')
        ax1.set_ylabel(r'$\|\nabla_\lambda L\|$ per cell')
        ax1.set_title('Primal feasibility')
        ax1.invert_xaxis()
        ax1.grid(True, alpha=0.3)
        ax2.plot(p_values, dual, color='tab:orange')
        ax2.set_xlabel('p')
        ax2.set_ylabel(r'$\|\nabla_w L\|$ per DOF')
        ax2.set_title('Dual feasibility')
        ax2.invert_xaxis()
        ax2.grid(True, alpha=0.3)
        fig.suptitle(title, y=1.02)
        plt.tight_layout()
        if save_fig:
            path = os.path.join(out_dir, filename)
            fig.savefig(path, dpi=120, bbox_inches='tight')
            print(f"Saved {path}")
        plt.show()

    save_panel_fig(
        primal_1step, dual_1step,
        'Experiment 5: p-continuation (1 SQP step per p)',
        'p_continuation_1step.png',
    )
    save_panel_fig(
        primal_adapt, dual_adapt,
        f'Experiment 5: p-continuation (adaptive, max {max_inner} steps per p)',
        'p_continuation_adaptive.png',
    )

    return p_values, primal_1step, dual_1step, primal_adapt, dual_adapt


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
