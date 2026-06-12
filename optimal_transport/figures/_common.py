"""Shared constants, style, and I/O helpers for all figure scripts.

Import this module first; it sets the non-interactive Agg backend before
any pyplot calls.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  (must follow use('Agg'))

# ── paths ─────────────────────────────────────────────────────────────────────
_FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_FIGURES_DIR)   # optimal_transport/
_REPO        = os.path.dirname(_PACKAGE_DIR)   # math270c/

# ── canonical grids ───────────────────────────────────────────────────────────
# Define once here; import everywhere so all figures use the same colour/marker.
GRIDS        = [(16, 16, 10), (32, 32, 20), (64, 64, 40)]
GRID_LABELS  = ['16×16×10', '32×32×20', '64×64×40']
GRID_COLORS  = ['tab:blue', 'tab:orange', 'tab:green']
GRID_MARKERS = ['o', 's', '^']

# ── literature-standard solver tolerances ─────────────────────────────────────
CG_TOL    = 1e-4   # cg_sgs path  (Haber & Horesh 2015)
GMRES_TOL = 0.1    # gmres_amg path
SQP_TOL   = 1e-4


def grid_style(grid):
    """Return (color, marker, label) for a grid triple.

    Falls back to gray/diamond/ASCII label for non-canonical grids so
    the scaling figure can include intermediate sizes.
    """
    try:
        idx = GRIDS.index(tuple(grid))
        return GRID_COLORS[idx], GRID_MARKERS[idx], GRID_LABELS[idx]
    except ValueError:
        n1, n2, n3 = grid
        return 'gray', 'D', f'{n1}×{n2}×{n3}'


def apply_style():
    """Set publication rcParams: serif font, sizes to match 11pt body text."""
    matplotlib.rcParams.update({
        'font.family':     'serif',
        'font.size':       11,
        'axes.labelsize':  11,
        'axes.titlesize':  11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })


def out_dir(script: str) -> str:
    """Return (and create) output/figures/<script>/ under the repo root."""
    d = os.path.join(_REPO, 'output', 'figures', script)
    os.makedirs(d, exist_ok=True)
    return d


def save_fig(fig, directory: str, stem: str) -> None:
    """Save *fig* as .png, then close it."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'{stem}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f'  → {path}')
    plt.close(fig)
