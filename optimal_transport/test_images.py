"""
Test density pairs (mu0, mu1) for the Benamou-Brenier SQP solver.

All functions return non-negative arrays of shape (n1, n2) with mu1
mass-matched to mu0.  The solver is most stable when the two totals agree.
"""

import numpy as np


def _grid(n1, n2):
    x = np.linspace(0.5 / n1, 1.0 - 0.5 / n1, n1)
    y = np.linspace(0.5 / n2, 1.0 - 0.5 / n2, n2)
    return np.meshgrid(x, y, indexing='ij')


def make_mu0(n1, n2, contrast, radius=0.35):
    """Four quarter-circles at corners, value=contrast inside, 1 outside."""
    v_low  = 1.0
    v_high = contrast * v_low
    X, Y = _grid(n1, n2)
    mu = v_low * np.ones((n1, n2))
    for cx, cy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
        mu[mask] = v_high
    return mu


def make_mu1(n1, n2, contrast, radius=0.35):
    """Single centered circle, value=contrast inside, 1 outside."""
    v_low  = 1.0
    v_high = contrast * v_low
    X, Y = _grid(n1, n2)
    mu = v_low * np.ones((n1, n2))
    mask = (X - 0.5)**2 + (Y - 0.5)**2 <= radius**2
    mu[mask] = v_high
    return mu


def make_images(n1, n2, contrast, radius=0.35):
    """Four quarter-circles -> centered circle (paper default)."""
    mu0 = make_mu0(n1, n2, contrast, radius)
    mu1 = make_mu1(n1, n2, contrast, radius)
    mu1 = mu1 * (mu0.sum() / mu1.sum())
    return mu0, mu1


# ---------------------------------------------------------------------------
# New test problems
# ---------------------------------------------------------------------------

def make_asymmetric_gaussian(n1, n2, sigma_src=0.15, sigma_tgt=0.08,
                              center_src=(0.25, 0.25), center_tgt=(0.75, 0.75)):
    """
    Wide Gaussian at bottom-left -> narrow Gaussian at top-right.
    Tests off-centre transport with a simultaneous width change.
    """
    X, Y = _grid(n1, n2)
    cx0, cy0 = center_src
    cx1, cy1 = center_tgt
    mu0 = np.exp(-((X - cx0)**2 + (Y - cy0)**2) / (2 * sigma_src**2))
    mu1 = np.exp(-((X - cx1)**2 + (Y - cy1)**2) / (2 * sigma_tgt**2))
    mu1 = mu1 * (mu0.sum() / mu1.sum())
    return mu0, mu1


def make_ring_to_disk(n1, n2, ring_radius=0.32, ring_width=0.06, disk_sigma=0.08):
    """
    Gaussian annular ring -> central Gaussian disk.
    Tests transport from a non-simply-connected (non-convex) source region.
    """
    X, Y = _grid(n1, n2)
    r    = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    mu0  = np.exp(-(r - ring_radius)**2 / (2 * ring_width**2))
    mu1  = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * disk_sigma**2))
    mu1  = mu1 * (mu0.sum() / mu1.sum())
    return mu0, mu1


def make_two_blobs(n1, n2, sigma=0.10, blob_sep=0.40):
    """
    Two separated Gaussian blobs -> single central Gaussian.
    Tests transport from a disconnected source support.
    """
    X, Y  = _grid(n1, n2)
    cx0a  = 0.5 - blob_sep / 2
    cx0b  = 0.5 + blob_sep / 2
    mu0   = (np.exp(-((X - cx0a)**2 + (Y - 0.5)**2) / (2 * sigma**2)) +
             np.exp(-((X - cx0b)**2 + (Y - 0.5)**2) / (2 * sigma**2)))
    mu1   = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma**2))
    mu1   = mu1 * (mu0.sum() / mu1.sum())
    return mu0, mu1
