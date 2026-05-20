"""
Generate test images mu0 (four corner quarter-circles) and mu1 (centered circle).
"""

import numpy as np


def make_mu0(n1, n2, contrast, radius=0.35):
    """Four quarter-circles at corners, value=contrast inside, 1 outside."""
    v_low  = 1.0
    v_high = contrast * v_low
    x = np.linspace(0.5 / n1, 1.0 - 0.5 / n1, n1)
    y = np.linspace(0.5 / n2, 1.0 - 0.5 / n2, n2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    mu = v_low * np.ones((n1, n2))
    for cx, cy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
        mu[mask] = v_high
    return mu


def make_mu1(n1, n2, contrast, radius=0.35):
    """Single centered circle, value=contrast inside, 1 outside."""
    v_low  = 1.0
    v_high = contrast * v_low
    x = np.linspace(0.5 / n1, 1.0 - 0.5 / n1, n1)
    y = np.linspace(0.5 / n2, 1.0 - 0.5 / n2, n2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    mu = v_low * np.ones((n1, n2))
    mask = (X - 0.5)**2 + (Y - 0.5)**2 <= radius**2
    mu[mask] = v_high
    return mu


def make_images(n1, n2, contrast, radius=0.35):
    """
    Return (mu0, mu1) with mass-normalized mu1 so total masses match.
    """
    mu0 = make_mu0(n1, n2, contrast, radius)
    mu1 = make_mu1(n1, n2, contrast, radius)
    mu1 = mu1 * (mu0.sum() / mu1.sum())
    return mu0, mu1
