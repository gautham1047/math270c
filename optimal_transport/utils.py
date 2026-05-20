"""Small helpers."""

import numpy as np


def reg_norm(x, eps=1e-8):
    """Regularized Euclidean norm: sqrt(x^2 + eps^2) - eps."""
    return np.sqrt(x**2 + eps**2) - eps


def print_stats_table(stats_list, label=""):
    """Print a compact table of SQP iteration statistics."""
    if label:
        print(f"\n{label}")
    header = f"{'iter':>5}  {'f':>12}  {'|C|':>10}  {'|grad_w|':>10}  {'alpha':>7}  {'inner':>6}"
    print(header)
    print("-" * len(header))
    for s in stats_list:
        print(f"{s['iter']:5d}  {s['f']:12.6e}  {s['h_viol']:10.3e}  "
              f"{s['kkt_w']:10.3e}  {s['alpha']:7.4f}  {s['inner_iters']:6}")
