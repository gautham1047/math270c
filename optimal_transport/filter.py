"""
Fletcher-Leyffer filter for globalization of the SQP method.

A trial point (f_trial, h_trial) is acceptable if it is not dominated by
any existing entry (f_j, h_j) in the filter (with margins gamma_f, gamma_h).
"""


class Filter:
    def __init__(self, gamma_f=1e-5, gamma_h=1e-5, beta=10.0, max_h_factor=2.0):
        self.entries      = []          # list of (f, h) tuples
        self.gamma_f      = gamma_f
        self.gamma_h      = gamma_h
        self.beta         = beta
        self.max_h_factor = max_h_factor
        self.h_max        = None
        self.h_current    = None        # most recently accepted constraint violation

    def initialize(self, f_0, h_0):
        self.entries   = []
        self.h_max     = self.beta * h_0
        self.h_current = h_0

    def is_acceptable(self, f_trial, h_trial):
        if self.h_max is not None and h_trial > self.h_max:
            return False
        # Reject if h would increase by more than max_h_factor from current level.
        # Prevents constraint violation from ratcheting upward across iterations.
        if self.h_current is not None and h_trial > self.max_h_factor * self.h_current:
            return False
        for (f_j, h_j) in self.entries:
            f_ok = f_trial <= f_j - self.gamma_f * h_j
            h_ok = h_trial <= (1.0 - self.gamma_h) * h_j
            if not (f_ok or h_ok):
                return False
        return True

    def accept(self, h_new):
        """Call after a trial point is accepted to update the current-h tracker."""
        self.h_current = h_new

    def add(self, f_new, h_new):
        self.entries = [(f, h) for (f, h) in self.entries
                        if not (f_new <= f and h_new <= h)]
        self.entries.append((f_new, h_new))
