"""Vendored fallback implementation of the observable Markov regime model.

This file is a verbatim copy of the math from the
`markov-hedge-fund-method` Claude Code skill
(installed at ~/.claude/skills/markov-hedge-fund-method/ on the dev machine).

The skill itself is NOT a pip package, so it's not portable across machines.
This vendored copy lets the bot run on any host without that local skill.

`markov.regime_detector` prefers the external skill when present, and falls
back to these functions otherwise. Behaviour is identical — same labels,
same transition matrix MLE, same stationary distribution, same signal.

Author of the underlying framework: Roan (@RohOnChain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["Bear", "Sideways", "Bull"]  # index 0, 1, 2


def label_regimes(close: pd.Series, window: int = 20, threshold: float = 0.02) -> pd.Series:
    """Label each day as Bull (2) / Sideways (1) / Bear (0) from rolling return."""
    rolling_return = close.pct_change(window)
    labels = pd.Series(1, index=close.index, dtype=int)  # default Sideways
    labels[rolling_return > threshold] = 2
    labels[rolling_return < -threshold] = 0
    return labels.dropna()


def build_transition_matrix(labels: pd.Series) -> np.ndarray:
    """MLE estimate of the 3x3 transition matrix from a sequence of labels."""
    n = 3
    counts = np.zeros((n, n), dtype=float)
    arr = labels.to_numpy()
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P with eigenvalue 1, normalised to sum to 1."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.real(eigvecs[:, idx])
    vec = np.abs(vec)
    return vec / vec.sum()


def n_step_forecast(P: np.ndarray, n: int) -> np.ndarray:
    """Chapman-Kolmogorov: P^n is the n-step transition matrix."""
    return np.linalg.matrix_power(P, n)


def signal_from_matrix(P: np.ndarray, current_state: int) -> float:
    """Signed signal: P(next=Bull|s) - P(next=Bear|s)."""
    return float(P[current_state, 2] - P[current_state, 0])
