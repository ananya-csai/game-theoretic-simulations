"""
ad_spend_game.py  —  Model 1: the "advertising war" between two rival firms
---------------------------------------------------------------------------
Two firms sell competing products online. Each chooses an advertising budget
from a small menu (e.g. Low / High). Advertising shifts market share towards
the firm that spends more and may also grow the total market.

    market size     M(s_A, s_B) = M0 + g * (s_A + s_B)
    market share    share_A    = s_A^γ / (s_A^γ + s_B^γ)     (Tullock / "attraction" model)
    profit          π_A        = M * share_A - s_A

γ (gamma) is how sensitive share is to relative spend, g is how much advertising
grows the overall market (g = 0 → pure business-stealing).

With the default numbers (M0 = 100, Low = 10, High = 30, g = 0, γ = 1) the game is
a Prisoner's Dilemma: High spend is a dominant strategy for both firms, but the
resulting equilibrium (High, High) leaves both firms worse off than (Low, Low).
"""
from __future__ import annotations

import numpy as np

from .normal_form import BimatrixGame


def market_share(s_i: float, s_j: float, gamma: float = 1.0) -> float:
    if s_i <= 0 and s_j <= 0:
        return 0.5
    a, b = s_i ** gamma, s_j ** gamma
    return a / (a + b)


def profit(s_i: float, s_j: float, M0: float = 100.0, growth: float = 0.0, gamma: float = 1.0) -> float:
    M = M0 + growth * (s_i + s_j)
    return M * market_share(s_i, s_j, gamma) - s_i


def build_ad_war(
    levels: tuple[float, ...] = (10.0, 30.0),
    labels: tuple[str, ...] = ("Low", "High"),
    M0: float = 100.0,
    growth: float = 0.0,
    gamma: float = 1.0,
    players: tuple[str, str] = ("Firm A", "Firm B"),
) -> BimatrixGame:
    """Construct the symmetric payoff matrix for the advertising war."""
    k = len(levels)
    A = np.zeros((k, k))
    B = np.zeros((k, k))
    for i, si in enumerate(levels):
        for j, sj in enumerate(levels):
            A[i, j] = profit(si, sj, M0, growth, gamma)
            B[i, j] = profit(sj, si, M0, growth, gamma)
    return BimatrixGame(A, B, list(labels), list(labels), players)


def classify_2x2(game: BimatrixGame) -> dict:
    """Classify a symmetric 2x2 game (strategy 0 = 'cooperate' = Low, 1 = 'defect' = High).

    Uses the standard labels:
        R = reward  (Low, Low)     T = temptation (High vs Low)
        P = punish  (High, High)   S = sucker     (Low vs High)
    """
    A = game.A
    R, S, T, P = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    if T > R and R > P and P > S:
        kind = "Prisoner's Dilemma"
    elif T > R and S > P:
        kind = "Chicken (Hawk-Dove)"
    elif R > T and P > S:
        kind = "Stag Hunt (coordination)"
    elif R >= T and S >= P:
        kind = "Harmony (Low spend dominant)"
    elif T > P and P > R:
        kind = "Deadlock (High spend efficient)"
    else:
        kind = "Boundary (ties)"
    return {"R": R, "S": S, "T": T, "P": P, "type": kind}


def critical_discount_factor(game: BimatrixGame) -> float | None:
    """Minimum discount factor δ for which Grim-Trigger sustains (Low, Low) in the
    infinitely repeated game:  δ* = (T - R) / (T - P).  None if not a PD."""
    c = classify_2x2(game)
    T, R, P = c["T"], c["R"], c["P"]
    if not (T > R > P):
        return None
    return float((T - R) / (T - P))


def game_type_grid(high_values, growth_values, low: float = 10.0, M0: float = 100.0, gamma: float = 1.0):
    """Sensitivity analysis: which kind of game arises for each (High budget, market growth) pair."""
    names = ["Prisoner's Dilemma", "Chicken (Hawk-Dove)", "Stag Hunt (coordination)",
             "Harmony (Low spend dominant)", "Deadlock (High spend efficient)", "Boundary (ties)"]
    grid = np.zeros((len(growth_values), len(high_values)), dtype=int)
    for r, g in enumerate(growth_values):
        for c, h in enumerate(high_values):
            kind = classify_2x2(build_ad_war((low, h), M0=M0, growth=g, gamma=gamma))["type"]
            grid[r, c] = names.index(kind)
    return grid, names
