"""
normal_form.py
--------------
A small, self-contained solver for two-player normal-form (bimatrix) games.

Used by the advertising-war model (Model 1). Everything is written from scratch
(no game-theory library) so that each step of the solution is visible:

* pure-strategy Nash equilibria (mutual best responses)
* strictly dominant strategies
* iterated elimination of strictly dominated strategies (IESDS)
* mixed-strategy Nash equilibria by support enumeration
* Pareto-optimal outcomes (to show when equilibrium is inefficient)

The mixed-strategy solver is cross-checked against the `nashpy` library in the
test-suite.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

TOL = 1e-9


@dataclass
class MixedEquilibrium:
    row_strategy: np.ndarray
    col_strategy: np.ndarray
    row_payoff: float
    col_payoff: float

    def is_pure(self) -> bool:
        return bool(np.isclose(self.row_strategy.max(), 1.0) and np.isclose(self.col_strategy.max(), 1.0))


@dataclass
class BimatrixGame:
    """Two-player game. A[i, j] = row player's payoff, B[i, j] = column player's payoff."""

    A: np.ndarray
    B: np.ndarray
    row_labels: list[str] = field(default_factory=list)
    col_labels: list[str] = field(default_factory=list)
    players: tuple[str, str] = ("Firm A", "Firm B")

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        if self.A.shape != self.B.shape:
            raise ValueError("Payoff matrices must have the same shape")
        m, n = self.A.shape
        if not self.row_labels:
            self.row_labels = [f"R{i}" for i in range(m)]
        if not self.col_labels:
            self.col_labels = [f"C{j}" for j in range(n)]

    @property
    def shape(self) -> tuple[int, int]:
        return self.A.shape

    # ------------------------------------------------------------------ pure NE
    def pure_nash(self) -> list[tuple[int, int]]:
        """All cells where each player is playing a best response to the other."""
        eqs = []
        m, n = self.shape
        for i in range(m):
            for j in range(n):
                row_best = self.A[i, j] >= self.A[:, j].max() - TOL
                col_best = self.B[i, j] >= self.B[i, :].max() - TOL
                if row_best and col_best:
                    eqs.append((i, j))
        return eqs

    def best_responses(self) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Row player's best replies to each column, and column player's to each row."""
        m, n = self.shape
        row_br = {j: [i for i in range(m) if self.A[i, j] >= self.A[:, j].max() - TOL] for j in range(n)}
        col_br = {i: [j for j in range(n) if self.B[i, j] >= self.B[i, :].max() - TOL] for i in range(m)}
        return row_br, col_br

    # -------------------------------------------------------------- dominance
    def strictly_dominant(self, player: int) -> int | None:
        """Index of a strictly dominant strategy for player 0 (row) or 1 (col), if one exists."""
        M = self.A if player == 0 else self.B.T  # rows of M = own strategies
        for s in range(M.shape[0]):
            others = [t for t in range(M.shape[0]) if t != s]
            if all(np.all(M[s] > M[t] + TOL) for t in others):
                return s
        return None

    def iesds(self) -> tuple[list[int], list[int], list[str]]:
        """Iterated elimination of strictly dominated (pure) strategies.

        Returns surviving row indices, surviving column indices and a human-readable log.
        """
        rows = list(range(self.shape[0]))
        cols = list(range(self.shape[1]))
        log = []
        changed = True
        while changed:
            changed = False
            for r in rows[:]:
                for r2 in rows:
                    if r2 != r and all(self.A[r2, c] > self.A[r, c] + TOL for c in cols):
                        rows.remove(r)
                        log.append(f"{self.players[0]}: '{self.row_labels[r]}' is strictly dominated by '{self.row_labels[r2]}'")
                        changed = True
                        break
            for c in cols[:]:
                for c2 in cols:
                    if c2 != c and all(self.B[r, c2] > self.B[r, c] + TOL for r in rows):
                        cols.remove(c)
                        log.append(f"{self.players[1]}: '{self.col_labels[c]}' is strictly dominated by '{self.col_labels[c2]}'")
                        changed = True
                        break
        return rows, cols, log

    # --------------------------------------------------------------- mixed NE
    def mixed_nash(self) -> list[MixedEquilibrium]:
        """All Nash equilibria (pure and mixed) by support enumeration.

        For every pair of equal-size supports (I, J) we solve the indifference
        conditions:  the column mix q makes the row player indifferent across I,
        and the row mix p makes the column player indifferent across J.  A
        candidate is accepted if probabilities are non-negative and no strategy
        outside the support earns strictly more.  Exact for non-degenerate games.
        """
        m, n = self.shape
        found: list[MixedEquilibrium] = []
        for k in range(1, min(m, n) + 1):
            for I in itertools.combinations(range(m), k):
                for J in itertools.combinations(range(n), k):
                    q = self._indifference(self.A[np.ix_(I, J)])        # col mix over J
                    p = self._indifference(self.B[np.ix_(I, J)].T)      # row mix over I
                    if q is None or p is None:
                        continue
                    P = np.zeros(m); P[list(I)] = p
                    Q = np.zeros(n); Q[list(J)] = q
                    row_vals = self.A @ Q
                    col_vals = P @ self.B
                    if row_vals.max() > row_vals[list(I)].min() + 1e-7:
                        continue
                    if col_vals.max() > col_vals[list(J)].min() + 1e-7:
                        continue
                    if any(np.allclose(P, e.row_strategy) and np.allclose(Q, e.col_strategy) for e in found):
                        continue
                    found.append(MixedEquilibrium(P, Q, float(P @ self.A @ Q), float(P @ self.B @ Q)))
        return found

    @staticmethod
    def _indifference(M: np.ndarray) -> np.ndarray | None:
        """Solve M x = u * 1, sum(x) = 1, x >= 0. Returns x or None."""
        k = M.shape[0]
        lhs = np.zeros((k + 1, k + 1))
        lhs[:k, :k] = M
        lhs[:k, k] = -1.0          # -u
        lhs[k, :k] = 1.0
        rhs = np.zeros(k + 1); rhs[k] = 1.0
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return None
        x = sol[:k]
        if np.any(x < -1e-9):
            return None
        return np.clip(x, 0, None) / np.clip(x, 0, None).sum()

    # ---------------------------------------------------------------- welfare
    def pareto_optimal(self) -> list[tuple[int, int]]:
        cells = [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1])]
        opt = []
        for c in cells:
            a, b = self.A[c], self.B[c]
            dominated = any(
                (self.A[d] >= a and self.B[d] >= b) and (self.A[d] > a or self.B[d] > b) for d in cells
            )
            if not dominated:
                opt.append(c)
        return opt

    def joint_payoff(self, cell: tuple[int, int]) -> float:
        return float(self.A[cell] + self.B[cell])

    def describe(self, cell: tuple[int, int]) -> str:
        i, j = cell
        return f"({self.row_labels[i]}, {self.col_labels[j]})"

    def to_table(self, decimals: int = 1) -> list[list[str]]:
        """Payoff table as strings, 'a, b' per cell, for printing."""
        header = [f"{self.players[0]} \\ {self.players[1]}"] + self.col_labels
        rows = [header]
        for i, rl in enumerate(self.row_labels):
            rows.append([rl] + [f"{self.A[i, j]:.{decimals}f}, {self.B[i, j]:.{decimals}f}" for j in range(self.shape[1])])
        return rows
