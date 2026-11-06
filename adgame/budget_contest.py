"""
budget_contest.py  —  Model 2: continuous advertising budgets among n firms
---------------------------------------------------------------------------
The discrete ad war (Model 1) only allows Low / High. Here each firm i picks any
budget x_i >= 0. Customers are won in proportion to spend (Tullock contest):

    π_i(x) = V * x_i / (x_1 + ... + x_n)  -  c_i * x_i

V   = value of the customer pool being fought over (e.g. ₹ lakh of profit)
c_i = firm i's effective cost per unit of advertising (a firm with a weak
      product or poor conversion needs to spend more per customer: higher c_i).

Results proved and computed here:
* Closed-form Nash equilibrium with asymmetric costs (including which firms
  drop out of advertising entirely).
* Best-response dynamics converge to that equilibrium.
* "Rent dissipation": with n identical firms total spend is (n-1)/n * V, so as
  competition grows almost all the value is handed to the ad platform.
"""
from __future__ import annotations

import numpy as np


def profits(V: float, costs, x) -> np.ndarray:
    costs = np.asarray(costs, float); x = np.asarray(x, float)
    X = x.sum()
    share = x / X if X > 0 else np.full_like(x, 1.0 / len(x))
    return V * share - costs * x


def best_response(V: float, others_total: float, c: float) -> float:
    """argmax_x  V x/(x + X_-i) - c x   =  max(0, sqrt(V X_-i / c) - X_-i).

    If nobody else advertises, any tiny positive spend wins everything; we
    return a small positive amount so dynamics can continue.
    """
    if others_total <= 0:
        return 1e-6 * V
    return max(0.0, float(np.sqrt(V * others_total / c) - others_total))


def nash_equilibrium(V: float, costs) -> np.ndarray:
    """Exact equilibrium budgets for the asymmetric Tullock contest.

    For the set S of active firms (|S| = m >= 2):
        X* = (m - 1) V / sum_{j in S} c_j
        x_i* = X* (1 - (m - 1) c_i / sum_{j in S} c_j)
    Firms are added cheapest-first; a firm is active only if its x_i* > 0.
    """
    costs = np.asarray(costs, float)
    n = len(costs)
    order = np.argsort(costs)
    active = list(order[:2])
    for idx in order[2:]:
        trial = active + [idx]
        C = costs[trial].sum(); m = len(trial)
        if 1 - (m - 1) * costs[idx] / C > 1e-12:
            active = trial
        else:
            break
    C = costs[active].sum(); m = len(active)
    X = (m - 1) * V / C
    x = np.zeros(n)
    for i in active:
        x[i] = X * (1 - (m - 1) * costs[i] / C)
    return x


def best_response_dynamics(V: float, costs, x0=None, iterations: int = 60):
    """Firms take turns (Gauss-Seidel) choosing the best reply to rivals' current budgets."""
    costs = np.asarray(costs, float)
    n = len(costs)
    x = np.full(n, 0.05 * V) if x0 is None else np.asarray(x0, float).copy()
    path = [x.copy()]
    for _ in range(iterations):
        for i in range(n):
            x[i] = best_response(V, x.sum() - x[i], costs[i])
        path.append(x.copy())
    return np.array(path)


def symmetric_summary(V: float, n: int, c: float = 1.0) -> dict:
    """Closed-form symmetric results."""
    x = (n - 1) * V / (n ** 2 * c)
    total = n * x
    return {
        "n": n,
        "spend_per_firm": x,
        "total_spend": total,
        "profit_per_firm": V / n - c * x,
        "industry_profit": V - c * total,
        "dissipation_ratio": c * total / V,   # share of the prize handed to the platform
    }


def collusive_benchmark(V: float, costs, floor: float = 1e-3) -> dict:
    """If firms could agree, they would each spend only a token amount (shares unchanged,
    costs ~0). We use a tiny equal floor spend so shares are defined."""
    costs = np.asarray(costs, float)
    x = np.full(len(costs), floor * V)
    return {"spend": x, "profits": profits(V, costs, x)}
