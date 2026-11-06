"""
auction_design.py  —  Model 5: display-ad auctions (real-time bidding) and
the platform's design choices
--------------------------------------------------------------------------
Every time a web page loads, one display-ad impression is sold in a
milliseconds-long auction. Bidders do not know rivals' values (incomplete
information), so we use Bayesian Nash equilibrium with values v ~ U[0, 1].

Questions answered by simulation and checked against theory:
1. Bid shading. In a first-price auction the equilibrium bid is
   b(v) = (n-1)/n * v.  We verify that this shading factor maximises a bidder's
   expected profit when rivals use it. (Google Ad Manager moved to first-price
   in 2019, which made bid shading a core ad-tech product.)
2. Revenue equivalence. First-price and second-price auctions earn the same
   expected revenue, (n-1)/(n+1).
3. Reserve price. A seller-optimal reserve r* = 0.5 (Myerson 1981) raises
   revenue even though it sometimes leaves the impression unsold.
"""
from __future__ import annotations

import numpy as np


def fpa_equilibrium_bid(v, n: int):
    return (n - 1) / n * np.asarray(v, float)


def shading_payoff_curve(n: int, factors, trials: int = 200_000, seed: int = 11) -> np.ndarray:
    """Expected profit of one bidder who bids k*v while the other n-1 use the equilibrium."""
    rng = np.random.default_rng(seed)
    v_me = rng.uniform(0, 1, trials)
    rivals = rng.uniform(0, 1, (trials, n - 1))
    best_rival = fpa_equilibrium_bid(rivals, n).max(axis=1)
    out = []
    for k in factors:
        b = k * v_me
        win = b > best_rival
        out.append(float(np.mean(win * (v_me - b))))
    return np.array(out)


def theoretical_shading_payoff(n: int, k):
    """Closed form of the curve above: E[(v - k v) P(win)] with P(win) = min(1, k v n/(n-1))^(n-1)."""
    k = np.asarray(k, float)
    res = []
    vs = np.linspace(0, 1, 4001)
    for kk in np.atleast_1d(k):
        pw = np.minimum(1.0, kk * vs * n / (n - 1)) ** (n - 1)
        res.append(np.trapezoid((1 - kk) * vs * pw, vs))
    return np.array(res)


def revenue_equivalence(n_values, trials: int = 200_000, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        v = rng.uniform(0, 1, (trials, n))
        s = np.sort(v, axis=1)
        spa = s[:, -2].mean()                         # second-highest value
        fpa = fpa_equilibrium_bid(s[:, -1], n).mean()  # highest equilibrium bid
        rows.append({"n": int(n), "first_price": float(fpa), "second_price": float(spa),
                     "theory": (n - 1) / (n + 1)})
    return {"rows": rows}


def reserve_revenue_theory(n: int, r):
    """Expected revenue of a second-price auction with reserve r, values U[0,1]:
    R(r) = 2n(1 - r^(n+1))/(n+1) - (1 - r^n)."""
    r = np.asarray(r, float)
    return 2 * n * (1 - r ** (n + 1)) / (n + 1) - (1 - r ** n)


def reserve_revenue_sim(n: int, reserves, trials: int = 200_000, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.sort(rng.uniform(0, 1, (trials, n)), axis=1)
    hi, second = v[:, -1], v[:, -2] if n > 1 else np.zeros(trials)
    out = []
    for r in reserves:
        price = np.where(hi >= r, np.maximum(second, r), 0.0)
        out.append(float(price.mean()))
    return np.array(out)


def unsold_probability(n: int, r: float) -> float:
    return float(r ** n)
