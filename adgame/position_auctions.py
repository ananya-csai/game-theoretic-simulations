"""
position_auctions.py  —  Model 3: sponsored-search (keyword) ad auctions
------------------------------------------------------------------------
When a user searches a keyword, the platform shows K ad slots. Higher slots get
more clicks. Advertiser i values a click at v_i and submits a bid b_i per click.

Click model:   clicks for bidder i in slot k = ctr[k] * quality[i]
Ranking:       by bid x quality ("ad rank", as used by Google Ads)

Mechanisms implemented
* GSP  (Generalized Second Price) – pay the minimum needed to keep your slot:
        price_k = b_(k+1) * q_(k+1) / q_(k)       (Google, Bing)
* VCG  (Vickrey–Clarke–Groves) – pay the externality you impose on others
        (truthful bidding is a dominant strategy; used by Facebook/Meta ads)
* GFP  (Generalized First Price) – pay your own bid (Overture, 1997–2002)

Also:
* `eos_equilibrium_bids` – the Edelman–Ostrovsky–Schwarz / Varian "lowest
  envy-free" equilibrium of GSP, which gives exactly the VCG outcome.
* `best_deviation` / `is_nash` – brute-force check of whether any advertiser
  could gain by changing its bid (used to prove truthful GSP is NOT stable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AuctionOutcome:
    mechanism: str
    order: list[int]            # bidder index in slot 1, 2, ... (only winners)
    slot: np.ndarray            # slot (0-based) of each bidder, -1 if no slot
    clicks: np.ndarray
    price_per_click: np.ndarray
    payment: np.ndarray
    utility: np.ndarray
    revenue: float
    welfare: float              # total value created = sum clicks * value


def _prep(bids, values, ctrs, quality):
    bids = np.asarray(bids, float)
    values = np.asarray(values, float)
    ctrs = np.asarray(ctrs, float)
    quality = np.ones_like(bids) if quality is None else np.asarray(quality, float)
    return bids, values, ctrs, quality


def rank(bids, quality, K: int, reserve: float = 0.0) -> list[int]:
    """Sort by ad-rank (bid * quality), ties broken by lower index; only bids >= reserve."""
    scores = bids * quality
    eligible = [i for i in range(len(bids)) if bids[i] >= reserve and bids[i] > 0]
    eligible.sort(key=lambda i: (-scores[i], i))
    return eligible[:K]


def _finish(name, order, bids, values, ctrs, quality, ppc):
    n = len(bids)
    slot = -np.ones(n, dtype=int)
    clicks = np.zeros(n)
    for k, i in enumerate(order):
        slot[i] = k
        clicks[i] = ctrs[k] * quality[i]
    payment = clicks * ppc
    utility = clicks * values - payment
    return AuctionOutcome(name, order, slot, clicks, ppc, payment, utility,
                          float(payment.sum()), float((clicks * values).sum()))


def gsp(bids, values, ctrs, quality=None, reserve: float = 0.0) -> AuctionOutcome:
    bids, values, ctrs, quality = _prep(bids, values, ctrs, quality)
    K = len(ctrs)
    full = rank(bids, quality, len(bids), reserve)
    order = full[:K]
    ppc = np.zeros(len(bids))
    for k, i in enumerate(order):
        if k + 1 < len(full):
            j = full[k + 1]
            ppc[i] = max(reserve, bids[j] * quality[j] / quality[i])
        else:
            ppc[i] = reserve
    return _finish("GSP", order, bids, values, ctrs, quality, ppc)


def gfp(bids, values, ctrs, quality=None, reserve: float = 0.0) -> AuctionOutcome:
    bids, values, ctrs, quality = _prep(bids, values, ctrs, quality)
    order = rank(bids, quality, len(ctrs), reserve)
    ppc = np.zeros(len(bids))
    for i in order:
        ppc[i] = bids[i]
    return _finish("GFP", order, bids, values, ctrs, quality, ppc)


def _welfare_of(order, reported, ctrs, quality):
    return sum(ctrs[k] * quality[i] * reported[i] for k, i in enumerate(order))


def vcg(bids, values, ctrs, quality=None, reserve: float = 0.0) -> AuctionOutcome:
    """Each winner pays (welfare of others without it) - (welfare of others with it),
    computed on reported bids. Allocation maximises reported welfare."""
    bids, values, ctrs, quality = _prep(bids, values, ctrs, quality)
    K = len(ctrs)
    order = rank(bids, quality, K, reserve)
    ppc = np.zeros(len(bids))
    W_all = _welfare_of(order, bids, ctrs, quality)
    for k, i in enumerate(order):
        others_with = W_all - ctrs[k] * quality[i] * bids[i]
        b2 = bids.copy(); b2[i] = 0.0
        order_wo = rank(b2, quality, K, reserve)
        others_without = _welfare_of(order_wo, b2, ctrs, quality)
        pay = others_without - others_with
        clicks = ctrs[k] * quality[i]
        ppc[i] = max(reserve, pay / clicks) if clicks > 0 else 0.0
    return _finish("VCG", order, bids, values, ctrs, quality, ppc)


MECHANISMS = {"GSP": gsp, "VCG": vcg, "GFP": gfp}


# ------------------------------------------------------------------ equilibria
def vcg_prices_per_click(values, ctrs) -> np.ndarray:
    """VCG per-click price for each slot when bidders bid truthfully (quality = 1).
    p_k * a_k = (a_k - a_{k+1}) v_(k+1) + p_{k+1} a_{k+1}."""
    v = np.sort(np.asarray(values, float))[::-1]
    a = np.append(np.asarray(ctrs, float), 0.0)
    K = len(ctrs)
    v = np.append(v, np.zeros(max(0, K + 1 - len(v))))
    p = np.zeros(K + 1)
    for k in range(K - 1, -1, -1):
        p[k] = ((a[k] - a[k + 1]) * v[k + 1] + p[k + 1] * a[k + 1]) / a[k]
    return p[:K]


def eos_equilibrium_bids(values, ctrs) -> np.ndarray:
    """Lowest envy-free (symmetric) Nash equilibrium of GSP (Edelman, Ostrovsky &
    Schwarz 2007; Varian 2007). The bidder in slot k+1 bids the VCG price of slot k,
    so every advertiser pays exactly its VCG price. Losers bid their value; the top
    bidder bids its value (any bid above the 2nd works)."""
    values = np.asarray(values, float)
    order = np.argsort(-values, kind="stable")
    p = vcg_prices_per_click(values, ctrs)
    K = len(ctrs)
    bids = values.copy()
    for pos in range(1, min(K + 1, len(values))):
        bids[order[pos]] = p[pos - 1]
    return bids


def best_deviation(mechanism, bids, values, ctrs, i: int, quality=None, eps: float = 1e-6):
    """Best utility advertiser i can get by changing only its own bid.

    Because outcomes depend only on the ranking, it is enough to try bids just
    above / below every rival's ad-rank, plus 0 and the bidder's value.
    """
    bids = np.asarray(bids, float)
    q = np.ones_like(bids) if quality is None else np.asarray(quality, float)
    base = mechanism(bids, values, ctrs, quality).utility[i]
    cands = {0.0, float(values[i]), float(bids[i])}
    for j in range(len(bids)):
        if j != i:
            t = bids[j] * q[j] / q[i]
            cands.update({max(0.0, t - eps), t + eps})
    best_u, best_b = base, float(bids[i])
    for c in sorted(cands):
        b2 = bids.copy(); b2[i] = c
        u = mechanism(b2, values, ctrs, quality).utility[i]
        if u > best_u + 1e-7:
            best_u, best_b = u, c
    return {"bidder": i, "current_utility": float(base), "best_utility": float(best_u),
            "best_bid": best_b, "gain": float(best_u - base)}


def is_nash(mechanism, bids, values, ctrs, quality=None) -> tuple[bool, list[dict]]:
    devs = [best_deviation(mechanism, bids, values, ctrs, i, quality) for i in range(len(bids))]
    return all(d["gain"] <= 1e-6 for d in devs), devs


# ---------------------------------------------------------------- simulation
def monte_carlo(n_bidders: int, ctrs, trials: int = 5000, seed: int = 1, value_high: float = 10.0) -> dict:
    """Average revenue and welfare over random keyword auctions (values ~ U[0, value_high])."""
    rng = np.random.default_rng(seed)
    ctrs = np.asarray(ctrs, float)
    rec = {k: {"revenue": [], "welfare": []} for k in
           ["VCG (truthful)", "GSP (truthful bids)", "GSP (equilibrium bids)"]}
    for _ in range(trials):
        v = rng.uniform(0, value_high, n_bidders)
        outs = {
            "VCG (truthful)": vcg(v, v, ctrs),
            "GSP (truthful bids)": gsp(v, v, ctrs),
            "GSP (equilibrium bids)": gsp(eos_equilibrium_bids(v, ctrs), v, ctrs),
        }
        for k, o in outs.items():
            rec[k]["revenue"].append(o.revenue)
            rec[k]["welfare"].append(o.welfare)
    return {k: {"revenue": float(np.mean(d["revenue"])), "revenue_sd": float(np.std(d["revenue"])),
                "welfare": float(np.mean(d["welfare"]))} for k, d in rec.items()}
