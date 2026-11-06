"""
dynamics.py  —  Model 4: how advertisers actually adjust bids over time
-----------------------------------------------------------------------
Equilibrium tells us where bidding *could* settle. Dynamics tell us whether it
*does*. Advertisers (or their automated bidding tools) repeatedly look at
rivals' bids and re-optimise.

1. GFP (pay-your-bid) best-response dynamics -> no stable point. Bids climb in
   small steps, collapse, and climb again: the "saw-tooth" bidding war observed
   on Overture's first-price keyword auctions (Edelman & Ostrovsky, 2007). This
   instability is one reason the industry switched to GSP.

2. GSP "balanced bidding" (Cary et al., 2007) -> each advertiser targets the slot
   that maximises its profit and bids so it is indifferent between that slot and
   the one above. With random, one-at-a-time updates this converges to the
   lowest envy-free equilibrium, i.e. the VCG prices.
"""
from __future__ import annotations

import numpy as np

from .position_auctions import gsp, vcg


def gfp_sawtooth(values=(10.0, 8.0), ctrs=(1.0, 0.6), reserve: float = 0.5, step: float = 0.1,
                 rounds: int = 400) -> dict:
    """Two advertisers, two slots, first-price. Bidders alternate best responses.

    Options for the bidder who moves:
      * outbid the rival by one `step` and take slot 1, paying its own bid, or
      * drop to the reserve and take slot 2 cheaply.
    """
    v = np.asarray(values, float); a = np.asarray(ctrs, float)
    bids = np.array([reserve, reserve])
    history = [bids.copy()]
    revenue = []
    for t in range(rounds):
        i = t % 2; j = 1 - i
        top_bid = round(bids[j] + step, 10)
        u_top = a[0] * (v[i] - top_bid) if top_bid <= v[i] else -np.inf
        u_bottom = a[1] * (v[i] - reserve)
        bids[i] = top_bid if u_top > u_bottom else reserve
        history.append(bids.copy())
        hi, lo = max(bids), min(bids)
        revenue.append(a[0] * hi + a[1] * lo)
    history = np.array(history)
    return {"bids": history, "revenue": np.array(revenue), "values": v, "ctrs": a,
            "peak_bid": float(history.max()), "mean_revenue": float(np.mean(revenue))}


def balanced_bid(i: int, bids, values, ctrs) -> float:
    """Cary et al. (2007) balanced-bidding rule for bidder i in a GSP auction."""
    v = values[i]
    a = np.asarray(ctrs, float)
    K = len(a)
    others = np.sort(np.delete(np.asarray(bids, float), i))[::-1]
    others = np.append(others, np.zeros(max(0, K + 1 - len(others))))
    utils = np.array([a[s] * (v - others[s]) for s in range(K)])
    s_star = int(np.argmax(utils))
    if utils[s_star] <= 0:
        return float(v)
    if s_star == 0:
        return float(v)
    return float(v - (a[s_star] / a[s_star - 1]) * (v - others[s_star]))


def gsp_balanced_bidding(values, ctrs, rounds: int = 300, seed: int = 3, start=None) -> dict:
    """Random asynchronous balanced-bidding updates. Returns bid paths and revenue path."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    n = len(values)
    bids = rng.uniform(0, values) if start is None else np.asarray(start, float).copy()
    path = [bids.copy()]
    rev = [gsp(bids, values, ctrs).revenue]
    for _ in range(rounds):
        i = int(rng.integers(0, n))
        bids[i] = balanced_bid(i, bids, values, ctrs)
        path.append(bids.copy())
        rev.append(gsp(bids, values, ctrs).revenue)
    vcg_rev = vcg(values, values, ctrs).revenue
    return {"bids": np.array(path), "revenue": np.array(rev), "vcg_revenue": float(vcg_rev),
            "final_bids": bids.copy(), "final_revenue": float(rev[-1])}
