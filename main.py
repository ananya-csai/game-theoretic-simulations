"""
Game-Theoretic Analysis of Online Advertising Competition
=========================================================
Runs every experiment, prints a readable report to the terminal and saves
figures (outputs/figures), tables (outputs/tables) and all numbers
(outputs/results.json).

    python main.py            # full run (about 30-60 seconds)
    python main.py --quick    # fewer Monte-Carlo trials, for a fast demo
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from adgame import plotting as P
from adgame.ad_spend_game import build_ad_war, classify_2x2, critical_discount_factor, game_type_grid
from adgame.auction_design import (reserve_revenue_sim, reserve_revenue_theory, revenue_equivalence,
                                   shading_payoff_curve, unsold_probability)
from adgame.budget_contest import best_response_dynamics, nash_equilibrium, profits, symmetric_summary
from adgame.dynamics import gfp_sawtooth, gsp_balanced_bidding
from adgame.position_auctions import (best_deviation, eos_equilibrium_bids, gfp, gsp, is_nash, monte_carlo,
                                      vcg)
from adgame.repeated_game import tournament

OUT = Path(__file__).parent / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"


# --------------------------------------------------------------------- helpers
def banner(text: str):
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def table(rows, header=True):
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for k, r in enumerate(rows):
        print("  " + " | ".join(str(x).ljust(w) for x, w in zip(r, widths)))
        if header and k == 0:
            print("  " + "-+-".join("-" * w for w in widths))


def write_csv(name, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# ----------------------------------------------------------------- experiments
def model1(results):
    banner("MODEL 1 — The advertising war (2 firms, Low vs High budget)")
    g = build_ad_war((10, 30), ("Low", "High"), M0=100)
    print("  Market value M0 = ₹100 lakh, Low = ₹10 lakh, High = ₹30 lakh, share = s_i/(s_i+s_j)\n")
    table(g.to_table())
    c = classify_2x2(g)
    ne = g.pure_nash()
    print(f"\n  Pure Nash equilibrium : {[g.describe(e) for e in ne]}  -> profits {g.A[ne[0]]:.0f}, {g.B[ne[0]]:.0f}")
    print(f"  Dominant strategy     : '{g.row_labels[g.strictly_dominant(0)]}' for both firms")
    print(f"  Pareto-optimal cells  : {[g.describe(e) for e in g.pareto_optimal()]}")
    print(f"  Game type             : {c['type']}  (T={c['T']:.0f} > R={c['R']:.0f} > P={c['P']:.0f} > S={c['S']:.0f})")
    _, _, log = g.iesds()
    for line in log:
        print(f"  IESDS                 : {line}")
    loss = g.joint_payoff((0, 0)) - g.joint_payoff(ne[0])
    print(f"  Joint profit lost at equilibrium: ₹{loss:.0f} lakh ({100 * loss / g.joint_payoff((0, 0)):.0f}% of the cooperative outcome)")
    P.payoff_matrix(g, FIG)
    write_csv("model1_payoff_matrix", g.to_table())

    # three-level version: links the discrete game to the continuous Model 2
    g3 = build_ad_war((10, 25, 40), ("Low (10)", "Medium (25)", "High (40)"), M0=100)
    print("\n  Three budget levels (10 / 25 / 40):\n")
    table(g3.to_table())
    ne3 = g3.pure_nash()
    print(f"\n  Nash equilibrium: {[g3.describe(e) for e in ne3]} — the same ₹25 lakh each that the continuous model predicts.")
    print(f"  Mixed-strategy search (support enumeration) finds {len(g3.mixed_nash())} equilibrium in total -> it is unique.")
    P.payoff_matrix(g3, FIG, "fig01b_payoff_matrix_3x3", "Three budget levels: unique equilibrium at Medium")

    high = np.arange(11, 81, 1.0)
    growth = np.round(np.arange(0, 2.001, 0.025), 3)
    grid, names = game_type_grid(high, growth)
    P.game_type_map(grid, names, high, growth, FIG, marker=(30, 0.0))
    shares = {names[k]: float(np.mean(grid == k)) for k in set(grid.flatten())}
    print("\n  Sensitivity (High budget 11–80, market growth 0–2):")
    for k, v in sorted(shares.items(), key=lambda kv: -kv[1]):
        print(f"     {k:<34s} {100 * v:5.1f}% of scenarios")

    banner("MODEL 1b — Repeated ad war (monthly budgets, 200 months)")
    delta = critical_discount_factor(g)
    print(f"  Grim-trigger sustains (Low, Low) if the discount factor δ >= (T-R)/(T-P) = {delta:.2f}")
    t0 = tournament(g, rounds=200, noise=0.0)
    t1 = tournament(g, rounds=200, noise=0.05)
    rows = [["Rank", "Strategy", "Avg profit (no mistakes)", "Avg profit (5% mistakes)", "Low-spend share"]]
    for r, (name, val) in enumerate(t0["ranking"], 1):
        i = t0["names"].index(name)
        rows.append([r, name, f"{val:.2f}", f"{t1['average'][i]:.2f}", f"{100 * t0['low_share'][i]:.0f}%"])
    table(rows)
    write_csv("model1b_tournament", rows)
    P.tournament_bars(t0, t1, FIG)
    results["model1"] = {
        "payoff_A": g.A, "payoff_B": g.B, "nash": ne, "type": c["type"], "T": c["T"], "R": c["R"], "P": c["P"],
        "S": c["S"], "joint_loss": loss, "three_level_payoff": g3.A, "three_level_nash": ne3,
        "game_type_shares": shares, "critical_delta": delta,
        "tournament_clean": t0["ranking"], "tournament_noisy": sorted(zip(t1["names"], t1["average"]), key=lambda x: -x[1]),
    }


def model2(results):
    banner("MODEL 2 — Continuous budgets: Tullock contest among n firms")
    V = 100.0
    costs = [1.0, 1.2, 1.5]
    labels = ["Firm 1 (c=1.0)", "Firm 2 (c=1.2)", "Firm 3 (c=1.5)"]
    ne = nash_equilibrium(V, costs)
    path = best_response_dynamics(V, costs, iterations=15)
    pr = profits(V, costs, ne)
    rows = [["Firm", "Cost per ₹ of ads", "NE budget", "Market share", "Profit"]]
    for k in range(3):
        rows.append([labels[k], costs[k], f"{ne[k]:.2f}", f"{100 * ne[k] / ne.sum():.1f}%", f"{pr[k]:.2f}"])
    table(rows)
    write_csv("model2_asymmetric_ne", rows)
    err = np.abs(path[-1] - ne).max()
    print(f"\n  Best-response dynamics after 15 rounds differ from the closed form by {err:.2e}")
    ne_drop = nash_equilibrium(V, [1.0, 1.5, 3.0])
    print(f"  With costs (1.0, 1.5, 3.0): NE = {np.round(ne_drop, 2)} -> the weakest firm stops advertising")
    P.br_convergence(path, ne, labels, FIG)

    sym = [symmetric_summary(V, n) for n in [2, 3, 4, 5, 6, 8, 10, 15, 20]]
    rows = [["n", "Spend per firm", "Total spend", "Profit per firm", "Value lost to platform"]]
    for s in sym:
        rows.append([s["n"], f"{s['spend_per_firm']:.2f}", f"{s['total_spend']:.2f}", f"{s['profit_per_firm']:.2f}",
                     f"{100 * s['dissipation_ratio']:.0f}%"])
    print()
    table(rows)
    write_csv("model2_dissipation", rows)
    P.dissipation(sym, FIG)
    results["model2"] = {"V": V, "costs": costs, "ne": ne, "profits": pr, "br_error": err, "dropout_ne": ne_drop,
                         "symmetric": sym}


def model3(results, trials):
    banner("MODEL 3 — Keyword (sponsored search) auctions: GSP vs VCG vs GFP")
    v = np.array([12.0, 10.0, 8.0, 6.0, 4.0])
    ctr = np.array([0.30, 0.20, 0.10])
    names = ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E"]
    print("  Values per click (₹): " + ", ".join(f"{n}={x:.0f}" for n, x in zip(names, v)))
    print("  Click-through rates : slot 1 = 30%, slot 2 = 20%, slot 3 = 10%\n")
    rows = [["Mechanism", "Bids", "Slot winners", "Price/click (slots 1-3)", "Revenue", "Welfare"]]
    eos = eos_equilibrium_bids(v, ctr)
    cases = {"VCG (truthful)": (vcg, v), "GSP (truthful bids)": (gsp, v), "GSP (equilibrium bids)": (gsp, eos),
             "GFP (truthful bids)": (gfp, v)}
    example = {}
    for label, (mech, b) in cases.items():
        o = mech(b, v, ctr)
        prices = [o.price_per_click[i] for i in o.order]
        rows.append([label, "[" + ", ".join(f"{x:g}" for x in b) + "]", ", ".join(names[i] for i in o.order),
                     ", ".join(f"{p:.1f}" for p in prices), f"{o.revenue:.2f}", f"{o.welfare:.2f}"])
        if label != "GFP (truthful bids)":
            example[label] = o.revenue
    table(rows)
    write_csv("model3_worked_example", rows)

    ok_vcg, _ = is_nash(vcg, v, v, ctr)
    ok_gsp, devs = is_nash(gsp, v, v, ctr)
    ok_eos, _ = is_nash(gsp, eos, v, ctr)
    d = devs[0]
    print(f"\n  Truthful bidding an equilibrium under VCG?  {ok_vcg}")
    print(f"  Truthful bidding an equilibrium under GSP?  {ok_gsp}")
    print(f"     -> {names[0]} earns {d['current_utility']:.2f} bidding 12, but {d['best_utility']:.2f} by bidding ≈{d['best_bid']:.0f} "
          f"and dropping to slot 2.")
    print(f"  EOS equilibrium bids {np.round(eos, 2).tolist()} form a Nash equilibrium of GSP? {ok_eos}")
    print(f"  Under GFP truthful bids leave every advertiser with zero profit, so they always shade -> no stable bids (Model 4).")

    # classic textbook counter-example (Edelman, Ostrovsky & Schwarz 2007)
    v2 = np.array([10.0, 4.0, 2.0]); c2 = [200, 199]
    d2 = best_deviation(gsp, v2, v2, c2, 0)
    print(f"  EOS textbook example (values 10/4/2, clicks 200/199): truthful profit {d2['current_utility']:.0f} vs "
          f"{d2['best_utility']:.0f} after shading to ≈{d2['best_bid']:.0f}")

    mc = monte_carlo(5, ctr, trials=trials, seed=1)
    rows = [["Mechanism", "Avg revenue", "Std dev", "Avg welfare", "Revenue / welfare"]]
    for k, s in mc.items():
        rows.append([k, f"{s['revenue']:.3f}", f"{s['revenue_sd']:.3f}", f"{s['welfare']:.3f}", f"{100 * s['revenue'] / s['welfare']:.1f}%"])
    print(f"\n  Monte Carlo: {trials:,} auctions, 5 bidders, values ~ U[0, 10]\n")
    table(rows)
    write_csv("model3_monte_carlo", rows)
    P.mechanism_comparison(example, mc, FIG)
    results["model3"] = {"values": v, "ctr": ctr, "eos_bids": eos, "example_revenue": example,
                         "vcg_truthful_is_ne": ok_vcg, "gsp_truthful_is_ne": ok_gsp, "eos_is_ne": ok_eos,
                         "deviation": d, "eos_textbook": d2, "monte_carlo": mc, "trials": trials}


def model4(results):
    banner("MODEL 4 — Bidding dynamics: why the industry moved from GFP to GSP")
    saw = gfp_sawtooth()
    b = saw["bids"]
    drops = int(np.sum(np.diff(b.max(axis=1)) < -1))
    print(f"  GFP (2 advertisers, values 10 and 8): bids climb to ₹{saw['peak_bid']:.1f} then crash to the ₹0.5 reserve.")
    print(f"  Number of crashes in 400 revisions: {drops}  -> bids never settle (no pure-strategy equilibrium)")
    P.sawtooth(saw, FIG)

    v = np.array([12.0, 10.0, 8.0, 6.0, 4.0]); ctr = np.array([0.30, 0.20, 0.10])
    runs = []
    for seed in range(20):
        r = gsp_balanced_bidding(v, ctr, rounds=300, seed=seed)
        conv = int(np.argmax(np.all(np.abs(r["bids"] - r["bids"][-1]) < 1e-9, axis=1)))
        runs.append({"seed": seed, "final_revenue": r["final_revenue"], "converged_at": conv,
                     "matches_eos": bool(np.allclose(r["final_bids"], eos_equilibrium_bids(v, ctr)))})
    r = gsp_balanced_bidding(v, ctr, rounds=300, seed=3)
    P.balanced_bidding(r, eos_equilibrium_bids(v, ctr), [f"Brand {c}" for c in "ABCDE"], FIG)
    n_ok = sum(x["matches_eos"] for x in runs)
    print(f"  GSP balanced bidding: {n_ok}/20 random starting points converge to the EOS equilibrium bids")
    print(f"  Median updates to converge: {int(np.median([x['converged_at'] for x in runs]))}; final revenue = "
          f"₹{r['final_revenue']:.2f} = VCG revenue ₹{r['vcg_revenue']:.2f}")
    results["model4"] = {"gfp_peak": saw["peak_bid"], "gfp_crashes": drops, "gfp_mean_revenue": saw["mean_revenue"],
                         "gsp_runs": runs, "gsp_converged": n_ok, "vcg_revenue": r["vcg_revenue"]}


def model5(results, trials):
    banner("MODEL 5 — Display-ad auctions (real-time bidding) and platform design")
    factors = np.round(np.linspace(0.3, 1.0, 71), 3)
    curves = {n: shading_payoff_curve(n, factors, trials=trials) for n in (2, 3, 5)}
    rows = [["Bidders n", "Best shading (simulated)", "Theory (n-1)/n"]]
    for n, ys in curves.items():
        rows.append([n, f"{factors[int(np.argmax(ys))]:.2f}", f"{(n - 1) / n:.2f}"])
    table(rows)
    write_csv("model5_shading", rows)
    P.shading(curves, factors, FIG)

    re_ = revenue_equivalence([2, 3, 4, 5, 7, 10], trials=trials)
    rows = [["n", "First-price revenue", "Second-price revenue", "Theory (n-1)/(n+1)"]]
    for r in re_["rows"]:
        rows.append([r["n"], f"{r['first_price']:.4f}", f"{r['second_price']:.4f}", f"{r['theory']:.4f}"])
    print()
    table(rows)
    write_csv("model5_revenue_equivalence", rows)
    P.revenue_equiv(re_["rows"], FIG)

    n = 2
    r = np.linspace(0, 1, 201)
    th = reserve_revenue_theory(n, r)
    sim_r = np.round(np.arange(0, 1.01, 0.1), 2)
    sim = reserve_revenue_sim(n, sim_r, trials=trials)
    k = int(np.argmax(th))
    print(f"\n  Optimal reserve (n = 2): r* = {r[k]:.2f}; revenue {th[0]:.4f} -> {th[k]:.4f} "
          f"(+{100 * (th[k] / th[0] - 1):.1f}%), impression unsold {100 * unsold_probability(n, r[k]):.0f}% of the time")
    P.reserve_curve(r, th, sim_r, sim, n, FIG)
    results["model5"] = {"shading": {n: float(factors[int(np.argmax(ys))]) for n, ys in curves.items()},
                         "revenue_equivalence": re_["rows"], "reserve_opt": float(r[k]),
                         "reserve_rev0": float(th[0]), "reserve_rev_opt": float(th[k]),
                         "unsold": unsold_probability(n, r[k])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer Monte-Carlo trials")
    args = ap.parse_args()
    trials_auction = 2_000 if args.quick else 10_000
    trials_bayes = 40_000 if args.quick else 200_000
    t0 = time.time()
    print("\nGAME-THEORETIC ANALYSIS OF ONLINE ADVERTISING COMPETITION")
    print("Manipal University Jaipur — Game Theory course project")
    results = {}
    model1(results)
    model2(results)
    model3(results, trials_auction)
    model4(results)
    model5(results, trials_bayes)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)
    banner("DONE")
    print("  Figures : outputs/figures/")
    print("  Tables  : outputs/tables/")
    print("  Numbers : outputs/results.json")
    print(f"  Time    : {time.time() - t0:.1f} s\n")


if __name__ == "__main__":
    main()
