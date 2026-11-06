"""
Interactive dashboard for the live demo.

    streamlit run app.py

Every slider re-runs the game-theory model instantly, so the class can ask
"what if?" during the presentation.
"""
import numpy as np
import pandas as pd
import streamlit as st

from adgame.ad_spend_game import build_ad_war, classify_2x2, critical_discount_factor
from adgame.auction_design import (reserve_revenue_theory, revenue_equivalence, shading_payoff_curve,
                                   unsold_probability)
from adgame.budget_contest import best_response_dynamics, nash_equilibrium, profits, symmetric_summary
from adgame.dynamics import gfp_sawtooth, gsp_balanced_bidding
from adgame.position_auctions import best_deviation, eos_equilibrium_bids, gfp, gsp, is_nash, vcg
from adgame.repeated_game import ALL_STRATEGIES, play_match, tournament

st.set_page_config(page_title="Online Ad Competition — Game Theory", page_icon="📈", layout="wide")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]

PAGES = ["Overview", "1 · Advertising war", "2 · Repeated game", "3 · Budget contest",
         "4 · Keyword auctions", "5 · Bidding dynamics", "6 · Display auctions"]
page = st.sidebar.radio("Model", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Game-Theoretic Analysis of Online Advertising Competition\n\nManipal University Jaipur · Game Theory")


def payoff_df(g):
    return pd.DataFrame([[f"{g.A[i, j]:.1f} , {g.B[i, j]:.1f}" for j in range(g.shape[1])] for i in range(g.shape[0])],
                        index=[f"A: {r}" for r in g.row_labels], columns=[f"B: {c}" for c in g.col_labels])


# ------------------------------------------------------------------ overview
if page == PAGES[0]:
    st.title("Game-Theoretic Analysis of Online Advertising Competition")
    st.markdown(
        "Online advertising is a market where **every firm's best move depends on what its rivals do** — "
        "exactly the situation game theory was built for. This dashboard runs five models:")
    c = st.columns(3)
    c[0].markdown("**1 · Advertising war**  \nTwo firms choose Low or High budgets → Prisoner's Dilemma.")
    c[1].markdown("**2 · Repeated game**  \nMonthly budgets: can firms keep spending low without a contract?")
    c[2].markdown("**3 · Budget contest**  \nContinuous budgets for *n* firms (Tullock contest).")
    c = st.columns(3)
    c[0].markdown("**4 · Keyword auctions**  \nGSP vs VCG vs GFP: who pays what for each ad slot.")
    c[1].markdown("**5 · Bidding dynamics**  \nWhy first-price bidding wars never settle.")
    c[2].markdown("**6 · Display auctions**  \nBid shading, revenue equivalence, reserve prices.")
    st.info("Pick a model from the sidebar. Every slider re-solves the game in real time.")

# ------------------------------------------------------------- 1 ad war
elif page == PAGES[1]:
    st.header("1 · The advertising war")
    st.caption("profit = market × share − spend,   market = M0 + g·(spend_A + spend_B),   share = s_A^γ / (s_A^γ + s_B^γ)")
    c1, c2, c3, c4, c5 = st.columns(5)
    M0 = c1.slider("Market value M0 (₹ lakh)", 20, 300, 100, 10)
    low = c2.slider("Low budget", 1, 50, 10)
    high = c3.slider("High budget", low + 1, 150, 30)
    growth = c4.slider("Market growth g", 0.0, 2.0, 0.0, 0.05)
    gamma = c5.slider("Share sensitivity γ", 0.2, 3.0, 1.0, 0.1)
    g = build_ad_war((low, high), ("Low", "High"), M0, growth, gamma)
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Payoff matrix (Firm A , Firm B)")
        st.table(payoff_df(g))
    with right:
        c = classify_2x2(g)
        ne = g.pure_nash()
        st.subheader("Solution")
        st.metric("Game type", c["type"])
        st.write("**Pure Nash equilibria:** " + (", ".join(g.describe(e) for e in ne) or "none"))
        for e in g.mixed_nash():
            if not e.is_pure():
                st.write(f"**Mixed NE:** each firm plays High with probability {e.row_strategy[1]:.2f}")
        dom = g.strictly_dominant(0)
        st.write("**Dominant strategy:** " + (g.row_labels[dom] if dom is not None else "none"))
        best = max(g.pareto_optimal(), key=g.joint_payoff)
        st.write(f"**Best joint outcome:** {g.describe(best)} = {g.joint_payoff(best):.1f}")
        if ne:
            st.write(f"**Joint profit at NE:** {g.joint_payoff(ne[0]):.1f}")
        d = critical_discount_factor(g)
        if d is not None:
            st.success(f"In the repeated game, cooperation (Low, Low) survives if firms value next month at least δ ≥ {d:.2f}")

# ------------------------------------------------------------- 2 repeated
elif page == PAGES[2]:
    st.header("2 · The repeated advertising war")
    g = build_ad_war()
    c1, c2 = st.columns(2)
    rounds = c1.slider("Months", 10, 500, 200, 10)
    noise = c2.slider("Execution mistakes (probability a move is flipped)", 0.0, 0.3, 0.0, 0.01)
    t = tournament(g, rounds=rounds, noise=noise)
    df = pd.DataFrame({"Strategy": t["names"], "Average profit": t["average"], "Low-spend share": t["low_share"]})
    df = df.sort_values("Average profit", ascending=False).reset_index(drop=True)
    left, right = st.columns([1, 1.2])
    left.dataframe(df.style.format({"Average profit": "{:.2f}", "Low-spend share": "{:.0%}"}), width="stretch")
    right.bar_chart(df, x="Strategy", y="Average profit", horizontal=True, sort="-Average profit", color=C[0])
    st.subheader("Head-to-head match")
    names = [s.name for s in ALL_STRATEGIES]
    a, b = st.columns(2)
    s1 = a.selectbox("Firm A strategy", names, index=names.index("Tit-for-Tat"))
    s2 = b.selectbox("Firm B strategy", names, index=names.index("Always High"))
    p1, p2, h1, h2 = play_match(g, ALL_STRATEGIES[names.index(s1)](), ALL_STRATEGIES[names.index(s2)](), 40, noise,
                                np.random.default_rng(1))
    st.write(f"Average profit over 40 months: **A = {p1:.1f}**, **B = {p2:.1f}**")
    st.dataframe(pd.DataFrame([["H" if x else "L" for x in h1], ["H" if x else "L" for x in h2]],
                              index=["Firm A", "Firm B"], columns=[str(i + 1) for i in range(40)]), width="stretch")

# ------------------------------------------------------------- 3 contest
elif page == PAGES[3]:
    st.header("3 · Continuous budgets: the Tullock contest")
    st.caption("profit_i = V · x_i / Σx − c_i · x_i")
    c1, c2 = st.columns(2)
    V = c1.slider("Value of the customer pool V (₹ lakh)", 10, 500, 100, 10)
    n = c2.slider("Number of firms", 2, 6, 3)
    cols = st.columns(n)
    costs = [cols[i].number_input(f"Cost c{i + 1}", 0.2, 5.0, [1.0, 1.2, 1.5, 2.0, 2.5, 3.0][i], 0.1) for i in range(n)]
    ne = nash_equilibrium(V, costs)
    pr = profits(V, costs, ne)
    st.dataframe(pd.DataFrame({"Firm": [f"Firm {i + 1}" for i in range(n)], "Cost": costs, "NE budget": ne,
                               "Share": ne / ne.sum(), "Profit": pr})
                 .style.format({"NE budget": "{:.2f}", "Share": "{:.1%}", "Profit": "{:.2f}", "Cost": "{:.1f}"}),
                 width="stretch")
    m = st.columns(3)
    m[0].metric("Total ad spend", f"₹{ne.sum():.1f} lakh")
    m[1].metric("Share of V paid to platform", f"{100 * np.dot(costs, ne) / V:.0f}%")
    m[2].metric("Firms that stop advertising", int(np.sum(ne == 0)))
    path = best_response_dynamics(V, costs, iterations=15)
    st.subheader("Best-response dynamics")
    st.line_chart(pd.DataFrame(path, columns=[f"Firm {i + 1}" for i in range(n)]), color=C[:n], x_label="Round", y_label="Budget")
    st.subheader("Rent dissipation with identical firms")
    sym = pd.DataFrame([symmetric_summary(V, k) for k in range(2, 21)]).set_index("n")
    st.line_chart(sym[["total_spend", "industry_profit"]].rename(columns={"total_spend": "Total ad spend",
                                                                          "industry_profit": "Industry profit"}), color=C[:2], x_label="Number of firms", y_label="₹ lakh")

# ------------------------------------------------------------- 4 auctions
elif page == PAGES[4]:
    st.header("4 · Keyword auctions: GSP vs VCG vs GFP")
    c1, c2 = st.columns(2)
    vals = c1.text_input("Advertiser values per click (₹)", "12, 10, 8, 6, 4")
    ctrs = c2.text_input("Click-through rate of each slot", "0.30, 0.20, 0.10")
    qual = st.text_input("Quality scores (optional, same length as values)", "")
    v = np.array([float(x) for x in vals.split(",")])
    a = np.array([float(x) for x in ctrs.split(",")])
    q = np.array([float(x) for x in qual.split(",")]) if qual.strip() else None
    names = [f"Brand {chr(65 + i)}" for i in range(len(v))]
    eos = eos_equilibrium_bids(v, a) if q is None else None
    cases = [("VCG — truthful", vcg, v), ("GSP — truthful bids", gsp, v)]
    if eos is not None:
        cases.append(("GSP — equilibrium bids", gsp, eos))
    cases.append(("GFP — truthful bids", gfp, v))
    rows = []
    for label, mech, b in cases:
        o = mech(b, v, a, q)
        ok, devs = is_nash(mech, b, v, a, q)
        rows.append({"Mechanism": label, "Winners (slot order)": ", ".join(names[i] for i in o.order),
                     "Price per click": ", ".join(f"{o.price_per_click[i]:.2f}" for i in o.order),
                     "Revenue": o.revenue, "Welfare": o.welfare, "Stable (Nash)?": "Yes" if ok else "No"})
    st.dataframe(pd.DataFrame(rows).style.format({"Revenue": "{:.3f}", "Welfare": "{:.3f}"}), width="stretch")
    if eos is not None:
        st.write("**Equilibrium bids (EOS / Varian):** " + ", ".join(f"{n}={b:.2f}" for n, b in zip(names, eos)))
    st.subheader("Can anyone gain by changing their bid under truthful GSP?")
    for i in range(len(v)):
        d = best_deviation(gsp, v, v, a, i, q)
        if d["gain"] > 1e-6:
            st.warning(f"{names[i]}: profit {d['current_utility']:.3f} → {d['best_utility']:.3f} by bidding ≈ {d['best_bid']:.2f}")

# ------------------------------------------------------------- 5 dynamics
elif page == PAGES[5]:
    st.header("5 · Bidding dynamics")
    st.subheader("First-price (GFP): the saw-tooth bidding war")
    c = st.columns(4)
    va = c[0].slider("Value A", 2.0, 20.0, 10.0, 0.5)
    vb = c[1].slider("Value B", 2.0, 20.0, 8.0, 0.5)
    ratio = c[2].slider("Slot-2 / slot-1 clicks", 0.1, 0.95, 0.6, 0.05)
    step = c[3].slider("Bid increment", 0.05, 1.0, 0.1, 0.05)
    res = gfp_sawtooth((va, vb), (1.0, ratio), 0.5, step, 300)
    st.line_chart(pd.DataFrame(res["bids"], columns=["Advertiser A", "Advertiser B"]), color=C[:2], x_label="Bid revision", y_label="Bid per click (₹)")
    st.subheader("GSP with balanced bidding: converges to the VCG outcome")
    seed = st.slider("Random starting point (seed)", 0, 50, 3)
    v = np.array([12.0, 10.0, 8.0, 6.0, 4.0]); a = np.array([0.3, 0.2, 0.1])
    r = gsp_balanced_bidding(v, a, rounds=120, seed=seed)
    left, right = st.columns(2)
    left.line_chart(pd.DataFrame(r["bids"], columns=[f"Brand {x}" for x in "ABCDE"]), color=C[:5], x_label="Update", y_label="Bid per click (₹)")
    right.line_chart(pd.DataFrame({"GSP revenue": r["revenue"], "VCG revenue": r["vcg_revenue"]}), color=[C[5], "#898781"], x_label="Update", y_label="Revenue per search (₹)")

# ------------------------------------------------------------- 6 display
elif page == PAGES[6]:
    st.header("6 · Display-ad auctions (real-time bidding)")
    n = st.slider("Number of bidders", 2, 10, 2)
    k = np.round(np.linspace(0.3, 1.0, 71), 3)
    curve = shading_payoff_curve(n, k, trials=60_000)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("First-price: how much to shade?")
        st.line_chart(pd.DataFrame({"Expected profit": curve}, index=k), color=C[0], x_label="Shading factor k", y_label="Expected profit")
        st.write(f"Best shading factor: **{k[np.argmax(curve)]:.2f}**  (theory (n−1)/n = {(n - 1) / n:.2f})")
    with c2:
        st.subheader("Reserve (floor) price")
        r = st.slider("Reserve price", 0.0, 1.0, 0.5, 0.01)
        grid = np.linspace(0, 1, 201)
        st.line_chart(pd.DataFrame({"Revenue": reserve_revenue_theory(n, grid)}, index=grid), color=C[1], x_label="Reserve price", y_label="Expected revenue")
        st.write(f"Revenue at r = {r:.2f}: **{float(reserve_revenue_theory(n, r)):.4f}** vs "
                 f"{float(reserve_revenue_theory(n, 0)):.4f} with no reserve; unsold {100 * unsold_probability(n, r):.0f}% of the time")
    st.subheader("Revenue equivalence")
    rows = revenue_equivalence([2, 3, 4, 5, 7, 10], trials=40_000)["rows"]
    st.dataframe(pd.DataFrame(rows).rename(columns={"first_price": "First-price", "second_price": "Second-price",
                                                    "theory": "Theory (n−1)/(n+1)"}), width="stretch")
