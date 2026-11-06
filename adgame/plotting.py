"""
plotting.py — every figure used in the report and slides.

Consistent style: thin marks, recessive grid, a fixed categorical colour order
(colour-blind-safe), and text in neutral ink rather than series colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 2.0, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Model 1
def payoff_matrix(game, out, name="fig01_payoff_matrix", title="Advertising war: payoff matrix (₹ lakh profit)"):
    m, n = game.shape
    nash = set(game.pure_nash())
    pareto = set(game.pareto_optimal())
    fig, ax = plt.subplots(figsize=(1.9 * n + 1.6, 1.25 * m + 1.3))
    ax.set_xlim(0, n); ax.set_ylim(0, m); ax.invert_yaxis(); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(n) + 0.5, [f"{game.players[1]}: {c}" for c in game.col_labels])
    ax.set_yticks(np.arange(m) + 0.5, [f"{game.players[0]}: {r}" for r in game.row_labels])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for i in range(m):
        for j in range(n):
            is_ne = (i, j) in nash
            face = "#cde2fb" if is_ne else "#f6f6f4"
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, facecolor=face,
                                       edgecolor=SERIES[0] if is_ne else GRID, linewidth=2 if is_ne else 1))
            ax.text(j + 0.5, i + 0.42, f"{game.A[i, j]:.1f} , {game.B[i, j]:.1f}", ha="center", va="center",
                    fontsize=13, color=INK, fontweight="bold" if is_ne else "normal")
            tags = []
            if is_ne:
                tags.append("Nash equilibrium")
            if (i, j) in pareto and game.joint_payoff((i, j)) == max(game.joint_payoff(c) for c in pareto):
                tags.append("best joint profit")
            if tags:
                ax.text(j + 0.5, i + 0.72, " · ".join(tags), ha="center", va="center", fontsize=8.5, color=INK2)
    ax.set_title(title, pad=34)
    return _save(fig, out, name)


def game_type_map(grid, names, high_values, growth_values, out, name="fig02_game_type_map", marker=None):
    present = sorted(set(grid.flatten()))
    palette = {0: SERIES[1], 1: SERIES[4], 2: SERIES[6], 3: SERIES[2], 4: SERIES[0], 5: "#c3c2b7"}
    cmap = ListedColormap([palette[k] for k in range(len(names))])
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.grid(False)
    ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5,
              extent=[high_values[0], high_values[-1], growth_values[0], growth_values[-1]])
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in present]
    labels = [names[k] for k in present]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    if marker:
        ax.plot(*marker, marker="o", markersize=9, color="white", markeredgecolor=INK, markeredgewidth=1.5)
        ax.annotate("base case\n(High = 30, g = 0)", marker, xytext=(marker[0] + 6, marker[1] + 0.25),
                    color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK))
    ax.set_xlabel("High advertising budget (Low fixed at 10)")
    ax.set_ylabel("Market growth g per ₹ of advertising")
    ax.set_title("Which game are the two firms playing?")
    return _save(fig, out, name)


def tournament_bars(t_clean, t_noisy, out, name="fig03_repeated_tournament"):
    names = t_clean["names"]
    order = np.argsort(-t_clean["average"])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    h = 0.38
    ax.barh(y - h / 2, t_clean["average"][order], height=h - 0.04, color=SERIES[0], label="no mistakes")
    ax.barh(y + h / 2, t_noisy["average"][order], height=h - 0.04, color=SERIES[1], label="5% execution mistakes")
    ax.set_yticks(y, [names[i] for i in order]); ax.invert_yaxis()
    ax.axvline(20, color=MUTED, lw=1, ls="--"); ax.text(20.3, -0.75, "one-shot NE profit = 20", color=INK2, fontsize=8.5)
    ax.axvline(40, color=MUTED, lw=1, ls=":"); ax.text(39.7, -0.75, "cooperative = 40", color=INK2, fontsize=8.5, ha="right")
    ax.set_xlabel("Average profit per month (₹ lakh)")
    ax.set_xlim(0, 44)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(len(names) - 0.5, -1.1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_title("Repeated ad war: 200-month round-robin tournament")
    return _save(fig, out, name)


# ---------------------------------------------------------------- Model 2
def br_convergence(path, ne, labels, out, name="fig04_best_response_convergence"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    it = np.arange(len(path))
    for k in range(path.shape[1]):
        ax.plot(it, path[:, k], color=SERIES[k], marker="o", markersize=3.5, label=labels[k])
        ax.axhline(ne[k], color=SERIES[k], lw=1, ls="--", alpha=0.6)
        ax.text(len(path) - 1, ne[k] + 0.6, f"NE = {ne[k]:.2f}", color=INK2, ha="right", fontsize=8.5)
    ax.set_xlabel("Round of best-response updates")
    ax.set_ylabel("Advertising budget (₹ lakh)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title("Budgets converge to the Nash equilibrium in a few rounds")
    return _save(fig, out, name)


def dissipation(rows, out, name="fig05_rent_dissipation"):
    n = [r["n"] for r in rows]
    spend = [100 * r["dissipation_ratio"] for r in rows]
    prof = [100 * r["industry_profit"] / (r["industry_profit"] + r["total_spend"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(n, spend, color=SERIES[1], marker="o", markersize=5, label="Total ad spend (goes to platform)")
    ax.plot(n, prof, color=SERIES[0], marker="o", markersize=5, label="Total profit kept by firms")
    for x, s, p in zip(n, spend, prof):
        if x in (3, 5, 10, 20):
            ax.text(x, s + 3, f"{s:.0f}%", ha="center", color=INK2, fontsize=8.5)
            ax.text(x, p - 6.5, f"{p:.0f}%", ha="center", color=INK2, fontsize=8.5)
    ax.set_xlabel("Number of competing advertisers (n)")
    ax.set_ylabel("Share of the customer value V (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(n)
    ax.legend(loc="center right")
    ax.set_title("More competitors → the platform captures almost all the value")
    return _save(fig, out, name)


# ---------------------------------------------------------------- Model 3
def mechanism_comparison(example, mc, out, name="fig06_mechanism_revenue"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = list(example.keys())
    colors = [SERIES[0], SERIES[1], SERIES[2]]
    for ax, data, title, unit in [
        (axes[0], example, "Worked example (5 bidders, 3 slots)", "Revenue per search (₹)"),
        (axes[1], {k: v["revenue"] for k, v in mc.items()}, "Monte Carlo, 10,000 random auctions", "Average revenue per search (₹)"),
    ]:
        vals = [data[k] for k in labels]
        bars = ax.bar(range(len(labels)), vals, color=colors, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=10)
        ax.set_xticks(range(len(labels)), [l.replace(" (", "\n(") for l in labels], fontsize=9)
        ax.set_ylabel(unit); ax.set_title(title, fontsize=11.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("GSP in equilibrium earns exactly the VCG revenue", x=0.02, ha="left", fontweight="bold", color=INK, fontsize=12.5)
    fig.tight_layout()
    return _save(fig, out, name)


# ---------------------------------------------------------------- Model 4
def sawtooth(res, out, name="fig07_gfp_sawtooth"):
    b = res["bids"]
    t = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, b[:, 0], color=SERIES[0], lw=1.6, label=f"Advertiser A (value ₹{res['values'][0]:.0f}/click)")
    ax.plot(t, b[:, 1], color=SERIES[1], lw=1.6, label=f"Advertiser B (value ₹{res['values'][1]:.0f}/click)")
    ax.set_xlabel("Bid revision (alternating moves)")
    ax.set_ylabel("Bid per click (₹)")
    ax.set_xlim(0, 160)
    ax.set_ylim(0, res["peak_bid"] * 1.25)
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("First-price (GFP) ad auction: a never-ending saw-tooth bidding war")
    return _save(fig, out, name)


def balanced_bidding(res, eos_bids, labels, out, name="fig08_gsp_convergence"):
    b = res["bids"]
    t = np.arange(len(b))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for k in range(b.shape[1]):
        ax.plot(t, b[:, k], color=SERIES[k], lw=1.6, label=labels[k])
        ax.axhline(eos_bids[k], color=SERIES[k], lw=0.9, ls="--", alpha=0.6)
    ax.set_xlim(0, 80)
    ax.set_xlabel("Bid update (one random advertiser at a time)")
    ax.set_ylabel("Bid per click (₹)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5, ncol=5)
    ax.set_title("Bids settle on the equilibrium (dashed)", fontsize=11.5)
    ax = axes[1]
    ax.plot(t, res["revenue"], color=SERIES[6], lw=1.8)
    ax.axhline(res["vcg_revenue"], color=MUTED, ls="--", lw=1)
    ax.text(80, res["vcg_revenue"] + 0.08, f"VCG revenue = ₹{res['vcg_revenue']:.2f}", ha="right", color=INK2, fontsize=9)
    ax.set_xlim(0, 80)
    ax.set_xlabel("Bid update")
    ax.set_ylabel("Platform revenue per search (₹)")
    ax.set_title("Revenue converges to VCG", fontsize=11.5)
    fig.suptitle("GSP with balanced bidding: stable and predictable", x=0.02, ha="left", fontweight="bold", color=INK, fontsize=12.5)
    fig.tight_layout()
    return _save(fig, out, name)


# ---------------------------------------------------------------- Model 5
def shading(curves, factors, out, name="fig09_bid_shading"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for idx, (n, ys) in enumerate(curves.items()):
        ax.plot(factors, ys, color=SERIES[idx], label=f"n = {n} bidders")
        k = factors[int(np.argmax(ys))]
        ax.plot(k, ys.max(), "o", color=SERIES[idx], markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(k, ys.max() + 0.006, f"best = {k:.2f}\n(theory {(n - 1) / n:.2f})", ha="center", color=INK2, fontsize=8.5)
    ax.set_xlabel("Shading factor k (bid = k × value)")
    ax.set_ylabel("Expected profit per impression")
    ax.set_ylim(0, max(max(v) for v in curves.values()) * 1.3)
    ax.legend(loc="upper right")
    ax.set_title("First-price display auctions: optimal bid shading = (n−1)/n")
    return _save(fig, out, name)


def revenue_equiv(rows, out, name="fig10_revenue_equivalence"):
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    nn = np.linspace(2, max(n), 100)
    ax.plot(nn, (nn - 1) / (nn + 1), color=MUTED, lw=1.2, ls="--", label="Theory (n−1)/(n+1)")
    ax.plot(n, [r["first_price"] for r in rows], "o", color=SERIES[0], markersize=9, label="First-price (simulated)")
    ax.plot(n, [r["second_price"] for r in rows], "s", color=SERIES[1], markersize=6, label="Second-price (simulated)")
    ax.set_xlabel("Number of bidders n"); ax.set_ylabel("Expected revenue per impression")
    ax.set_xticks(n)
    ax.legend(loc="lower right")
    ax.set_title("Revenue equivalence: both formats earn the same")
    return _save(fig, out, name)


def reserve_curve(r, theory, sim_r, sim, n, out, name="fig11_reserve_price"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(r, theory, color=SERIES[0], label="Theory")
    ax.plot(sim_r, sim, "o", color=SERIES[1], markersize=6, label="Simulation (200,000 auctions)")
    k = int(np.argmax(theory))
    ax.axvline(r[k], color=MUTED, ls="--", lw=1)
    ax.text(r[k] + 0.02, theory[k] + 0.012, f"optimal reserve r* = {r[k]:.2f}\nrevenue {theory[k]:.3f} (+{100 * (theory[k] / theory[0] - 1):.0f}%)",
            color=INK2, fontsize=9)
    ax.set_xlabel("Reserve (floor) price r"); ax.set_ylabel("Expected revenue per impression")
    ax.set_ylim(0, theory.max() * 1.3)
    ax.legend(loc="lower left")
    ax.set_title(f"Platform design: a floor price raises revenue (n = {n})")
    return _save(fig, out, name)
