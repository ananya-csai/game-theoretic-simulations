"""
Test-suite: every model is checked against a known theoretical result.
Run with:   python -m pytest -v
"""
import numpy as np
import pytest

from adgame.ad_spend_game import build_ad_war, classify_2x2, critical_discount_factor
from adgame.auction_design import (reserve_revenue_sim, reserve_revenue_theory, revenue_equivalence,
                                   shading_payoff_curve, theoretical_shading_payoff)
from adgame.budget_contest import best_response_dynamics, nash_equilibrium, profits, symmetric_summary
from adgame.dynamics import gfp_sawtooth, gsp_balanced_bidding
from adgame.normal_form import BimatrixGame
from adgame.position_auctions import best_deviation, eos_equilibrium_bids, gsp, is_nash, vcg, vcg_prices_per_click
from adgame.repeated_game import tournament


# ------------------------------------------------------------- normal form
def test_prisoners_dilemma_structure():
    g = build_ad_war()
    assert g.pure_nash() == [(1, 1)]                 # (High, High)
    assert g.strictly_dominant(0) == 1 and g.strictly_dominant(1) == 1
    assert (0, 0) in g.pareto_optimal() and (1, 1) not in g.pareto_optimal()
    assert classify_2x2(g)["type"] == "Prisoner's Dilemma"
    assert critical_discount_factor(g) == pytest.approx(0.2)


def test_mixed_equilibrium_matching_pennies():
    g = BimatrixGame([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    eqs = g.mixed_nash()
    assert len(eqs) == 1
    assert np.allclose(eqs[0].row_strategy, [0.5, 0.5]) and np.allclose(eqs[0].col_strategy, [0.5, 0.5])


def test_mixed_equilibria_match_nashpy():
    nash = pytest.importorskip("nashpy")
    A = np.array([[3, 0], [5, 1]]); B = A.T                       # PD
    C = np.array([[0, -1, 2], [3, 1, 0], [-2, 4, 1]]); D = -C       # random zero-sum 3x3
    E = np.array([[2, 0], [0, 1]]); F = np.array([[1, 0], [0, 2]])  # battle of the sexes
    for X, Y in [(A, B), (C, D), (E, F)]:
        ours = BimatrixGame(X, Y).mixed_nash()
        theirs = list(nash.Game(X, Y).support_enumeration())
        assert len(ours) == len(theirs)
        for p, q in theirs:
            assert any(np.allclose(p, e.row_strategy, atol=1e-6) and np.allclose(q, e.col_strategy, atol=1e-6) for e in ours)


def test_three_level_game_matches_continuous_ne():
    g = build_ad_war((10, 25, 40))
    assert g.pure_nash() == [(1, 1)]
    assert nash_equilibrium(100, [1, 1]) == pytest.approx([25, 25])


def test_tournament_reciprocity_beats_always_high():
    t = tournament(build_ad_war(), rounds=200)
    rank = [n for n, _ in t["ranking"]]
    assert rank.index("Tit-for-Tat") < rank.index("Always High")
    assert rank[-1] == "Always High"


# ------------------------------------------------------------- Tullock
def test_tullock_symmetric_closed_form():
    for n in [2, 3, 5, 10]:
        s = symmetric_summary(100, n)
        assert s["total_spend"] == pytest.approx(100 * (n - 1) / n)
        ne = nash_equilibrium(100, [1.0] * n)
        assert ne == pytest.approx([s["spend_per_firm"]] * n)


def test_tullock_asymmetric_is_equilibrium():
    V, c = 100.0, np.array([1.0, 1.2, 1.5])
    ne = nash_equilibrium(V, c)
    base = profits(V, c, ne)
    for i in range(3):                       # no firm gains from any other budget
        for x in np.linspace(0, 60, 601):
            trial = ne.copy(); trial[i] = x
            assert profits(V, c, trial)[i] <= base[i] + 1e-9


def test_best_response_dynamics_converge():
    c = [1.0, 1.2, 1.5]
    path = best_response_dynamics(100, c, iterations=30)
    assert path[-1] == pytest.approx(nash_equilibrium(100, c), abs=1e-8)


def test_high_cost_firm_drops_out():
    assert nash_equilibrium(100, [1.0, 1.5, 3.0]) == pytest.approx([24, 16, 0])


# ------------------------------------------------------------- position auctions
V5 = np.array([12.0, 10.0, 8.0, 6.0, 4.0]); CTR = np.array([0.3, 0.2, 0.1])


def test_vcg_truthful_is_equilibrium():
    rng = np.random.default_rng(0)
    for _ in range(25):
        v = rng.uniform(0, 10, 5)
        assert is_nash(vcg, v, v, CTR)[0]


def test_gsp_truthful_not_equilibrium_eos_example():
    v = np.array([10.0, 4.0, 2.0])
    d = best_deviation(gsp, v, v, [200, 199], 0)
    assert d["current_utility"] == pytest.approx(1200) and d["best_utility"] == pytest.approx(1592)


def test_eos_equilibrium_is_nash_and_equals_vcg():
    rng = np.random.default_rng(1)
    for _ in range(25):
        v = rng.uniform(0, 10, 5)
        b = eos_equilibrium_bids(v, CTR)
        assert is_nash(gsp, b, v, CTR)[0]
        assert gsp(b, v, CTR).revenue == pytest.approx(vcg(v, v, CTR).revenue)


def test_vcg_prices_formula_matches_generic_vcg():
    o = vcg(V5, V5, CTR)
    assert [o.price_per_click[i] for i in o.order] == pytest.approx(vcg_prices_per_click(V5, CTR))


def test_gsp_revenue_with_truthful_bids_is_at_least_vcg():
    rng = np.random.default_rng(2)
    for _ in range(50):
        v = rng.uniform(0, 10, 6)
        assert gsp(v, v, CTR).revenue >= vcg(v, v, CTR).revenue - 1e-9


def test_quality_score_ranking():
    # lower bid but higher quality wins the top slot
    o = gsp([5.0, 4.0], [5.0, 4.0], [0.3], quality=[0.5, 1.0])
    assert o.order == [1]
    assert o.price_per_click[1] == pytest.approx(5.0 * 0.5 / 1.0)


# ------------------------------------------------------------- dynamics
def test_gfp_never_settles():
    b = gfp_sawtooth(rounds=400)["bids"]
    assert np.sum(np.diff(b.max(axis=1)) < -1) >= 5


def test_gsp_balanced_bidding_converges_to_vcg():
    for seed in range(10):
        r = gsp_balanced_bidding(V5, CTR, rounds=300, seed=seed)
        assert r["final_revenue"] == pytest.approx(r["vcg_revenue"])
        assert r["final_bids"] == pytest.approx(eos_equilibrium_bids(V5, CTR))


# ------------------------------------------------------------- Bayesian auctions
def test_bid_shading_optimum():
    k = np.round(np.linspace(0.3, 1.0, 71), 3)
    for n in (2, 3, 5):
        sim = shading_payoff_curve(n, k, trials=100_000)
        assert abs(k[np.argmax(sim)] - (n - 1) / n) <= 0.02
        theo = theoretical_shading_payoff(n, [(n - 1) / n])[0]
        assert sim.max() == pytest.approx(theo, rel=0.03)


def test_revenue_equivalence():
    for r in revenue_equivalence([2, 4, 8], trials=100_000)["rows"]:
        assert r["first_price"] == pytest.approx(r["theory"], abs=0.005)
        assert r["second_price"] == pytest.approx(r["theory"], abs=0.005)


def test_optimal_reserve_is_half():
    r = np.linspace(0, 1, 1001)
    for n in (1, 2, 3, 5):
        assert r[np.argmax(reserve_revenue_theory(n, r))] == pytest.approx(0.5, abs=1e-3)
    assert reserve_revenue_sim(2, [0.5], trials=200_000)[0] == pytest.approx(5 / 12, abs=0.004)
