"""
repeated_game.py  —  Model 1b: the advertising war played every month
----------------------------------------------------------------------
Real advertisers do not choose budgets once; they revise them every week or
month and watch what the rival does. This module plays the ad-war payoff
matrix repeatedly (an Axelrod-style round-robin tournament) to show when firms
can *tacitly* keep budgets low without any explicit agreement.

Action 0 = Low spend ("cooperate"), action 1 = High spend ("defect").
"""
from __future__ import annotations

import itertools

import numpy as np

from .normal_form import BimatrixGame

LOW, HIGH = 0, 1


class Strategy:
    name = "base"

    def reset(self):
        pass

    def move(self, my_hist: list[int], opp_hist: list[int], rng: np.random.Generator) -> int:
        raise NotImplementedError


class AlwaysHigh(Strategy):
    name = "Always High"

    def move(self, my_hist, opp_hist, rng):
        return HIGH


class AlwaysLow(Strategy):
    name = "Always Low"

    def move(self, my_hist, opp_hist, rng):
        return LOW


class TitForTat(Strategy):
    """Start Low, then copy the rival's last move."""
    name = "Tit-for-Tat"

    def move(self, my_hist, opp_hist, rng):
        return LOW if not opp_hist else opp_hist[-1]


class GrimTrigger(Strategy):
    """Start Low; switch to High forever once the rival ever goes High."""
    name = "Grim Trigger"

    def move(self, my_hist, opp_hist, rng):
        return HIGH if HIGH in opp_hist else LOW


class Pavlov(Strategy):
    """Win-stay, lose-shift: repeat last move if both chose the same, else switch."""
    name = "Pavlov"

    def move(self, my_hist, opp_hist, rng):
        if not my_hist:
            return LOW
        return my_hist[-1] if my_hist[-1] == opp_hist[-1] else 1 - my_hist[-1]


class GenerousTFT(Strategy):
    """Tit-for-Tat that forgives a High move with probability 0.1."""
    name = "Generous TFT"

    def move(self, my_hist, opp_hist, rng):
        if not opp_hist or opp_hist[-1] == LOW:
            return LOW
        return LOW if rng.random() < 0.1 else HIGH


class RandomStrategy(Strategy):
    name = "Random"

    def move(self, my_hist, opp_hist, rng):
        return int(rng.integers(0, 2))


ALL_STRATEGIES = [AlwaysHigh, AlwaysLow, TitForTat, GrimTrigger, Pavlov, GenerousTFT, RandomStrategy]


def play_match(game: BimatrixGame, s1: Strategy, s2: Strategy, rounds: int = 200, noise: float = 0.0,
               rng: np.random.Generator | None = None):
    """Play `rounds` periods. With probability `noise` a chosen action is flipped (execution error)."""
    rng = rng or np.random.default_rng(0)
    h1, h2 = [], []
    pay1 = pay2 = 0.0
    for _ in range(rounds):
        a = s1.move(h1, h2, rng)
        b = s2.move(h2, h1, rng)
        if noise and rng.random() < noise:
            a = 1 - a
        if noise and rng.random() < noise:
            b = 1 - b
        h1.append(a); h2.append(b)
        pay1 += game.A[a, b]
        pay2 += game.B[a, b]
    return pay1 / rounds, pay2 / rounds, h1, h2


def tournament(game: BimatrixGame, strategies=None, rounds: int = 200, noise: float = 0.0, seed: int = 7):
    """Round-robin (including self-play). Returns per-strategy average payoff per round,
    the pairwise payoff matrix and the share of Low-spend moves."""
    strategies = strategies or ALL_STRATEGIES
    rng = np.random.default_rng(seed)
    k = len(strategies)
    pair = np.zeros((k, k))
    low_share = np.zeros(k)
    counts = np.zeros(k)
    for i, j in itertools.combinations_with_replacement(range(k), 2):
        p1, p2, h1, h2 = play_match(game, strategies[i](), strategies[j](), rounds, noise, rng)
        pair[i, j] = p1
        pair[j, i] = p2
        low_share[i] += h1.count(LOW) / rounds; counts[i] += 1
        low_share[j] += h2.count(LOW) / rounds; counts[j] += 1
    avg = pair.mean(axis=1)
    order = np.argsort(-avg)
    return {
        "names": [s.name for s in strategies],
        "pairwise": pair,
        "average": avg,
        "low_share": low_share / counts,
        "ranking": [(strategies[i].name, float(avg[i])) for i in order],
    }
