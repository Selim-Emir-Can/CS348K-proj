"""Regression tests for the analytic estimators that back the §5
'Game-Space Exploration' figures.

These are the load-bearing computations: a sign error or off-by-one in any
of them silently changes the report's narrative ("flat" vs "noisy" hazard,
"max-divergence pair", "novelty triple"). The tests do not exercise the
Monopoly engine -- they pin down the math against synthetic inputs where
the right answer is known by hand.

Run from the project root:

    set PYTHONPATH=. && python -m pytest tests/ -v

The tests are pytest-style but only depend on `assert`, so they also run
under stdlib unittest discovery if pytest is not installed.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow `from scripts.* import *` when invoked directly without setting
# PYTHONPATH (useful for IDE-launched test runs).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hazard_curves import (
    _wilson,
    cash_bankruptcy_hazard,
    first_monopoly_hazard,
    game_end_hazard,
)
from scripts.llm_character import (
    _extract_reason,
    bootstrap_divergence,
    concept_counts,
    divergence_matrix,
    l1_divergence,
    normalize_freq,
    per_board_corpus,
    per_board_quality,
    per_board_stats,
    permutation_test_max_divergence,
)
from scripts.novelty_search import max_min_tuple, normalize_vecs


# --------------------------------------------------------------------------- #
# Wilson 95% CI                                                                #
# --------------------------------------------------------------------------- #

def test_wilson_known_values():
    """The Wilson 95% CI for p=0.5 at n=100 is well-known: ~ (0.404, 0.596).
    A future formula drift in `_wilson` would silently widen or narrow every
    confidence band the report draws on its hazard plots."""
    lo, hi = _wilson(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_zero_n():
    """n=0 must NOT divide by zero; both bounds collapse to 0."""
    lo, hi = _wilson(0.0, 0)
    assert lo == 0.0 and hi == 0.0


def test_wilson_extremes_clipped():
    """Bounds clip to [0, 1] even at p=0 and p=1."""
    lo0, hi0 = _wilson(0.0, 50)
    assert lo0 == 0.0
    assert 0.0 < hi0 < 0.1
    lo1, hi1 = _wilson(1.0, 50)
    assert hi1 == 1.0
    assert 0.9 < lo1 < 1.0


# --------------------------------------------------------------------------- #
# Game-end hazard                                                              #
# --------------------------------------------------------------------------- #

def test_game_end_hazard_synthetic():
    """Three games ending at turns [5, 5, 10]:
      t=1..4 : 0/3, 0/3, 0/3, 0/3        -> hazard 0
      t=5    : 2/3                        -> hazard 2/3
      t=6..9 : at_risk=1, ended=0         -> hazard 0
      t=10   : at_risk=1, ended=1         -> hazard 1
    """
    games = [{'rounds': r, 'snapshots': []} for r in [5, 5, 10]]
    out = game_end_hazard(games, max_t=10)
    h = out['hazard']
    assert h[0] == 0.0 and h[3] == 0.0
    assert h[4] == pytest.approx(2 / 3, abs=1e-9)   # turn 5 (index 4)
    assert h[5] == 0.0 and h[8] == 0.0
    assert h[9] == pytest.approx(1.0, abs=1e-9)     # turn 10 (index 9)


# --------------------------------------------------------------------------- #
# Cash-level bankruptcy hazard                                                 #
# --------------------------------------------------------------------------- #

def test_cash_bankruptcy_hazard_synthetic():
    """One game, one player. Cash trajectory across 4 snapshots:
        $100 -> $200 -> $50 -> BANKRUPT
    With cash bins [0, 75, 250]:
        triple t=1: cash 100 (bin 1)  -> not bankrupt at t=2
        triple t=2: cash 200 (bin 1)  -> not bankrupt at t=3
        triple t=3: cash  50 (bin 0)  -> BANKRUPT at t=4   <- the only event
    Expected: bin 0 has counts=1 events=1 hazard=1.0;
              bin 1 has counts=2 events=0 hazard=0.0.
    """
    snaps = [
        {'turn': 1, 'players': [{'name': 'A', 'money': 100, 'is_bankrupt': False, 'monopolies': 0}], 'any_monopoly': False},
        {'turn': 2, 'players': [{'name': 'A', 'money': 200, 'is_bankrupt': False, 'monopolies': 0}], 'any_monopoly': False},
        {'turn': 3, 'players': [{'name': 'A', 'money':  50, 'is_bankrupt': False, 'monopolies': 0}], 'any_monopoly': False},
        {'turn': 4, 'players': [{'name': 'A', 'money':   0, 'is_bankrupt': True,  'monopolies': 0}], 'any_monopoly': False},
    ]
    games = [{'rounds': 4, 'snapshots': snaps}]
    out = cash_bankruptcy_hazard(games, cash_edges=[0, 75, 250])
    assert out['counts'] == [1, 2]
    assert out['events'] == [1, 0]
    assert out['hazard'][0] == pytest.approx(1.0)
    assert out['hazard'][1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# First-monopoly hazard (right-censoring)                                      #
# --------------------------------------------------------------------------- #

def test_first_monopoly_hazard_right_censoring():
    """A game with NO monopoly ever forming must contribute to `at_risk` at
    every turn but never to `events`. Two games, max_t=3:
      g1: any_monopoly=[False, True,  True]    -> first event at t=2
      g2: any_monopoly=[False, False, False]   -> right-censored
    Expected hazards:
      t=1: at_risk=2 events=0 -> 0
      t=2: at_risk=2 events=1 -> 0.5
      t=3: at_risk=1 events=0 -> 0   (g1 already had its event; g2 still at risk)
    """
    g1 = {'rounds': 3, 'snapshots': [
        {'turn': 1, 'players': [], 'any_monopoly': False},
        {'turn': 2, 'players': [], 'any_monopoly': True},
        {'turn': 3, 'players': [], 'any_monopoly': True},
    ]}
    g2 = {'rounds': 3, 'snapshots': [
        {'turn': 1, 'players': [], 'any_monopoly': False},
        {'turn': 2, 'players': [], 'any_monopoly': False},
        {'turn': 3, 'players': [], 'any_monopoly': False},
    ]}
    out = first_monopoly_hazard([g1, g2], max_t=3)
    h = out['hazard']
    assert h[0] == pytest.approx(0.0)
    assert h[1] == pytest.approx(0.5)
    assert h[2] == pytest.approx(0.0)
    assert out['n_first_events'] == 1
    assert out['n_right_censored'] == 1


# --------------------------------------------------------------------------- #
# Novelty: max-min tuple                                                       #
# --------------------------------------------------------------------------- #

def test_max_min_tuple_known_optimum():
    """Four points whose pairwise distances are constructed so the max-min
    triple is unambiguous. Distance matrix:
                 0     1    2    3
            0 [ inf  10.0  9.0  1.0 ]
            1 [10.0  inf  8.0  2.0 ]
            2 [ 9.0  8.0  inf  3.0 ]
            3 [ 1.0  2.0  3.0  inf ]
    Triples (min pairwise):
      (0,1,2): min(10, 9, 8) = 8.0   <- best
      (0,1,3): min(10, 1, 2) = 1.0
      (0,2,3): min(9, 1, 3)  = 1.0
      (1,2,3): min(8, 2, 3)  = 2.0
    """
    D = np.array([
        [np.inf, 10.0,  9.0, 1.0],
        [10.0,  np.inf, 8.0, 2.0],
        [9.0,   8.0,   np.inf, 3.0],
        [1.0,   2.0,   3.0,   np.inf],
    ])
    chosen, min_d = max_min_tuple(D, K=3)
    assert sorted(chosen) == [0, 1, 2]
    assert min_d == pytest.approx(8.0)


def test_max_min_tuple_pool_too_small():
    D = np.full((2, 2), np.inf)
    with pytest.raises(ValueError):
        max_min_tuple(D, K=3)


# --------------------------------------------------------------------------- #
# Novelty: normalize_vecs                                                      #
# --------------------------------------------------------------------------- #

def test_normalize_vecs_handles_collapsed_dim():
    """A dimension where lo == hi must NOT divide by zero. The normalised
    output for that dim is just (vec - lo) / 1.0 (i.e., shifted, not scaled).
    """
    vecs = np.array([[5.0, 0.0], [10.0, 0.0]])
    bounds = [(0.0, 20.0), (0.0, 0.0)]
    norm = normalize_vecs(vecs, bounds)
    assert norm[0, 0] == pytest.approx(0.25)
    assert norm[1, 0] == pytest.approx(0.5)
    # Collapsed dim: (0 - 0)/1 == 0 for both
    assert norm[0, 1] == 0.0 and norm[1, 1] == 0.0


# --------------------------------------------------------------------------- #
# LLM character: _extract_reason                                               #
# --------------------------------------------------------------------------- #

def test_extract_reason_well_formed():
    text, ok = _extract_reason('REASON: cash too low.\nANSWER: PASS')
    assert ok is True
    assert text == 'cash too low.'


def test_extract_reason_well_formed_inline():
    """The regex also matches when ANSWER follows REASON without a newline."""
    text, ok = _extract_reason('REASON: complete the orange set. ANSWER: BUY')
    assert ok is True
    assert 'orange set' in text


def test_extract_reason_malformed_returns_format_ok_false():
    """A response with no REASON tag falls back to the truncation heuristic
    and MUST report format_ok=False so the analysis pass can drop it."""
    text, ok = _extract_reason('I would buy this property because it seems good')
    assert ok is False
    assert len(text) <= 200


# --------------------------------------------------------------------------- #
# LLM character: concept counting                                              #
# --------------------------------------------------------------------------- #

def test_concept_counts_dedups_per_reasoning():
    """A concept mentioned three times in a single reasoning hits its count
    EXACTLY ONCE -- otherwise one verbose reasoning dominates the corpus.
    """
    counts = concept_counts(['cash cash cash and more cash'])
    assert counts['cash'] == 1


def test_concept_counts_handles_empty():
    counts = concept_counts(['', None, 'no concepts mentioned here'])
    assert sum(counts.values()) == 0


def test_l1_divergence_known_values():
    """L1 divergence is 0.5 * sum(|p - q|). Two distributions that put all
    mass on a single (different) concept have divergence 1.0.
    """
    p = {'cash': 1.0, 'risk': 0.0, 'monopoly': 0.0, 'trade': 0.0,
         'rent': 0.0, 'bankrupt': 0.0, 'accumulate': 0.0, 'defensive': 0.0,
         'opponent': 0.0, 'group': 0.0}
    q = {'cash': 0.0, 'risk': 1.0, 'monopoly': 0.0, 'trade': 0.0,
         'rent': 0.0, 'bankrupt': 0.0, 'accumulate': 0.0, 'defensive': 0.0,
         'opponent': 0.0, 'group': 0.0}
    assert l1_divergence(p, q) == pytest.approx(1.0)
    assert l1_divergence(p, p) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Per-board corpus filtering                                                   #
# --------------------------------------------------------------------------- #

def test_per_board_corpus_drops_backend_errors():
    decisions = [
        {'board': 'X', 'reason': 'good',  'decision': 'BUY',
         'format_ok': True, 'backend_error': None},
        {'board': 'X', 'reason': 'noise', 'decision': 'BUY',
         'format_ok': False, 'backend_error': None},
        {'board': 'X', 'reason': 'oom',   'decision': 'BUY',
         'format_ok': True, 'backend_error': 'OOMError: ...'},
    ]
    corpora = per_board_corpus(decisions)
    assert corpora['X'] == ['good']


def test_per_board_quality_rates():
    decisions = [
        {'board': 'X', 'reason': '', 'decision': 'BUY', 'format_ok': True,
         'backend_error': None},
        {'board': 'X', 'reason': '', 'decision': 'BUY', 'format_ok': False,
         'backend_error': None},
        {'board': 'X', 'reason': '', 'decision': 'BUY', 'format_ok': True,
         'backend_error': 'oops'},
        {'board': 'X', 'reason': '', 'decision': 'BUY', 'format_ok': True,
         'backend_error': None},
    ]
    q = per_board_quality(decisions)['X']
    assert q['n_total'] == 4
    assert q['n_format_ok'] == 2
    assert q['n_backend_error'] == 1
    assert q['format_ok_rate'] == pytest.approx(0.5)
    assert q['backend_error_rate'] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Stats guards                                                                 #
# --------------------------------------------------------------------------- #

def test_permutation_test_high_p_when_corpora_identical():
    """Two boards with identical reasoning text MUST produce a high p-value
    (the headline 'character induction' finding cannot survive when there's
    no per-board signal)."""
    corpora = {
        'A': ['cash is tight'] * 50,
        'B': ['cash is tight'] * 50,
    }
    out = permutation_test_max_divergence(corpora, n_resamples=100, seed=0)
    assert out['observed'] == pytest.approx(0.0)
    assert out['p_value'] >= 0.5


def test_permutation_test_low_p_when_corpora_disjoint():
    """Two boards whose reasonings hit DIFFERENT concept dictionaries should
    produce a low p-value (signal recoverable above the H0 null)."""
    corpora = {
        'A': ['cash is the priority'] * 30,
        'B': ['monopoly the orange group'] * 30,
    }
    out = permutation_test_max_divergence(corpora, n_resamples=200, seed=0)
    assert out['observed'] > 0.5
    assert out['p_value'] < 0.05


def test_bootstrap_divergence_zero_when_identical_corpora():
    corpora = {'A': ['cash low'] * 20, 'B': ['cash low'] * 20}
    out = bootstrap_divergence(corpora, n_resamples=100, seed=0)
    assert out['mean'] == pytest.approx(0.0)
    assert out['max_pair_ci_lo'] == pytest.approx(0.0)
    assert out['max_pair_ci_hi'] == pytest.approx(0.0)
