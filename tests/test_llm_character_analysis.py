"""Unit tests for the within-board noise floor and grounded-rate
precondition added in Step 3.

These functions can't be visually inspected from the smoke-run output
because the heuristic backend emits constant text; small synthetic
fixtures exercise the actual measurement code paths.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# scripts/ isn't a package, so import via path
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'llm_character', str(REPO_ROOT / 'scripts' / 'llm_character.py'))
llm_character = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llm_character)


# --------------------------------------------------------------------------- #
# within_board_noise_floor                                                     #
# --------------------------------------------------------------------------- #

def test_noise_floor_zero_for_identical_text():
    """Identical reasoning text within a board => L1 between any halves is 0."""
    corpora = {
        'A': ['the same thing'] * 20,
        'B': ['the same thing'] * 20,
    }
    n = llm_character.within_board_noise_floor(corpora, n_splits=10, seed=0)
    assert n['pooled_mean'] == 0.0
    assert n['pooled_std'] == 0.0
    assert n['meaningful_threshold'] == 0.0


def test_noise_floor_nonzero_for_varied_text():
    """Two boards each with mixed concept vocabulary should give a nonzero
    within-board L1 (random halves draw different concept proportions)."""
    corpora = {
        'A': ['I will trade for cash', 'I want a monopoly', 'pay rent next turn',
              'cash is risk', 'risk vs reward', 'opponent is bankrupt',
              'accumulate properties', 'block the opponent'] * 5,
        'B': ['save money', 'avoid risk', 'complete the set', 'collect rent',
              'they own three', 'I have cash'] * 5,
    }
    n = llm_character.within_board_noise_floor(corpora, n_splits=50, seed=1)
    assert n['pooled_mean'] > 0.0
    assert n['pooled_std'] >= 0.0
    assert n['meaningful_threshold'] > n['pooled_mean']


def test_noise_floor_corpus_too_small():
    """Small corpora should emit a 'too small' note and not crash."""
    corpora = {'A': ['one block'], 'B': []}
    n = llm_character.within_board_noise_floor(corpora, n_splits=5, seed=0)
    assert 'note' in n['per_board']['A']
    assert n['per_board']['A']['samples'] == []


# --------------------------------------------------------------------------- #
# per_board_grounded_rate                                                      #
# --------------------------------------------------------------------------- #

def _decision(board, prop_name, reason, format_ok=True, backend_error=None):
    return {
        'board': board, 'prop_name': prop_name, 'reason': reason,
        'format_ok': format_ok, 'backend_error': backend_error,
        'decision': 'BUY',
    }


def test_grounded_rate_counts_property_name():
    decisions = [
        _decision('A', 'Boardwalk', 'I will buy boardwalk for monopoly potential'),
        _decision('A', 'Boardwalk', 'I should accumulate'),
        _decision('A', 'Boardwalk', 'opponent is winning'),
    ]
    g = llm_character.per_board_grounded_rate(decisions)
    assert g['A']['n_total'] == 3
    assert g['A']['n_has_prop'] == 1   # only the first reasoning mentions 'boardwalk'


def test_grounded_rate_counts_money_integer():
    decisions = [
        _decision('A', 'X', 'I have $250 and rent is 60'),
        _decision('A', 'X', 'cash is plentiful'),
    ]
    g = llm_character.per_board_grounded_rate(decisions)
    assert g['A']['n_has_money'] == 1


def test_grounded_rate_counts_owned_keyword():
    decisions = [
        _decision('A', 'X', 'i own three properties already'),
        _decision('A', 'X', 'pass on this one'),
    ]
    g = llm_character.per_board_grounded_rate(decisions)
    assert g['A']['n_has_owned'] == 1


def test_grounded_rate_skips_failed_decisions():
    decisions = [
        _decision('A', 'X', 'i own three properties'),
        _decision('A', 'X', 'malformed', format_ok=False),
        _decision('A', 'X', 'crashed', backend_error='RuntimeError'),
    ]
    g = llm_character.per_board_grounded_rate(decisions)
    assert g['A']['n_total'] == 1   # the two failed decisions are dropped


def test_grounded_rate_per_board_independence():
    """Property-name dictionary is per-board so a property name on board B
    should not count as 'has_prop' if the reasoning is on board A."""
    decisions = [
        _decision('A', 'Alpha', 'I want Alpha for the monopoly'),
        _decision('B', 'Beta',  'I want Alpha for the monopoly'),  # mentions A's prop
    ]
    g = llm_character.per_board_grounded_rate(decisions)
    # Board A: 'Alpha' is in its prop dict -> grounded
    assert g['A']['n_has_prop'] == 1
    # Board B: 'Alpha' is NOT in its prop dict -> not grounded by prop
    # (could still be grounded by 'monopoly'? no - that's a concept, not a prop name)
    assert g['B']['n_has_prop'] == 0
