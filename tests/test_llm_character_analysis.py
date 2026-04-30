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


# --------------------------------------------------------------------------- #
# format_pass_rate_gate (round 1 §1.3)                                          #
# --------------------------------------------------------------------------- #

def test_format_pass_rate_gate_passes_above_threshold():
    decisions = (
        [_decision('A', 'X', f'r{i}', format_ok=True) for i in range(8)] +
        [_decision('A', 'X', f'b{i}', format_ok=False) for i in range(2)]
    )
    out = llm_character.format_pass_rate_gate(decisions, threshold=0.70)
    assert out['per_board']['A']['rate'] == pytest.approx(0.8)
    assert out['per_board']['A']['pass'] is True
    assert out['all_pass'] is True
    assert out['failing'] == []


def test_format_pass_rate_gate_fails_below_threshold():
    decisions = (
        [_decision('A', 'X', f'r{i}', format_ok=True) for i in range(6)] +
        [_decision('A', 'X', f'b{i}', format_ok=False) for i in range(4)]
    )
    out = llm_character.format_pass_rate_gate(decisions, threshold=0.70)
    assert out['per_board']['A']['rate'] == pytest.approx(0.6)
    assert out['per_board']['A']['pass'] is False
    assert out['all_pass'] is False
    assert 'A' in out['failing']


def test_format_pass_rate_gate_per_board_independence():
    """One bad board fails the gate without dragging down the others."""
    decisions = (
        [_decision('good', 'X', 'ok', format_ok=True) for _ in range(10)] +
        [_decision('bad',  'X', 'r', format_ok=True) for _ in range(2)] +
        [_decision('bad',  'X', 'b', format_ok=False) for _ in range(8)]
    )
    out = llm_character.format_pass_rate_gate(decisions, threshold=0.70)
    assert out['per_board']['good']['pass'] is True
    assert out['per_board']['bad']['pass'] is False
    assert out['failing'] == ['bad']


# --------------------------------------------------------------------------- #
# Robustness checks (round 1 §1.3)                                              #
# --------------------------------------------------------------------------- #

def _canned_decisions():
    """Two boards, deliberately different reasoning vocabularies, large
    enough corpora to support the noise-floor split."""
    a_text = ('I will buy this for the monopoly and complete the colour set; '
              'cash is plentiful and rent return is excellent.')
    b_text = ('I should pass; risk of bankruptcy looks high if rent income '
              'on opponent properties drains my cash.')
    out = []
    for i in range(20):
        out.append(_decision('A', 'X', f'{a_text} ({i})'))
        out.append(_decision('B', 'X', f'{b_text} ({i})'))
    return out


def test_empath_robustness_returns_symmetric_zero_diag_or_skips():
    """Empath may not be installed in the dev image; the function should
    return a non-crashing dict either way. When empath IS available, the
    returned matrix is symmetric and zero on the diagonal."""
    out = llm_character.empath_robustness(_canned_decisions(),
                                            n_within_splits=10, seed=0)
    if 'note' in out and out['note'].startswith('empath unavailable'):
        # OK if empath isn't installed in the dev image; we don't fail
        # the test purely on dependency availability.
        return
    matrix = out['matrix']
    assert len(matrix) == len(out['boards'])
    n = len(matrix)
    if n == 0:
        return
    for i in range(n):
        assert matrix[i][i] == pytest.approx(0.0, abs=1e-9)
        for j in range(n):
            assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-9)


def test_sbert_robustness_returns_non_nan_matrix_or_skips():
    """sentence-transformers + downloaded weights may not be available;
    skip cleanly. If it IS available, the cosine matrix must be finite."""
    import math
    out = llm_character.sbert_robustness(_canned_decisions())
    if 'note' in out and out.get('note', '').startswith(
            ('sentence-transformers unavailable', 'failed to load')):
        return
    matrix = out.get('cosine_matrix') or []
    if not matrix:
        return
    for row in matrix:
        for v in row:
            assert not math.isnan(v)


# --------------------------------------------------------------------------- #
# Generation-config recording (round 1 §1.3)                                    #
# --------------------------------------------------------------------------- #

def test_character_player_records_gen_cfg():
    """Round 1 lock: every per-decision record carries a gen_cfg snapshot
    of the locked generation parameters (model, max_new_tokens, do_sample,
    temperature, seed)."""
    p = llm_character.CharacterLLMPlayer(
        'LLM-test', backend='heuristic', max_new_tokens=160, base_seed=42)
    cfg = p._gen_cfg
    assert cfg['max_new_tokens'] == 160
    assert cfg['do_sample'] is False
    assert cfg['temperature'] == 0.0
    assert cfg['seed'] == 42
    assert 'model' in cfg

