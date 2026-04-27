"""Smoke tests for optimizer.rule_patches and optimizer.rule_sandbox.

Per CEO plan Step 2: smoke test with a hand-written VALID diff and a
hand-written INVALID diff before adding LLM in the loop. The tests below
exercise:

  - validate_patch     -> accepts valid, rejects invalid with a clear reason
  - apply_patch        -> mutates only the targeted GameMechanics field
  - parse_llm_response -> handles fenced JSON, bare JSON, and parse errors
  - run_sandboxed      -> end-to-end subprocess run on a real cfg

The end-to-end sandbox test runs a small game count (n_games=4) so the test
suite stays under a couple of seconds.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import GameConfig
from optimizer.rule_patches import (
    apply_patch, apply_patches, parse_llm_response, validate_patch,
    summarise_patch, render_patch_as_diff,
    RejectedPatch, append_rejection, REJECTED_CORPUS_PATH,
)
from optimizer.rule_sandbox import run_sandboxed


# --------------------------------------------------------------------------- #
# validate_patch                                                                #
# --------------------------------------------------------------------------- #

@pytest.fixture
def mini_cfg():
    return GameConfig.from_yaml(str(REPO_ROOT / 'configs' / 'mini'))


def test_validate_salary_change_valid(mini_cfg):
    err = validate_patch({'kind': 'salary_change', 'value': 300}, mini_cfg)
    assert err is None


def test_validate_salary_change_out_of_bounds(mini_cfg):
    err = validate_patch({'kind': 'salary_change', 'value': 99999}, mini_cfg)
    assert err is not None and 'out of bounds' in err


def test_validate_salary_change_wrong_type(mini_cfg):
    err = validate_patch({'kind': 'salary_change', 'value': 300.5}, mini_cfg)
    assert err is not None and 'must be int' in err


def test_validate_unknown_kind(mini_cfg):
    err = validate_patch({'kind': 'nuclear_option', 'value': 1}, mini_cfg)
    assert err is not None and 'whitelist' in err


def test_validate_property_payout_unknown_group(mini_cfg):
    # mini board has only Brown/Lightblue/Pink/Orange
    err = validate_patch({'kind': 'property_payout_mult',
                           'group': 'Indigo', 'rent_multiplier': 1.5},
                          mini_cfg)
    assert err is not None and 'not on board' in err


def test_validate_card_effect_change_valid(mini_cfg):
    err = validate_patch({'kind': 'card_effect_change',
                           'deck': 'chance', 'index': 0,
                           'new_text': 'Pay the bank $50'},
                          mini_cfg)
    assert err is None


def test_validate_card_effect_change_freeform_text_rejected(mini_cfg):
    # Free-form text outside the allowed token grammar is rejected so the
    # LLM can't smuggle e.g. shell metacharacters or arbitrary instructions
    # into the card text.
    err = validate_patch({'kind': 'card_effect_change',
                           'deck': 'chance', 'index': 0,
                           'new_text': 'eval(__import__("os").system("rm -rf"))'},
                          mini_cfg)
    assert err is not None and 'token grammar' in err


# --------------------------------------------------------------------------- #
# apply_patch                                                                  #
# --------------------------------------------------------------------------- #

def test_apply_salary_change_only_touches_salary(mini_cfg):
    patched = apply_patch(mini_cfg, {'kind': 'salary_change', 'value': 350})
    assert patched.settings.mechanics.salary == 350
    # Confirm nothing else moved
    assert patched.settings.mechanics.luxury_tax == mini_cfg.settings.mechanics.luxury_tax
    assert patched.settings.mechanics.income_tax == mini_cfg.settings.mechanics.income_tax
    assert len(patched.cells) == len(mini_cfg.cells)


def test_apply_property_payout_mult_on_orange(mini_cfg):
    from monopoly.core.cell import Property
    patched = apply_patch(mini_cfg, {'kind': 'property_payout_mult',
                                       'group': 'Orange',
                                       'rent_multiplier': 1.5})
    orig_orange = [c for c in mini_cfg.cells
                   if isinstance(c, Property) and c.group == 'Orange']
    new_orange = [c for c in patched.cells
                   if isinstance(c, Property) and c.group == 'Orange']
    assert len(new_orange) == len(orig_orange) > 0
    for o, n in zip(orig_orange, new_orange):
        assert n.rent_base == int(round(o.rent_base * 1.5))
        assert n.cost_base == o.cost_base   # cost untouched


def test_apply_invalid_patch_raises(mini_cfg):
    with pytest.raises(ValueError):
        apply_patch(mini_cfg, {'kind': 'salary_change', 'value': -100})


def test_apply_patches_sequential(mini_cfg):
    patched = apply_patches(mini_cfg, [
        {'kind': 'salary_change', 'value': 250},
        {'kind': 'jail_fine_change', 'value': 75},
    ])
    assert patched.settings.mechanics.salary == 250
    assert patched.settings.mechanics.exit_jail_fine == 75


# --------------------------------------------------------------------------- #
# parse_llm_response                                                            #
# --------------------------------------------------------------------------- #

def test_parse_fenced_json_object():
    txt = 'Some preamble.\n```json\n{"kind": "salary_change", "value": 300}\n```\nTrailing.'
    patches, err = parse_llm_response(txt)
    assert err is None
    assert patches == [{'kind': 'salary_change', 'value': 300}]


def test_parse_fenced_json_list():
    txt = ('```json\n['
           '{"kind": "salary_change", "value": 300},'
           '{"kind": "jail_fine_change", "value": 75}'
           ']\n```')
    patches, err = parse_llm_response(txt)
    assert err is None
    assert len(patches) == 2


def test_parse_bare_json():
    txt = '{"kind": "salary_change", "value": 300}'
    patches, err = parse_llm_response(txt)
    assert err is None and patches == [{'kind': 'salary_change', 'value': 300}]


def test_parse_garbage():
    patches, err = parse_llm_response('the answer is 42')
    assert patches is None and err is not None


# --------------------------------------------------------------------------- #
# Diff rendering and summary (cosmetic, but exercise both paths)               #
# --------------------------------------------------------------------------- #

def test_render_diff_contains_value():
    diff = render_patch_as_diff({'kind': 'salary_change', 'value': 300})
    assert '300' in diff and 'salary' in diff


def test_summarise_patch_is_short():
    s = summarise_patch({'kind': 'salary_change', 'value': 300})
    assert s == 'salary -> 300'


# --------------------------------------------------------------------------- #
# Rejected corpus persistence                                                  #
# --------------------------------------------------------------------------- #

def test_append_rejection_writes_jsonl(tmp_path):
    p = tmp_path / 'rej.jsonl'
    rej = RejectedPatch(raw_response='```json\n{"kind":"bogus"}\n```',
                         parsed_patch={'kind': 'bogus'},
                         reason="kind='bogus' not in whitelist",
                         iteration=3, board_label='default')
    append_rejection(rej, path=p)
    append_rejection(rej, path=p)   # idempotent append
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec['parsed_patch'] == {'kind': 'bogus'}
    assert rec['iteration'] == 3


# --------------------------------------------------------------------------- #
# End-to-end sandbox: hand-written valid patch (subprocess)                    #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_sandbox_e2e_valid_patch():
    """Smoke run: salary x1.5 on the mini board, n_games=4. Expect ok=True
    and aggregate_score finite."""
    mini = str(REPO_ROOT / 'configs' / 'mini')
    pool = str(REPO_ROOT / 'optimizer' / 'strategy_pool.json')
    matchups = [(0, 1), (2, 3)]
    res = run_sandboxed(
        cfg_yaml_path=mini,
        patches=[{'kind': 'salary_change', 'value': 300}],
        pool_path=pool,
        matchups=matchups,
        n_games=4,
        base_seed=42,
        max_turns=120,
        per_game_timeout_seconds=10.0,
    )
    assert res.ok, f'sandbox failed: {res.failure_reason}'
    assert res.n_games_completed >= 4
    assert res.aggregate_score is not None and res.aggregate_score >= 0


@pytest.mark.slow
def test_sandbox_e2e_invalid_patch_clean_failure():
    """The sandbox must fail cleanly (ok=False with a reason) when given an
    invalid patch — NOT crash the subprocess into a non-JSON exit."""
    mini = str(REPO_ROOT / 'configs' / 'mini')
    pool = str(REPO_ROOT / 'optimizer' / 'strategy_pool.json')
    matchups = [(0, 1)]
    res = run_sandboxed(
        cfg_yaml_path=mini,
        patches=[{'kind': 'salary_change', 'value': -1}],   # out of bounds
        pool_path=pool,
        matchups=matchups,
        n_games=2,
        base_seed=42,
        max_turns=80,
    )
    assert res.ok is False
    assert res.failure_reason is not None
    assert 'apply_patches' in res.failure_reason or 'out of bounds' in res.failure_reason
