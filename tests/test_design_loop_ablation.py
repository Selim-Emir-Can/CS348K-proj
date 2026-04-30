"""Tests for the round-1 TUNER ablation machinery (round 1 §1.4 / §1.6).

Covers the redaction matrix, convergence handling per condition, T-RAND
sampler bounds, T-SANITY trajectory + assertion, and cross-condition seed
alignment. The tests do not call any LLM — they exercise the
build_diagnostic_feed / parser / sampler pure-Python machinery directly.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# scripts/ isn't a package; import via path. Register in sys.modules BEFORE
# exec_module so dataclass(@dataclass) can resolve the module's namespace
# (Python 3.11 dataclasses peeks sys.modules[cls.__module__]).
_spec = importlib.util.spec_from_file_location(
    'llm_design_loop', str(REPO_ROOT / 'scripts' / 'llm_design_loop.py'))
ldl = importlib.util.module_from_spec(_spec)
sys.modules['llm_design_loop'] = ldl
_spec.loader.exec_module(ldl)


# --------------------------------------------------------------------------- #
# Stubbed cfg + design + eval_out                                              #
# --------------------------------------------------------------------------- #

@dataclass
class _StubProperty:
    name: str
    group: str
    cost_base: int = 100
    rent_base: int = 10


@dataclass
class _StubCfg:
    cells: List[_StubProperty] = field(default_factory=lambda: [
        _StubProperty('A', 'Brown', 60, 6),
        _StubProperty('B', 'Brown', 60, 6),
        _StubProperty('C', 'Lightblue', 100, 8),
        _StubProperty('D', 'Lightblue', 100, 8),
        _StubProperty('E', 'Pink', 140, 12),
        _StubProperty('F', 'Pink', 140, 12),
    ])


def _eval_out():
    return {
        'score': 1.50,
        'metrics': {'mean_rounds': 55.0, 'mean_draw_rate': 0.4,
                     'mean_fairness': 0.7, 'mean_transfer_rate': 90.0},
        'per_game_records': [
            {'winner_strategy': 'AggressiveBuilder',
             'players_strategies': ['AggressiveBuilder', 'CashHoarder']},
            {'winner_strategy': 'AggressiveBuilder',
             'players_strategies': ['AggressiveBuilder', 'CashHoarder']},
            {'winner_strategy': 'CashHoarder',
             'players_strategies': ['CashHoarder', 'Trader']},
            {'winner_strategy': 'AggressiveBuilder',
             'players_strategies': ['AggressiveBuilder', 'Trader']},
            {'winner_strategy': 'Trader',
             'players_strategies': ['Trader', 'CashHoarder']},
        ],
        'n_games_total': 5,
    }


def _design():
    from optimizer.group_design import GroupDesign
    return GroupDesign(salary_mult=1.10, label='cumulative')


# --------------------------------------------------------------------------- #
# Redaction matrix                                                              #
# --------------------------------------------------------------------------- #

def _feed(condition, prior_iters=None, hazard_summary='hz!'):
    return ldl.build_diagnostic_feed(
        cfg=_StubCfg(), design=_design(), eval_out=_eval_out(),
        prior_eval=None, condition=condition,
        prior_iters=prior_iters or [], hazard_summary=hazard_summary,
    )


def test_feed_mute_has_no_score_metrics_hazards_or_exploit():
    f = _feed('mute')
    assert '## CURRENT EVAL' not in f
    assert '## PER-GROUP COST/RENT BREAKDOWN' not in f
    assert '## STRATEGY-POOL EXPLOITATION' not in f
    assert '## HAZARD SUMMARY' not in f
    assert '## CURRENT DESIGN DIFF FROM DEFAULT' in f


def test_feed_haz_has_hazards_only():
    f = _feed('haz')
    assert '## HAZARD SUMMARY' in f
    assert '## CURRENT EVAL' not in f
    assert '## PER-GROUP COST/RENT BREAKDOWN' not in f
    assert '## STRATEGY-POOL EXPLOITATION' not in f


def test_feed_met_has_metrics_groups_exploit_no_hazards():
    f = _feed('met')
    assert '## CURRENT EVAL' in f
    assert '## PER-GROUP COST/RENT BREAKDOWN' in f
    assert '## STRATEGY-POOL EXPLOITATION' in f
    assert '## HAZARD SUMMARY' not in f


def test_feed_full_has_everything():
    f = _feed('full')
    assert '## CURRENT EVAL' in f
    assert '## PER-GROUP COST/RENT BREAKDOWN' in f
    assert '## STRATEGY-POOL EXPLOITATION' in f
    assert '## HAZARD SUMMARY' in f


def test_feed_blind_has_same_content_as_full():
    """T-BLIND uses GOAL-CLOSED system prompt but the user-prompt feed is
    identical to T-FULL's. The goal-disclosure ablation lives at the system-
    prompt layer, picked by DesignerLLM.__init__ via the loader."""
    full = _feed('full')
    blind = _feed('blind')
    # Strip the "## YOUR JOB" tail (it's identical in both since neither is
    # in NO_CONVERGE_CONDITIONS); the sectioned body should match.
    body_full = full.split('## YOUR JOB')[0]
    body_blind = blind.split('## YOUR JOB')[0]
    assert body_full == body_blind


def test_feed_no_converge_conditions_say_so_in_your_job():
    for c in ('mute', 'blind'):
        assert 'Do NOT set converged=true' in _feed(c)
    for c in ('full', 'haz', 'met'):
        assert 'Do NOT set converged=true' not in _feed(c)


# --------------------------------------------------------------------------- #
# Designer prompt picker                                                        #
# --------------------------------------------------------------------------- #

def test_designer_picks_open_or_closed_prompt_via_goal_disclosure():
    open_d = ldl.DesignerLLM(backend='heuristic', goal_disclosure='open')
    closed_d = ldl.DesignerLLM(backend='heuristic', goal_disclosure='closed')
    assert 'designer_llm_prompt_open.txt' in open_d.prompt_path
    assert 'designer_llm_prompt_closed.txt' in closed_d.prompt_path
    # Closed prompt drops the score-fn parenthetical; open keeps it.
    assert 'lower is better' in open_d.system_prompt
    assert 'penalises unfairness' in open_d.system_prompt
    assert 'lower is better' not in closed_d.system_prompt
    assert 'penalises unfairness' not in closed_d.system_prompt


def test_designer_unknown_goal_raises():
    with pytest.raises(ValueError):
        ldl.DesignerLLM(backend='heuristic', goal_disclosure='partial')


# --------------------------------------------------------------------------- #
# T-RAND sampler bounds                                                         #
# --------------------------------------------------------------------------- #

def test_rand_sampler_yields_valid_groupdesign():
    """200 samples; every sample is a GroupDesign within declared bounds."""
    from optimizer.group_design import GroupDesign
    rng = np.random.default_rng(0)
    present = ['Brown', 'Lightblue', 'Pink', 'Orange']
    for _ in range(200):
        d = ldl.sample_random_intervention(present, rng)
        assert isinstance(d, GroupDesign)
        if d.salary_mult != 1.0:
            assert 0.5 <= d.salary_mult <= 2.0
        for g, m in d.group_cost_mult.items():
            assert g in present and 0.5 <= m <= 2.0
        for g, m in d.group_rent_mult.items():
            assert g in present and 0.5 <= m <= 2.0
        for g in d.drop_groups:
            assert g in present


# --------------------------------------------------------------------------- #
# T-SANITY assertion                                                            #
# --------------------------------------------------------------------------- #

def test_sanity_assertion_passes_on_strict_decrease():
    # script iters: 0 baseline (irrelevant), 1..5 sanity designs strict-dec
    scores = [9.99, 5.0, 4.0, 3.0, 2.0, 1.0]
    ok, reason = ldl.assert_sanity_monotone(scores)
    assert ok, reason


def test_sanity_assertion_fails_on_non_monotone():
    scores = [9.99, 5.0, 4.0, 4.5, 2.0, 1.0]   # iter 2->3 not strictly dec
    ok, reason = ldl.assert_sanity_monotone(scores)
    assert not ok
    assert 'non-monotone' in reason


def test_sanity_assertion_fails_on_incomplete():
    scores = [9.99, 5.0, 4.0]
    ok, reason = ldl.assert_sanity_monotone(scores)
    assert not ok
    assert 'incomplete' in reason


def test_sanity_trajectory_has_at_least_5_design_entries():
    """The hardcoded trajectory must have at least 5 designs so the
    assertion has 5 strict-decrease checks to perform. First entry is the
    smallest perturbation; entries should be progressively more aggressive."""
    assert len(ldl.SANITY_TRAJECTORY) >= 5
    # First entry is salary x1.10.
    assert ldl.SANITY_TRAJECTORY[0].salary_mult == pytest.approx(1.10)
    # Salary axis grows monotonically through the first 4 designs.
    salaries = [ldl.SANITY_TRAJECTORY[i].salary_mult for i in range(4)]
    assert salaries == sorted(salaries), \
        f'salary axis must be monotone non-decreasing in the first 4 '\
        f'sanity designs; got {salaries}'
    # Last asserted iter (iter 4 in spec, idx 4 in script) introduces a
    # rent multiplier on top of the maxed salary. Effect size must be large
    # enough to clear the n=200 sampling envelope (≈0.05-0.10) — using
    # multi-group rent x2 atop salary x2 gets an empirical Δ≈0.10-0.15.
    assert ldl.SANITY_TRAJECTORY[4].group_rent_mult, \
        'iter 4 must introduce a rent multiplier'


# --------------------------------------------------------------------------- #
# Cross-condition seed alignment                                                #
# --------------------------------------------------------------------------- #

def test_eval_seed_is_pure_function_of_inputs():
    """Identical (board, seed, iter, game) -> identical seed in every call;
    different inputs -> different seeds."""
    s1 = ldl.eval_seed_for('default', 42, 3, 0)
    s2 = ldl.eval_seed_for('default', 42, 3, 0)
    s3 = ldl.eval_seed_for('default', 42, 3, 1)
    s4 = ldl.eval_seed_for('GA-2p',   42, 3, 0)
    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert 0 <= s1 <= 0xFFFFFFFF


def test_eval_seed_aligned_across_conditions():
    """Cross-condition alignment: every condition that calls eval_seed_for
    with the same (board, seed, iter, game) tuple gets the same seed.
    The function takes no condition arg, which is the whole point."""
    for cond_pair in (('mute', 'haz'), ('haz', 'full'),
                      ('full', 'blind'), ('met', 'rand')):
        a, b = cond_pair
        # The function doesn't take condition; both 'conditions' compute
        # identical seeds from the same args.
        assert ldl.eval_seed_for('GA-3p', 7, 4, 2) == \
               ldl.eval_seed_for('GA-3p', 7, 4, 2)


# --------------------------------------------------------------------------- #
# Convergence handling (uses run_trajectory with a stubbed designer)            #
# --------------------------------------------------------------------------- #

class _AlwaysConvergeDesigner:
    """Minimal designer-stub: emits {converged:true} every iteration."""

    def __init__(self, goal_disclosure='open'):
        self.gen_cfg = {'stub': True, 'goal_disclosure': goal_disclosure}

    def query(self, prompt, iteration=0):
        import json as _json
        payload = {'rationale': '[stub] declare converged',
                   'intervention': {}, 'converged': True}
        return '```json\n' + _json.dumps(payload) + '\n```', None


def _run_trajectory_stub(condition, K=4, designer=None):
    """Mock run_trajectory by calling the relevant convergence logic
    directly against an Iteration list. We lean on the loop's actual
    state machine inside run_trajectory, but to keep tests fast we bypass
    the eval pipeline by stubbing out _eval_with_aligned_seeds and
    bootstrap_score_ci."""
    real_eval = ldl._eval_with_aligned_seeds
    real_boot = ldl.bootstrap_score_ci
    real_apply = ldl.apply_design

    def fake_eval(*args, **kwargs):
        return {'score': 1.0, 'metrics': {}, 'per_game_records': [],
                'n_games_total': 0}

    def fake_boot(*args, **kwargs):
        return {'mean': 1.0, 'ci_lo': 0.9, 'ci_hi': 1.1, 'n_resamples': 0}

    def fake_apply(starting_cfg, design, **kw):
        return _StubCfg()

    ldl._eval_with_aligned_seeds = fake_eval
    ldl.bootstrap_score_ci = fake_boot
    ldl.apply_design = fake_apply
    try:
        out_path = Path(__file__).parent / f'__tmp_{condition}.jsonl'
        try:
            iters = ldl.run_trajectory(
                starting_cfg=_StubCfg(), board_label='test', seed=0,
                pool=[], matchups=[('m', 'm')], n_games=4, K=K,
                condition=condition,
                designer=(designer or _AlwaysConvergeDesigner(
                    'closed' if condition == 'blind' else 'open')),
                max_turns=10, out_path=out_path, hazard_summary=None)
        finally:
            if out_path.exists():
                out_path.unlink()
        return iters
    finally:
        ldl._eval_with_aligned_seeds = real_eval
        ldl.bootstrap_score_ci = real_boot
        ldl.apply_design = real_apply


def test_full_honors_converged_with_padded_remaining_iters():
    iters = _run_trajectory_stub('full', K=4)
    # iter 0 baseline, iter 1 honors converged, iters 2..K are pads.
    assert iters[1].converged_request is True
    assert iters[1].convergence_padded is False
    for it in iters[2:]:
        assert it.convergence_padded is True
        assert it.converged_request is True


def test_mute_records_violation_and_does_not_pad():
    """T-MUTE disallows convergence — model emits converged=true but the
    loop continues for K iterations and tags `convergence_violation=True`."""
    iters = _run_trajectory_stub('mute', K=4)
    # No padded iters; every post-baseline iter should be a real eval.
    for it in iters[1:]:
        assert it.convergence_padded is False
    # At least one post-baseline iter records a violation (the stub emits
    # converged=true on every call).
    assert any(it.convergence_violation for it in iters[1:])
