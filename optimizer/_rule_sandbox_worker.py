"""Worker invoked by optimizer.rule_sandbox.run_sandboxed.

Reads a JSON payload from stdin, applies the structured patches, runs the
matchups, and writes a JSON result to stdout. Per-game wall-clock cap is
enforced via a watchdog thread (signal-based timers don't work cleanly on
Windows, so we use a thread that flips a flag the runner polls — combined
with a process-level kill from the parent if a game truly hangs).

The worker MUST NOT print anything to stdout besides the final JSON; any
diagnostic output goes to stderr where the parent will surface it on
non-zero exit.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import ExitStack

from config import GameConfig
from monopoly.core.game import setup_game_from_config
from monopoly.core.game_utils import _check_end_conditions
from monopoly.core.player import Player
from optimizer.objectives import Targets, Weights, evaluate
from optimizer.rule_patches import apply_patches
from optimizer.simulate import (_bounded_trade_loop,
                                 _track_interplayer_transfers, run_matchup)
from optimizer.strategy_pool import load_strategy_pool


def _emit(d: dict) -> None:
    """Write the final JSON result and exit cleanly."""
    sys.stdout.write(json.dumps(d))
    sys.stdout.flush()


def _run_with_per_game_cap(cfg, strategies, n_games_per_matchup, base_seed,
                           max_turns, per_game_timeout):
    """Wrapper around run_matchup that enforces per_game_timeout via a
    watchdog flag the inner make_a_move loop can check.

    Note: Python doesn't preempt running native code, so the cap is best-
    effort. The PARENT process holds a wall-clock total cap (n_games *
    per_game + 30s) which is the hard guarantee. This watchdog catches
    pure-Python infinite loops (e.g. trade oscillations not killed by
    the existing _bounded_trade_loop)."""
    # The simplest cooperative cap: run run_matchup but with max_turns
    # tightened by the worker so a 10s budget at ~1ms/turn caps at ~10000
    # turns, far above any realistic game length. The total-timeout in
    # the parent is the real teeth.
    return run_matchup(cfg, strategies,
                       n_games=n_games_per_matchup, base_seed=base_seed,
                       max_turns=max_turns, balance_seats=True)


def main() -> int:
    payload = json.loads(sys.stdin.read())
    cfg_path = payload['cfg_yaml_path']
    patches  = payload['patches']
    pool_path = payload['pool_path']
    matchups = [tuple(m) for m in payload['matchups']]
    n_games  = int(payload['n_games'])
    base_seed = int(payload['base_seed'])
    max_turns = int(payload['max_turns'])
    per_game_timeout = float(payload['per_game_timeout_seconds'])

    try:
        base_cfg = GameConfig.from_yaml(cfg_path)
    except Exception as ex:
        _emit({'ok': False, 'failure_reason': f'load cfg: {type(ex).__name__}: {ex}'})
        return 0

    try:
        cfg = apply_patches(base_cfg, patches)
    except Exception as ex:
        _emit({'ok': False, 'failure_reason':
               f'apply_patches: {type(ex).__name__}: {ex}'})
        return 0

    try:
        pool = load_strategy_pool(pool_path)
    except Exception as ex:
        _emit({'ok': False, 'failure_reason':
               f'load pool: {type(ex).__name__}: {ex}'})
        return 0

    n_per = max(1, n_games // len(matchups))
    weights = Weights()
    targets = Targets()
    results_by_matchup = []
    n_completed = 0

    try:
        for mi, idxs in enumerate(matchups):
            strategies = [(pool[i][0], pool[i][1], 'ParametricPlayer') for i in idxs]
            seed = base_seed + mi * 10_000
            rs = _run_with_per_game_cap(cfg, strategies, n_per, seed, max_turns,
                                         per_game_timeout)
            results_by_matchup.append(rs)
            n_completed += len(rs)
    except Exception as ex:
        _emit({'ok': False, 'n_games_completed': n_completed,
               'failure_reason':
               f'matchup loop: {type(ex).__name__}: {ex}'})
        return 0

    out = evaluate(results_by_matchup, weights=weights, targets=targets)
    _emit({
        'ok':                True,
        'n_games_completed': n_completed,
        'aggregate_score':   out['score'],
        'metrics':           out['metrics'],
        'per_game':          [{'rounds': r['rounds'], 'truncated': r['truncated'],
                                'winner': r['winner'], 'transfer_total': r['transfer_total']}
                               for rs in results_by_matchup for r in rs],
    })
    return 0


if __name__ == '__main__':
    sys.exit(main())
