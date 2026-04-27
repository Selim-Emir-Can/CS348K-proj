"""Subprocess sandbox for evaluating LLM-emitted rule patches.

CEO plan #5 mandates per-game subprocess isolation with a 10s wall-clock cap.
The motivation is two-fold:
  - A patch that creates infinite money or breaks a termination condition
    can run a single game forever — the wall-clock cap kills it.
  - A patch that crashes the engine (uncaught exception, recursion) bubbles
    up as a non-zero exit code and is recorded as a rejection.

Implementation choices
----------------------
- We do NOT rewrite source files on disk. Patches are STRUCTURED (see
  optimizer.rule_patches) and applied at runtime to a deepcopied GameConfig.
  That means the sandbox subprocess just receives a serialised
  (cfg_path, patches_json, run_args) bundle on stdin and runs games.
- Subprocess uses the SAME Python interpreter as the parent (sys.executable)
  so we don't pay venv-discovery cost per game.
- One subprocess per (design, board, n_games) cell, not per individual
  game. The 10s cap is enforced PER GAME inside the subprocess via a worker
  thread + signal-style cooperative timer (signals don't work cleanly on
  Windows, so we use a watchdog thread).

Returns
-------
A SandboxResult that aggregates the per-game outcomes (or records that the
sandbox exited abnormally), and is JSON-serialisable for the trajectory log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# Path to the worker script (sibling file, written below).
_WORKER_PATH = Path(__file__).with_name('_rule_sandbox_worker.py')


@dataclass
class SandboxResult:
    ok:                bool
    n_games_requested: int
    n_games_completed: int
    aggregate_score:   Optional[float]
    metrics:           Dict[str, Any] = field(default_factory=dict)
    per_game:          List[dict] = field(default_factory=list)
    failure_reason:    Optional[str] = None
    wall_seconds:      float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def run_sandboxed(cfg_yaml_path: str,
                  patches: List[dict],
                  pool_path: str,
                  matchups: List[List[int]],
                  n_games: int,
                  base_seed: int,
                  max_turns: int,
                  per_game_timeout_seconds: float = 10.0,
                  total_timeout_seconds: Optional[float] = None) -> SandboxResult:
    """Run `n_games` games on (cfg + patches) in a child process.

    Args:
      cfg_yaml_path:   GameConfig YAML to load (canonical or mini)
      patches:         list of structured patches (see optimizer.rule_patches)
      pool_path:       optimizer/strategy_pool.json
      matchups:        list of strategy-pool index tuples (one per matchup)
      n_games:         total games requested across all matchups
      base_seed:       reproducibility
      max_turns:       hard turn cap per game (defensive against monotone games)
      per_game_timeout_seconds:  CEO-pinned; default 10s
      total_timeout_seconds:     optional outer cap; defaults to
                                 n_games * per_game_timeout_seconds + 30s grace
    """
    if total_timeout_seconds is None:
        total_timeout_seconds = n_games * per_game_timeout_seconds + 30.0

    payload = {
        'cfg_yaml_path':              str(cfg_yaml_path),
        'patches':                    patches,
        'pool_path':                  str(pool_path),
        'matchups':                   [list(m) for m in matchups],
        'n_games':                    int(n_games),
        'base_seed':                  int(base_seed),
        'max_turns':                  int(max_turns),
        'per_game_timeout_seconds':   float(per_game_timeout_seconds),
    }

    t0 = time.perf_counter()
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = (str(Path(__file__).resolve().parent.parent)
                              + os.pathsep + env.get('PYTHONPATH', ''))
        proc = subprocess.run(
            [sys.executable, str(_WORKER_PATH)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=total_timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            ok=False, n_games_requested=n_games, n_games_completed=0,
            aggregate_score=None,
            failure_reason=f'subprocess wall-clock exceeded {total_timeout_seconds:.1f}s',
            wall_seconds=time.perf_counter() - t0,
        )

    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return SandboxResult(
            ok=False, n_games_requested=n_games, n_games_completed=0,
            aggregate_score=None,
            failure_reason=(f'subprocess returncode={proc.returncode}; '
                             f'stderr_tail={proc.stderr[-400:]!r}'),
            wall_seconds=wall,
        )

    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as ex:
        return SandboxResult(
            ok=False, n_games_requested=n_games, n_games_completed=0,
            aggregate_score=None,
            failure_reason=f'malformed worker stdout: {ex}; tail={proc.stdout[-400:]!r}',
            wall_seconds=wall,
        )

    return SandboxResult(
        ok=bool(out.get('ok', False)),
        n_games_requested=n_games,
        n_games_completed=int(out.get('n_games_completed', 0)),
        aggregate_score=out.get('aggregate_score'),
        metrics=out.get('metrics', {}),
        per_game=out.get('per_game', []),
        failure_reason=out.get('failure_reason'),
        wall_seconds=wall,
    )
