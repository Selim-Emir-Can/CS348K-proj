"""LLM-driven board-design optimisation.

Same DesignSpace + GA + objective as scripts/optimize_board.py, but with
all-LLM seats instead of the 30-strategy ParametricPlayer pool. Per the
project plan (Task 2): pop=8, generations=5 → 40 candidate boards, 5
seeds per candidate.

This is the "LLMs as designers" experiment. The headline question: do
boards optimised against LLM-only play look different from the boards
the rule-based GA produced?

Wall-time estimate: 40 candidates * 5 seeds * ~30 LLM calls/game *
~8 s/call ≈ 13 hours on a single 11 GB GPU. Use tmux/nohup if launching
remote.

Decision logging is enabled only for the BEST candidate per generation
(written to logs/optimizer_llm/.../decisions/gen<N>/seed_<S>.jsonl) so
disk usage stays bounded — full per-eval logging would be GBs.

Usage (from monopoly/):
    python scripts/optimize_board_llm.py --n-players 2 \
        --pop 8 --generations 5 --n-seeds 5 \
        --model-name models/qwen2.5-1.5B \
        --run-name llm_ga_2p

Smoke (single GA generation, 1 seed; ~6 minutes):
    python scripts/optimize_board_llm.py --n-players 2 \
        --pop 4 --generations 1 --n-seeds 1 \
        --run-name llm_ga_smoke
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Make ``monopoly/`` importable when run as a script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from agents import LLMPlayer
from config import GameConfig
from optimizer.design_space import DesignSpace
from optimizer.objectives import Targets, Weights, evaluate
from optimizer.search import genetic_algorithm
from optimizer.simulate import run_single_game
from player_settings import StandardPlayerSettings


# --------------------------------------------------------------------------- #
# Model warm-up (mirrors eval_llm_on_boards.py)                                  #
# --------------------------------------------------------------------------- #

def warmup_model_or_die(model_name: str | None) -> None:
    """Load the LLM and run one tiny generation before the GA starts.

    Without this the silent fallback in ``LLMPlayer._should_buy_logged``
    would let the GA "succeed" on degenerate BUY-only data — ~13h
    wasted before anyone notices.
    """
    print(f'[warmup] loading {model_name or os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")} ...',
          flush=True)
    probe = LLMPlayer('warmup', StandardPlayerSettings(),
                      backend='local', model_name=model_name)
    try:
        out = probe._query_local(
            "STATE:\n  cash: $1500\n  property: B1 Test\n  group: Lightblue\n"
            "  cost: $100\n  base_rent: $6\n  you_own_total: 0\n"
            "  you_own_in_group: 0\n  group_size: 3\n  opp_own_in_group: 0\n"
            "  opp_own_total: 0")
    except Exception as exc:
        print(f'[warmup] FAILED: {exc!r}', file=sys.stderr)
        sys.exit(2)
    print(f'[warmup] OK; sample response: {out[:120]!r}', flush=True)


# --------------------------------------------------------------------------- #
# Eval fn: decode → run n_seeds all-LLM games → score                           #
# --------------------------------------------------------------------------- #

def _build_eval_fn(base_cfg, n_players, n_seeds, base_seed, weights, targets,
                   max_turns, model_name, decisions_root: Path | None):
    """Return (eval_fn(vec), space, state).

    ``state`` is a mutable dict the caller uses to track which generation
    we're in and to decide whether to enable decision logging on the
    candidate currently being evaluated. (We only log decisions for the
    best of each generation, identified post-hoc; here we just record
    every JSONL into a temp path and re-link the best-of-gen later.)

    For Task 2 we keep things simple: log EVERY candidate's games into
    candidate-scoped subdirs, then have the post-hoc analyser pick the
    best per gen. Disk: 40 candidates × 5 games × ~5 KB/JSONL ≈ 1 MB.
    Trivial.
    """
    space = DesignSpace(base_cfg)
    state = {'eval_idx': 0}

    def eval_fn(vec):
        eval_idx = state['eval_idx']
        state['eval_idx'] += 1

        cfg = space.decode(vec)
        results = []
        eval_dir = (decisions_root / f'eval_{eval_idx:03d}'
                     if decisions_root is not None else None)
        if eval_dir is not None:
            eval_dir.mkdir(parents=True, exist_ok=True)
        seat_names = [f'LLM_p{i}' for i in range(n_players)]
        settings = StandardPlayerSettings()
        players_spec = [(name, settings, 'LLMPlayer') for name in seat_names]

        # Inner bar: per-seed liveness across the 5 games.
        inner = tqdm(total=n_seeds, desc=f'  eval {eval_idx:03d}', unit='game',
                     leave=False, dynamic_ncols=True, position=1)
        for s_idx in range(n_seeds):
            seed = base_seed + s_idx
            common_meta = {
                'eval_idx':   eval_idx,
                'seed':       seed,
                'n_players':  n_players,
                'game_id':    f'eval{eval_idx:03d}_s{seed}',
                'model_name': model_name,
            }
            log_path = (eval_dir / f'seed_{seed}.jsonl'
                        if eval_dir is not None else None)
            player_kwargs_list = []
            for i, name in enumerate(seat_names):
                player_kwargs_list.append({
                    'backend':            'local',
                    'model_name':         model_name,
                    'decision_log_path':  str(log_path) if log_path else None,
                    'decision_log_meta':  {**common_meta, 'seat_idx': i,
                                            'seat_name': name},
                })

            def _on_turn(turn_n, players, transfer_so_far, _bar=inner):
                if turn_n % 1 == 0:
                    _bar.set_postfix(seed=seed, turn=turn_n,
                                     xfer=int(transfer_so_far),
                                     refresh=True)

            r = run_single_game(
                cfg, players_spec, seed=seed, max_turns=max_turns,
                player_kwargs_list=player_kwargs_list, on_turn=_on_turn)
            # objectives.fairness_within_matchup reads strategy_names; in
            # rule-based runs that's the strategy name (e.g. AggressiveBuilder),
            # but here every seat is the same LLMPlayer, so we use seat names.
            # Fairness across identical seats measures seat-position bias —
            # a meaningful design quality on its own.
            r['strategy_names'] = list(r['player_names'])
            results.append(r)
            inner.update(1)
        inner.close()

        # Wrap as a single matchup so we can reuse evaluate().
        out = evaluate([results], weights=weights, targets=targets)
        # Carry the per-seed game stats forward in 'extra' so the JSONL has
        # them for post-hoc analysis.
        out['per_game'] = [{k: v for k, v in r.items() if k != 'strategy_names'}
                           for r in results]
        return out['score'], {**out['metrics'],
                              'per_game_rounds': [r['rounds'] for r in results],
                              'per_game_winner': [r['winner'] for r in results]}

    return eval_fn, space, state


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',     default='default_config.yaml')
    ap.add_argument('--n-players',  type=int, default=2, choices=(2, 3))
    ap.add_argument('--pop',         type=int, default=8)
    ap.add_argument('--generations', type=int, default=5)
    ap.add_argument('--elitism',     type=int, default=2)
    ap.add_argument('--n-seeds',     type=int, default=5,
                    help='Games per candidate. Each seed plays one all-LLM game.')
    ap.add_argument('--max-turns',   type=int, default=200)
    ap.add_argument('--base-seed',   type=int, default=42)
    ap.add_argument('--search-seed', type=int, default=0)
    # Objective weights (same defaults as the rule-based GA)
    ap.add_argument('--w-fair',     type=float, default=1.0)
    ap.add_argument('--w-fmax',     type=float, default=0.5)
    ap.add_argument('--w-len',      type=float, default=0.5)
    ap.add_argument('--w-draw',     type=float, default=0.3)
    ap.add_argument('--w-money',    type=float, default=0.3)
    ap.add_argument('--target-rounds',   type=float, default=60.0)
    ap.add_argument('--target-transfer', type=float, default=100.0)
    # Model
    ap.add_argument('--model-name', default='models/qwen2.5-1.5B',
                    help='Local Qwen path or HF repo id.')
    # Output
    ap.add_argument('--run-name',   default=None)
    ap.add_argument('--out-dir',    default='logs/optimizer_llm')
    ap.add_argument('--no-decision-log', action='store_true',
                    help='Skip per-decision JSONL writes (smaller disk, faster '
                         'eval). Default keeps logging for downstream analysis.')
    args = ap.parse_args()

    # Output paths first so we can name decision logs by run.
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or f'llm_ga_{args.n_players}p_{ts}'
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path  = out_dir / 'evals.jsonl'
    meta_path = out_dir / 'meta.json'
    best_path = out_dir / 'best_design.json'
    decisions_root = (None if args.no_decision_log
                      else out_dir / 'decisions')

    # Make sure the model loads before we burn ~13h on doomed candidates.
    warmup_model_or_die(args.model_name)

    cfg = GameConfig.from_yaml(args.config)
    weights = Weights(args.w_fair, args.w_fmax, args.w_len, args.w_draw, args.w_money)
    targets = Targets(args.target_rounds, args.target_transfer)

    eval_fn, space, state = _build_eval_fn(
        cfg, args.n_players, args.n_seeds, args.base_seed, weights, targets,
        max_turns=args.max_turns, model_name=args.model_name,
        decisions_root=decisions_root)

    # Meta file (so the report can reproduce any number).
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'run_name':      run_name,
            'search':        'ga_llm',
            'n_players':     args.n_players,
            'n_seeds':       args.n_seeds,
            'max_turns':     args.max_turns,
            'pop':           args.pop,
            'generations':   args.generations,
            'elitism':       args.elitism,
            'weights':       vars(weights),
            'targets':       vars(targets),
            'base_seed':     args.base_seed,
            'search_seed':   args.search_seed,
            'config':        args.config,
            'model_name':    args.model_name,
            'decision_log':  not args.no_decision_log,
        }, f, indent=2)
    print(f'Meta written to {meta_path}')

    n_total = args.pop + (args.generations - 1) * (args.pop - args.elitism)
    pbar = tqdm(total=n_total, desc='ga_llm', dynamic_ncols=True, position=0)
    best = [float('inf')]
    best_entry = [None]
    fh = open(run_path, 'w', encoding='utf-8')

    def on_iter(entry):
        fh.write(json.dumps(entry) + '\n')
        fh.flush()
        if entry['score'] < best[0]:
            best[0] = entry['score']
            best_entry[0] = entry
            with open(best_path, 'w', encoding='utf-8') as bf:
                json.dump(entry, bf, indent=2)
        pbar.set_postfix(best=f"{best[0]:.3f}", cur=f"{entry['score']:.3f}")
        pbar.update(1)

    history = genetic_algorithm(
        space, eval_fn,
        pop_size=args.pop, generations=args.generations,
        elitism=args.elitism, seed=args.search_seed,
        on_iter=on_iter,
    )
    fh.close()
    pbar.close()
    print(f'\n=== Done. {len(history)} evaluations. Best score: {best[0]:.3f} ===')
    print(f'Best design vec written to {best_path}')


if __name__ == '__main__':
    main()
