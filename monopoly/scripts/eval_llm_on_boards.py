"""Evaluate one or more board designs under all-LLM seats; log every decision.

For each (board, n_players) cell we play a fixed number of seeded games where
*every* seat is an ``LLMPlayer``. Per-game metrics (winner, rounds, draw,
money-transfer-rate, n_llm_calls, n_prefiltered) are aggregated into a
``summary.csv``; per-decision JSONL records (prompt, raw response, parse path,
timing, gen_meta) are emitted to ``decisions/<board_tag>/seed_<k>.jsonl``.

Why this script exists: Task 1 of the LLM-eval branch — quantify how the LLM
plays on the GA-winner board vs. the default board, and gather text artefacts
of the LLM's decision process so we can examine its reasoning.

Default matrix (matches the project plan):
  --boards default ga_2p_winner       --n-players 2  --n-seeds 20
  --boards default ga_3p_winner       --n-players 3  --n-seeds 20

Boards are looked up by tag:
  - ``default``        : identity vec on the supplied --config (default_config.yaml).
  - ``ga_2p_winner``   : best (lowest ``score``) entry from --ga-2p-jsonl.
  - ``ga_3p_winner``   : best entry from --ga-3p-jsonl.

Seed schedule is shared across boards within a run (CRN), so a board change is
the only difference between two same-seed games.

Example invocation (run from monopoly/):
    python scripts/eval_llm_on_boards.py \
        --config default_config.yaml \
        --ga-2p-jsonl logs/optimizer/ga_2p_mask.jsonl \
        --ga-3p-jsonl logs/optimizer/ga_3p_mask.jsonl \
        --n-seeds 20 \
        --out-dir logs/llm_eval/run1

For a 1-game smoke test:
    python scripts/eval_llm_on_boards.py --boards default --n-players 2 \
        --n-seeds 1 --out-dir logs/llm_eval/smoke
"""
import argparse
import csv
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

from config import GameConfig
from agents import LLMPlayer
from optimizer.design_space import DesignSpace
from optimizer.simulate import run_single_game
from player_settings import StandardPlayerSettings


# --------------------------------------------------------------------------- #
# Model warm-up                                                                 #
# --------------------------------------------------------------------------- #

def warmup_model_or_die(model_name: str | None) -> None:
    """Load the LLM (and run a 1-token generation) before the first game.

    Catches the ``FileNotFoundError(2)`` symptom we saw on a partial HF
    download (84 MB present, ~3 GB expected): without this, the per-game
    try/except in ``LLMPlayer._should_buy_logged`` silently falls back to
    'BUY' on every decision and the eval reports look fine while producing
    zero real LLM signal. Failing here gives the operator a single clear
    error before any GPU time is spent.
    """
    print(f'[warmup] loading {model_name or os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")} ...',
          flush=True)
    probe = LLMPlayer('warmup', StandardPlayerSettings(),
                      backend='local', model_name=model_name)
    try:
        # _query_local does the same path the real decisions will take.
        out = probe._query_local("Cash $1500. Decision: buy 'B1 Test' (group "
                                 "Lightblue) for $100, base rent $6. You own 0 "
                                 "properties total, 0 in this group. Opponent "
                                 "owns 0 in this group, 0 properties total.")
    except Exception as exc:
        print(f'[warmup] FAILED to load/generate from the model: {exc!r}',
              file=sys.stderr)
        print('[warmup] hint: check that the HF cache has complete model '
              'weights (delete any *.incomplete blobs under '
              'models/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/blobs '
              'and re-run with internet).', file=sys.stderr)
        sys.exit(2)
    print(f'[warmup] OK; sample response: {out[:120]!r}', flush=True)


# --------------------------------------------------------------------------- #
# Board loading                                                                 #
# --------------------------------------------------------------------------- #

def _best_vec_from_jsonl(path: str):
    """Return (vec, score, iter, full_record) from the lowest-score line."""
    best = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if 'score' not in e or 'vec' not in e:
                continue
            if best is None or e['score'] < best['score']:
                best = e
    if best is None:
        raise ValueError(f'No scored entries in {path}')
    return best


def load_board(tag: str, base_cfg, args):
    """Resolve a board tag to a (decoded_cfg, source_metadata) pair."""
    space = DesignSpace(base_cfg)
    if tag == 'default':
        vec = space.identity_vec()
        meta = {'tag': 'default', 'source': 'identity_vec'}
        return space.decode(vec), vec.tolist(), meta
    if tag == 'ga_2p_winner':
        if not args.ga_2p_jsonl:
            raise ValueError('--ga-2p-jsonl required to resolve ga_2p_winner')
        rec = _best_vec_from_jsonl(args.ga_2p_jsonl)
        vec = np.asarray(rec['vec'])
        meta = {'tag': 'ga_2p_winner', 'source': args.ga_2p_jsonl,
                'iter': rec.get('iter'), 'score': rec.get('score')}
        return space.decode(vec), vec.tolist(), meta
    if tag == 'ga_3p_winner':
        if not args.ga_3p_jsonl:
            raise ValueError('--ga-3p-jsonl required to resolve ga_3p_winner')
        rec = _best_vec_from_jsonl(args.ga_3p_jsonl)
        vec = np.asarray(rec['vec'])
        meta = {'tag': 'ga_3p_winner', 'source': args.ga_3p_jsonl,
                'iter': rec.get('iter'), 'score': rec.get('score')}
        return space.decode(vec), vec.tolist(), meta
    raise ValueError(f'unknown board tag: {tag}')


# --------------------------------------------------------------------------- #
# One game                                                                      #
# --------------------------------------------------------------------------- #

def run_one_llm_game(cfg, n_players: int, seed: int, board_tag: str,
                     decisions_dir: Path, max_turns: int, model_name: str,
                     game_bar=None, progress_every_turns: int = 1):
    """Play a single game with all-LLM seats and a per-seat decision log.

    If ``game_bar`` is a tqdm instance, its postfix is updated each turn
    with the current round number and player-to-player money transfer
    total — same liveness pattern phase_a_validation.py uses, so games
    that take 5+ minutes still show progress in the outer bar.
    """
    settings = StandardPlayerSettings()
    seat_names = [f'LLM_p{i}' for i in range(n_players)]
    players_spec = [(name, settings, 'LLMPlayer') for name in seat_names]

    # One JSONL file per game; both seats append to the same file so the log
    # shows the full conversation between LLM agents in time-order.
    log_path = decisions_dir / f'seed_{seed}.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    common_meta = {
        'board_tag':   board_tag,
        'seed':        seed,
        'n_players':   n_players,
        'game_id':     f'{board_tag}_n{n_players}_s{seed}',
        'model_name':  model_name,
    }
    player_kwargs_list = []
    for i, name in enumerate(seat_names):
        player_kwargs_list.append({
            'backend':            'local',
            'model_name':         model_name,
            'decision_log_path':  str(log_path),
            'decision_log_meta':  {**common_meta, 'seat_idx': i,
                                    'seat_name': name},
        })

    # Per-turn liveness callback: refreshes the outer tqdm postfix with
    # the current turn number. With ECHO+retries each LLM call is ~8 s so
    # one game can take a couple of minutes; this keeps the bar moving.
    def _on_turn(turn_n, players, transfer_so_far, _bar=game_bar):
        if _bar is None:
            return
        if turn_n % progress_every_turns == 0:
            _bar.set_postfix(turn=turn_n, xfer=int(transfer_so_far),
                              refresh=True)

    t0 = time.perf_counter()
    result = run_single_game(
        cfg, players_spec, seed=seed, max_turns=max_turns,
        player_kwargs_list=player_kwargs_list,
        on_turn=_on_turn if game_bar is not None else None,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    # Count log records produced by this game so summary.csv has a quick stat.
    n_decisions = 0
    n_prefiltered = 0
    n_llm_calls = 0
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get('game_id') != common_meta['game_id']:
                    continue
                n_decisions += 1
                if rec.get('prefilter') == 'sent_to_llm':
                    n_llm_calls += 1
                else:
                    n_prefiltered += 1

    rounds = max(1, result['rounds'])
    return {
        'board_tag':       board_tag,
        'n_players':       n_players,
        'seed':            seed,
        'game_id':         common_meta['game_id'],
        'winner':          result['winner'] or '',
        'rounds':          result['rounds'],
        'truncated':       int(result['truncated']),
        'transfer_total':  result['transfer_total'],
        'transfer_rate':   result['transfer_total'] / rounds,
        'net_worth':       json.dumps(result['net_worth']),
        'bankrupt':        json.dumps(result['bankrupt']),
        'n_decisions':     n_decisions,
        'n_llm_calls':     n_llm_calls,
        'n_prefiltered':   n_prefiltered,
        'wall_ms':         wall_ms,
        'log_path':        str(log_path),
    }


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='default_config.yaml',
                    help='Base GameConfig (used both for "default" and as the '
                         'DesignSpace base when decoding GA-winner vecs).')
    ap.add_argument('--boards', nargs='+',
                    default=['default', 'ga_2p_winner'],
                    help='Board tags to evaluate. Choices: default, '
                         'ga_2p_winner, ga_3p_winner.')
    ap.add_argument('--n-players', type=int, default=2)
    ap.add_argument('--n-seeds', type=int, default=20,
                    help='Games per board (each uses a unique seed).')
    ap.add_argument('--base-seed', type=int, default=42)
    ap.add_argument('--max-turns', type=int, default=200)
    ap.add_argument('--ga-2p-jsonl', default='logs/optimizer/ga_2p_mask.jsonl')
    ap.add_argument('--ga-3p-jsonl', default='logs/optimizer/ga_3p_mask.jsonl')
    ap.add_argument('--model-name', default=None,
                    help='Override default LLM model. Default: env LLM_MODEL '
                         'or Qwen/Qwen2.5-1.5B-Instruct.')
    ap.add_argument('--out-dir', default='logs/llm_eval/run1')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_root = out_dir / 'decisions'

    base_cfg = GameConfig.from_yaml(args.config)

    # Resolve every board up front so failures surface before we start playing.
    resolved = []
    for tag in args.boards:
        cfg, vec, meta = load_board(tag, base_cfg, args)
        resolved.append((tag, cfg, vec, meta))

    # Make sure the LLM can actually load + generate before we burn any time.
    warmup_model_or_die(args.model_name)

    # Persist the run-level metadata so the analyser doesn't have to re-derive.
    run_meta = {
        'config':           args.config,
        'n_players':        args.n_players,
        'n_seeds':          args.n_seeds,
        'base_seed':        args.base_seed,
        'max_turns':        args.max_turns,
        'model_name':       args.model_name or os.environ.get(
            'LLM_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct'),
        'boards':           [{'tag': t, 'vec': v, 'source': m}
                             for t, _, v, m in resolved],
    }
    with open(out_dir / 'run_meta.json', 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, indent=2)

    summary_rows = []
    summary_path = out_dir / 'summary.csv'
    summary_fields = [
        'board_tag', 'n_players', 'seed', 'game_id', 'winner', 'rounds',
        'truncated', 'transfer_total', 'transfer_rate', 'net_worth',
        'bankrupt', 'n_decisions', 'n_llm_calls', 'n_prefiltered',
        'wall_ms', 'log_path',
    ]
    with open(summary_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()

    # Outer bar over total games across every (board, seed) pair. The
    # description switches to "<tag> seed=<k>" each time we move to a new
    # game; the postfix shows current turn + transfer total via on_turn.
    total_games = len(resolved) * args.n_seeds
    with tqdm(total=total_games, desc='games', unit='game',
              dynamic_ncols=True) as game_bar:
        for tag, cfg, vec, meta in resolved:
            decisions_dir = decisions_root / tag
            for k in range(args.n_seeds):
                seed = args.base_seed + k
                game_bar.set_description(f'{tag} seed={seed}', refresh=False)
                game_bar.set_postfix(turn=0, xfer=0, refresh=True)
                row = run_one_llm_game(
                    cfg, args.n_players, seed, tag, decisions_dir,
                    args.max_turns, run_meta['model_name'],
                    game_bar=game_bar)
                summary_rows.append(row)
                with open(summary_path, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=summary_fields)
                    writer.writerow(row)
                game_bar.write(
                    f'[{tag}] seed={seed}  '
                    f'winner={row["winner"]!r:>10}  rounds={row["rounds"]:>4}  '
                    f'decisions={row["n_decisions"]:>3}  '
                    f'llm_calls={row["n_llm_calls"]:>3}  '
                    f'wall_s={row["wall_ms"]/1000.0:.1f}')
                game_bar.update(1)

    print(f'\n=== Done. Wrote {len(summary_rows)} rows to {summary_path} ===')


if __name__ == '__main__':
    main()
