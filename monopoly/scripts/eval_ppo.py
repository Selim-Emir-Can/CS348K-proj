"""Evaluate a RecurrentPPO agent in a 2-player tournament vs RuleBased.

Also supports a --self-play-baseline mode that measures the seat-order noise
floor (RuleBased vs RuleBased, both seats). That baseline defines the "50% ±
noise" CI against which the IL sanity-check and PPO fine-tune are compared.

Debugging: pass --save-trace <path> to dump per-game details (final cash,
net-worth, owned properties, DQN action history, winner) as JSONL, one game
per line. Pass --verbose-game <n> to also print a detailed human-readable
summary of the first <n> games to stdout.

Usage (from monopoly/):
    python scripts/eval_ppo.py                                # il_checkpoint vs RuleBased
    python scripts/eval_ppo.py --model models/ppo_best --n 1000
    python scripts/eval_ppo.py --self-play-baseline --n 1000  # RB-vs-RB noise floor
    python scripts/eval_ppo.py --action-dist                  # print DQN action histogram
    python scripts/eval_ppo.py --save-trace eval_trace.jsonl --verbose-game 5
"""
import argparse
import json
import math
import os

import numpy as np
from sb3_contrib import RecurrentPPO
from tqdm import tqdm

from config import GameConfig
from monopoly_env import HistorySingleAgentMonopolyEnv


_ACTION_LABELS = ['no-buy/hold', 'buy/hold', 'no-buy/mod',
                  'buy/mod', 'no-buy/aggr', 'buy/aggr']


def aggregate_debug_stats(snapshots, dqn_name: str = 'DQN',
                          rb_name: str = 'RuleBased') -> dict:
    """Compute debug metrics from a list of per-game snapshots (from _snapshot_game).

    Returns scalars suitable for printing or logging to W&B:
      avg_rounds, truncation_rate, dqn_bankruptcy_rate, rb_bankruptcy_rate,
      avg_dqn_nw, avg_rb_nw, avg_dqn_props, avg_dqn_monopolies,
      action_entropy (bits over the 6 actions), action_counts (list of 6).
    """
    if not snapshots:
        return {}
    n = len(snapshots)
    act_totals = np.zeros(6, dtype=np.int64)
    rounds, trunc, dqn_bk, rb_bk = 0, 0, 0, 0
    dqn_nw, rb_nw, dqn_props, dqn_mono = 0, 0, 0, 0
    for s in snapshots:
        rounds += s['rounds']
        trunc  += int(s['truncated'])
        p_dqn = s['players'].get(dqn_name, {})
        p_rb  = s['players'].get(rb_name,  {})
        dqn_bk    += int(p_dqn.get('bankrupt', False))
        rb_bk     += int(p_rb.get('bankrupt',  False))
        dqn_nw    += p_dqn.get('net_worth', 0)
        rb_nw     += p_rb.get('net_worth',  0)
        dqn_props += p_dqn.get('n_properties', 0)
        dqn_mono  += p_dqn.get('n_monopolies', 0)
        act_totals += np.array(s['action_counts'], dtype=np.int64)
    total_actions = int(act_totals.sum())
    if total_actions > 0:
        p = act_totals / total_actions
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = float(-np.sum(np.where(p > 0, p * np.log2(p), 0.0)))
        entropy = max(entropy, 0.0)   # clamp -0.0 rounding artefact
    else:
        entropy = 0.0
    return {
        'avg_rounds':            rounds / n,
        'truncation_rate':       trunc  / n,
        'dqn_bankruptcy_rate':   dqn_bk / n,
        'rb_bankruptcy_rate':    rb_bk  / n,
        'avg_dqn_net_worth':     dqn_nw / n,
        'avg_rb_net_worth':      rb_nw  / n,
        'avg_dqn_properties':    dqn_props / n,
        'avg_dqn_monopolies':    dqn_mono  / n,
        'action_entropy_bits':   entropy,
        'action_counts':         act_totals.tolist(),
        'total_actions':         total_actions,
    }


def format_debug_stats(stats: dict) -> str:
    """Human-readable one-block summary of aggregate_debug_stats output."""
    if not stats:
        return '(no snapshots)'
    lines = [
        f"  avg_rounds={stats['avg_rounds']:.1f}  "
        f"trunc={100*stats['truncation_rate']:.1f}%  "
        f"DQN_bkrpt={100*stats['dqn_bankruptcy_rate']:.1f}%  "
        f"RB_bkrpt={100*stats['rb_bankruptcy_rate']:.1f}%",
        f"  avg_nw: DQN=${stats['avg_dqn_net_worth']:.0f}  "
        f"RB=${stats['avg_rb_net_worth']:.0f}  "
        f"avg_DQN_props={stats['avg_dqn_properties']:.1f}  "
        f"avg_DQN_mono={stats['avg_dqn_monopolies']:.2f}",
        f"  action entropy={stats['action_entropy_bits']:.3f} bits  "
        f"(0=constant, log2(6)={np.log2(6):.2f}=uniform)",
    ]
    counts = stats['action_counts']
    total  = max(stats['total_actions'], 1)
    act_str = '  '.join(f"{i}={100*c/total:4.1f}%"
                        for i, c in enumerate(counts) if c > 0)
    lines.append(f"  actions: {act_str}")
    return '\n'.join(lines)


def _wilson_ci(wins: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _build_cfg(config_path: str, mode: str):
    """mode: 'dqn_vs_rb' or 'rb_vs_rb'."""
    cfg = GameConfig.from_yaml(config_path)
    if mode == 'rb_vs_rb':
        cfg.players = [
            {'name': 'DQN',       'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
            {'name': 'RuleBased', 'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
        ]
    else:
        cfg.players = [
            {'name': 'DQN',       'settings': 'RuleBasedPlayerSettings',
             'player_class': 'DQNPlayer',            'starting_money': 1500},
            {'name': 'RuleBased', 'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
        ]
    return cfg


def _snapshot_game(sa_env, player_names, seed_used: int, action_history,
                   dqn_seat: str) -> dict:
    """Extract a JSONL-friendly summary of the finished game state."""
    multi   = sa_env._multi
    players = multi._players
    board   = multi.board
    name_to_idx = multi._name_to_idx

    alive = [a for a in player_names
             if not players[name_to_idx[a]].is_bankrupt]
    winner = alive[0] if len(alive) == 1 else None

    per_player = {}
    for name in player_names:
        p = players[name_to_idx[name]]
        owned = [c.name for c in p.owned] if hasattr(p, 'owned') else []
        mortgaged = [c.name for c in getattr(p, 'owned', [])
                     if getattr(c, 'is_mortgaged', False)]
        per_player[name] = {
            'bankrupt':      bool(p.is_bankrupt),
            'money':         int(p.money),
            'net_worth':     int(p.net_worth()),
            'position':      int(p.position),
            'owned':         owned,
            'mortgaged':     mortgaged,
            'n_properties':  len(owned),
            'n_monopolies':  multi._count_monopolies(p),
        }

    act_counts = [0] * 6
    for a in action_history:
        act_counts[a] += 1

    return {
        'seed':      seed_used,
        'winner':    winner,
        'rounds':    int(multi._round),
        'truncated': bool(winner is None),
        'dqn_seat':  dqn_seat,
        'players':   per_player,
        'action_counts':   act_counts,
        'action_history':  list(action_history),
    }


def _format_game_summary(snap: dict) -> str:
    """Human-readable per-game summary for --verbose-game."""
    lines = [
        f"--- seed={snap['seed']}  rounds={snap['rounds']}  "
        f"winner={snap['winner'] or 'NONE (truncated)'}  ---"
    ]
    for name, pdata in snap['players'].items():
        status = 'BANKRUPT' if pdata['bankrupt'] else f"${pdata['money']}"
        lines.append(
            f"  {name:12s}  {status:>10s}  nw=${pdata['net_worth']:>5d}  "
            f"props={pdata['n_properties']:2d}  mono={pdata['n_monopolies']}  "
            f"mortg={len(pdata['mortgaged'])}"
        )
    counts = snap['action_counts']
    total = sum(counts) or 1
    nz = [(i, c) for i, c in enumerate(counts) if c > 0]
    act_str = '  '.join(f"{i}:{c}({100*c/total:.0f}%)" for i, c in nz)
    lines.append(f"  DQN actions: {act_str}")
    return '\n'.join(lines)


def run_tournament(cfg, model, n_games: int, seed: int,
                   track_actions: bool = False,
                   trace_path: str = None,
                   verbose_first_n: int = 0,
                   collect_snapshots: bool = False,
                   show_progress: bool = True,
                   progress_leave: bool = True,
                   progress_desc: str = 'Evaluating'):
    """Run a 2-player tournament and return per-player wins + action counts.

    When model is None, both seats play their configured strategy (used for
    the RuleBased-vs-RuleBased noise-floor baseline). Otherwise the DQN seat
    queries the model with LSTM state carried through each game.

    If collect_snapshots=True (or a trace_path / verbose_first_n is set), every
    game's per-game snapshot is captured and the full list is returned as the
    fifth tuple element. Otherwise that element is None.
    """
    sa_env = HistorySingleAgentMonopolyEnv(
        cfg, agent_name='DQN', seed=seed, history_len=1)
    player_names = sa_env._multi.possible_agents

    wins = {name: 0 for name in player_names}
    clear_winners = 0
    action_counts = [0] * 6
    snapshots = []
    keep_snapshots = collect_snapshots or (trace_path is not None) or (verbose_first_n > 0)

    trace_fh = open(trace_path, 'w') if trace_path else None
    iterator = (tqdm(range(n_games), desc=progress_desc,
                     leave=progress_leave, dynamic_ncols=True)
                if show_progress else range(n_games))

    for i in iterator:
        seed_used = seed + i
        obs, _    = sa_env.reset(seed=seed_used)
        state     = None
        ep_start  = np.array([True])
        done      = False
        action_history = []

        while not done:
            if model is not None:
                action, state = model.predict(
                    obs[np.newaxis], state=state,
                    episode_start=ep_start, deterministic=True)
                a = int(action[0])
            else:
                # No model: DQN seat plays fixed action 5 (RuleBased-equivalent)
                # since DQN uses RuleBasedPlayerSettings in rb_vs_rb mode.
                a = 5
            action_history.append(a)
            if track_actions:
                action_counts[a] += 1
            ep_start = np.array([False])
            obs, _, terminated, truncated, _ = sa_env.step(a)
            done = terminated or truncated

        alive = [a for a in player_names
                 if not sa_env._multi._players[
                     sa_env._multi._name_to_idx[a]].is_bankrupt]
        if len(alive) == 1:
            wins[alive[0]] += 1
            clear_winners += 1

        if keep_snapshots:
            snap = _snapshot_game(sa_env, player_names, seed_used,
                                  action_history, dqn_seat='DQN')
            snapshots.append(snap)
            if trace_fh is not None:
                trace_fh.write(json.dumps(snap) + '\n')
            if i < verbose_first_n:
                tqdm.write(_format_game_summary(snap))

    if trace_fh is not None:
        trace_fh.close()

    return wins, clear_winners, action_counts, player_names, (snapshots if keep_snapshots else None)


def _print_results(wins, clear_winners, n_games, player_names, action_counts=None):
    print(f'\nTournament: {n_games} games, {clear_winners} with a sole winner '
          f'({100 * clear_winners / n_games:.1f}%)')
    print('Win rates (sole winner only, 95% Wilson CI):')
    for name in player_names:
        lo, hi = _wilson_ci(wins[name], n_games)
        print(f'  {name:14s}: {100 * wins[name] / n_games:5.1f}%  '
              f'[{100 * lo:.1f}%, {100 * hi:.1f}%]')
    if action_counts is not None and sum(action_counts) > 0:
        total = sum(action_counts)
        print(f'\nDQN action distribution ({total:,} actions):')
        for i, (label, count) in enumerate(zip(_ACTION_LABELS, action_counts)):
            print(f'  {i} {label:15s}: {count:6d}  ({100 * count / total:.1f}%)')


def _write_summary(summary_path: str, mode: str, wins, clear_winners,
                   n_games, player_names, action_counts, debug_stats=None):
    """Write an aggregate tournament summary alongside the per-game trace."""
    ci = {name: _wilson_ci(wins[name], n_games) for name in player_names}
    summary = {
        'mode':           mode,
        'n_games':        n_games,
        'clear_winners':  clear_winners,
        'draw_rate':      1.0 - clear_winners / n_games,
        'wins':           {n: wins[n] for n in player_names},
        'win_rate':       {n: wins[n] / n_games for n in player_names},
        'win_rate_ci95':  {n: list(ci[n]) for n in player_names},
        'action_counts':  {_ACTION_LABELS[i]: action_counts[i]
                           for i in range(6)},
    }
    if debug_stats:
        summary['debug_stats'] = debug_stats
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='models/il_checkpoint',
                        help='Path to a RecurrentPPO checkpoint. Ignored in --self-play-baseline.')
    parser.add_argument('--n',      type=int, default=1000,
                        help='Number of games in the tournament.')
    parser.add_argument('--seed',   type=int, default=77777)
    parser.add_argument('--config', default='default_config.yaml')
    parser.add_argument('--self-play-baseline', action='store_true',
                        help='Run RuleBased vs RuleBased to measure the seat-order noise floor.')
    parser.add_argument('--action-dist', action='store_true',
                        help='Also print the DQN action-frequency histogram.')
    parser.add_argument('--save-trace',  default=None,
                        help='Path to a JSONL file; one per-game snapshot per line '
                             '(final cash, owned properties, actions, winner, etc.).')
    parser.add_argument('--save-summary', default=None,
                        help='Path to a JSON file with aggregate tournament results. '
                             'Auto-derived from --save-trace if omitted (same stem, .summary.json).')
    parser.add_argument('--verbose-game', type=int, default=0,
                        help='Print a detailed human-readable summary of the first N games.')
    args = parser.parse_args()

    # Auto-derive summary path from trace path if not provided.
    summary_path = args.save_summary
    if summary_path is None and args.save_trace:
        root, _ = os.path.splitext(args.save_trace)
        summary_path = root + '.summary.json'

    if args.self_play_baseline:
        print('Running RuleBased-vs-RuleBased baseline (no model)...')
        cfg = _build_cfg(args.config, mode='rb_vs_rb')
        wins, clear, counts, names, snaps = run_tournament(
            cfg, model=None, n_games=args.n, seed=args.seed,
            track_actions=args.action_dist,
            trace_path=args.save_trace,
            verbose_first_n=args.verbose_game,
            collect_snapshots=True)
        _print_results(wins, clear, args.n, names,
                       action_counts=counts if args.action_dist else None)
        stats = aggregate_debug_stats(snaps) if snaps else {}
        if stats:
            print('\nDebug stats:')
            print(format_debug_stats(stats))
        if summary_path:
            _write_summary(summary_path, 'rb_vs_rb', wins, clear,
                           args.n, names, counts, debug_stats=stats)
            print(f'Summary written to {summary_path}')
        if args.save_trace:
            print(f'Per-game trace written to {args.save_trace}')
        return

    print(f'Loading model from {args.model}...')
    model = RecurrentPPO.load(args.model)
    cfg = _build_cfg(args.config, mode='dqn_vs_rb')
    wins, clear, counts, names, snaps = run_tournament(
        cfg, model=model, n_games=args.n, seed=args.seed,
        track_actions=args.action_dist,
        trace_path=args.save_trace,
        verbose_first_n=args.verbose_game,
        collect_snapshots=True)
    _print_results(wins, clear, args.n, names,
                   action_counts=counts if args.action_dist else None)
    stats = aggregate_debug_stats(snaps) if snaps else {}
    if stats:
        print('\nDebug stats:')
        print(format_debug_stats(stats))
    if summary_path:
        _write_summary(summary_path, 'dqn_vs_rb', wins, clear,
                       args.n, names, counts, debug_stats=stats)
        print(f'Summary written to {summary_path}')
    if args.save_trace:
        print(f'Per-game trace written to {args.save_trace}')


if __name__ == '__main__':
    main()
