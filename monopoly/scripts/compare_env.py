"""Compare MonopolyEnv statistics against the original monopoly_game_from_config.

Both paths run on identical seeds so any divergence is a logic bug, not noise.
A small N (~200) is enough to spot systematic differences in win rates.

Usage (from monopoly/):
    python scripts/compare_env.py           # 200 games
    python scripts/compare_env.py --n 1000
"""
import argparse
import random

from monopoly.core.game import setup_game_from_config, setup_players_from_config
from monopoly.core.game_utils import _check_end_conditions
from monopoly.core.move_result import MoveResult
from monopoly.log import Log
from monopoly_env import MonopolyEnv
from config import GameConfig
from settings import SimulationSettings


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _make_seeds(n: int, master_seed: int = 0):
    rng = random.Random(master_seed)
    return [rng.getrandbits(32) for _ in range(n)]


def _summarise(bankruptcies: list, n_games: int, player_names: list, n_moves: int):
    """Return a summary dict from a list of (game_idx, player_name, round)."""
    from collections import defaultdict

    losses_by_player = defaultdict(int)
    last_bankruptcy_per_game = {}
    bankruptcies_per_game = defaultdict(int)

    for game_idx, name, turn in bankruptcies:
        losses_by_player[name] += 1
        last_bankruptcy_per_game[game_idx] = max(
            last_bankruptcy_per_game.get(game_idx, 0), turn
        )
        bankruptcies_per_game[game_idx] += 1

    # Clear winner = exactly (n_players - 1) bankruptcies in a game
    n_players = len(player_names)
    clear_winner_games = sum(
        1 for g, cnt in bankruptcies_per_game.items()
        if cnt == n_players - 1
    )

    # Median game length (finished games only)
    finished_lengths = sorted(
        turn for g, turn in last_bankruptcy_per_game.items()
        if bankruptcies_per_game[g] == n_players - 1
    )
    all_lengths = finished_lengths + [n_moves] * (n_games - len(finished_lengths))
    all_lengths.sort()

    survival_rates = {
        name: (n_games - losses_by_player[name]) / n_games
        for name in player_names
    }

    return {
        "clear_winner_pct": 100 * clear_winner_games / n_games,
        "median_length_finished": (
            finished_lengths[len(finished_lengths) // 2] if finished_lengths else None
        ),
        "median_length_all": all_lengths[len(all_lengths) // 2],
        "survival_rates": survival_rates,
    }


def _print_summary(label: str, s: dict, player_names: list):
    print(f"\n{'-'*50}")
    print(f"  {label}")
    print(f"{'-'*50}")
    print(f"  Clear winner games : {s['clear_winner_pct']:.1f}%")
    if s["median_length_finished"] is not None:
        print(f"  Median length (finished) : {s['median_length_finished']}")
    print(f"  Median length (all)      : {s['median_length_all']}")
    print("  Survival rates:")
    for name in player_names:
        print(f"    {name:10s}: {s['survival_rates'][name]*100:.1f}%")


# --------------------------------------------------------------------------- #
# Two execution paths                                                           #
# --------------------------------------------------------------------------- #

def run_original(seeds: list, cfg: GameConfig) -> list:
    """Run via the original monopoly_game_from_config logic (single-process)."""
    null_log = Log(disabled=True)
    bankruptcies = []

    for game_idx, seed in enumerate(seeds):
        board, dice, elog, blog = setup_game_from_config(game_idx + 1, seed, cfg)
        elog.disabled = True
        blog.disabled = True
        players = setup_players_from_config(board, dice, cfg)

        for turn_n in range(1, SimulationSettings.n_moves + 1):
            if _check_end_conditions(players, null_log, game_idx + 1, turn_n):
                break
            for player in players:
                if player.is_bankrupt:
                    continue
                result = player.make_a_move(board, players, dice, elog)
                if result == MoveResult.BANKRUPT:
                    bankruptcies.append((game_idx + 1, player.name, turn_n))

    return bankruptcies


def run_env(seeds: list, cfg: GameConfig) -> list:
    """Run via MonopolyEnv."""
    bankruptcies = []
    env = MonopolyEnv(cfg)

    for game_idx, seed in enumerate(seeds):
        env.reset(seed=seed)
        seen_terminated = set()

        while env.agents:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
            else:
                env.step(0)

            # Record newly bankrupt players
            for a in env.possible_agents:
                if a not in seen_terminated and env.terminations.get(a):
                    p = env._players[env._name_to_idx[a]]
                    if p.is_bankrupt:
                        bankruptcies.append((game_idx + 1, a, env._round))
                    seen_terminated.add(a)

    return bankruptcies


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of games")
    parser.add_argument("--config", default="default_config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = GameConfig.from_yaml(args.config)
    if cfg.players:
        player_names = [p["name"] for p in cfg.players]
    else:
        player_names = [name for name, _ in cfg.settings.players_list]

    seeds = _make_seeds(args.n, args.seed)
    print(f"Running {args.n} games on both paths (master seed={args.seed})…")

    orig_bankruptcies = run_original(seeds, cfg)
    env_bankruptcies  = run_env(seeds, cfg)

    orig = _summarise(orig_bankruptcies, args.n, player_names, SimulationSettings.n_moves)
    env  = _summarise(env_bankruptcies,  args.n, player_names, SimulationSettings.n_moves)

    _print_summary("Original (monopoly_game_from_config)", orig, player_names)
    _print_summary("Env      (MonopolyEnv)",               env,  player_names)

    # Quick diff
    # Confidence-interval-aware threshold: 2 * SE at 50% rate
    import math
    ci_half = 2 * math.sqrt(0.25 / args.n) * 100   # ~95% half-width in pp
    print(f"\n{'-'*50}")
    print(f"  Survival rate delta (env - original)  [95% CI half-width: {ci_half:.1f}pp]:")
    max_diff = 0.0
    for name in player_names:
        diff = (env["survival_rates"][name] - orig["survival_rates"][name]) * 100
        max_diff = max(max_diff, abs(diff))
        print(f"    {name:10s}: {diff:+.1f}pp")
    print(f"\n  Max |delta| = {max_diff:.1f}pp  -->  ", end="")
    print("CONSISTENT" if max_diff < ci_half else "DIVERGED -- investigate")


if __name__ == "__main__":
    main()
