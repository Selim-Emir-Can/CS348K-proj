"""Benchmark: run N games through MonopolyEnv and report throughput.

Usage (from monopoly/):
    python scripts/benchmark_env.py            # 10 000 games
    python scripts/benchmark_env.py --n 1000   # quick check
"""
import argparse
import time

from config import GameConfig
from monopoly_env import MonopolyEnv


def run_games(n_games: int, cfg: GameConfig, seed: int = 0) -> dict:
    env = MonopolyEnv(cfg, seed=seed)
    player_names = env.possible_agents
    total_steps = 0
    total_rounds = 0
    sole_wins = {a: 0 for a in player_names}
    sole_winner_games = 0

    for game_i in range(n_games):
        env.reset(seed=seed + game_i)
        while env.agents:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
            else:
                env.step(0)
            total_steps += 1
        total_rounds += env._round

        alive = [a for a in player_names
                 if not env._players[env._name_to_idx[a]].is_bankrupt]
        if len(alive) == 1:
            sole_wins[alive[0]] += 1
            sole_winner_games += 1

    return {
        "games": n_games,
        "total_steps": total_steps,
        "total_rounds": total_rounds,
        "avg_steps_per_game": total_steps / n_games,
        "avg_rounds_per_game": total_rounds / n_games,
        "sole_winner_games": sole_winner_games,
        "sole_wins": sole_wins,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="number of games")
    parser.add_argument("--config", default="default_config.yaml", help="config YAML")
    args = parser.parse_args()

    cfg = GameConfig.from_yaml(args.config)

    print(f"Benchmarking {args.n:,} games...")
    t0 = time.perf_counter()
    stats = run_games(args.n, cfg)
    elapsed = time.perf_counter() - t0

    sole_pct = 100 * stats["sole_winner_games"] / stats["games"]
    print(f"\nResults")
    print(f"  Games            : {stats['games']:>10,}")
    print(f"  Total steps      : {stats['total_steps']:>10,}")
    print(f"  Elapsed          : {elapsed:>10.2f} s")
    print(f"  Games / second   : {stats['games'] / elapsed:>10.1f}")
    print(f"  Steps / second   : {stats['total_steps'] / elapsed:>10.1f}")
    print(f"  Avg steps / game : {stats['avg_steps_per_game']:>10.1f}")
    print(f"  Avg rounds / game: {stats['avg_rounds_per_game']:>10.1f}")
    print(f"  Sole-winner games: {stats['sole_winner_games']:>10,}  ({sole_pct:.1f}%)")
    print(f"\n  Win rate (sole winner only):")
    for name, w in stats["sole_wins"].items():
        print(f"    {name:12s}: {100 * w / stats['games']:5.1f}%")

    target = 60.0
    status = "PASS" if elapsed < target else "FAIL"
    print(f"\n  Target < {target:.0f}s -- {status} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
