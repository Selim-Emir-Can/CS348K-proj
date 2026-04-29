"""Evaluate a trained DQN agent in a tournament vs RuleBased and Random.

Usage (from monopoly/):
    python scripts/eval_dqn.py                          # loads models/best_model.zip
    python scripts/eval_dqn.py --model models/dqn_monopoly
"""
import argparse

import numpy as np
from stable_baselines3 import DQN
from tqdm import tqdm

from config import GameConfig
from monopoly_env import MonopolyEnv


def run_tournament(cfg, model, n_games=1000, seed=0):
    env = MonopolyEnv(cfg, seed=seed)
    player_names = env.possible_agents
    wins = {a: 0 for a in player_names}
    sole_winner_games = 0

    for i in tqdm(range(n_games), desc='Evaluating'):
        obs_dict = {}
        env.reset(seed=seed + i)

        # Grab initial obs for the DQN agent
        dqn_agent = next(a for a in player_names if 'DQN' in a)
        obs_dict[dqn_agent] = env.observe(dqn_agent)

        while env.agents:
            a = env.agent_selection
            if env.terminations[a] or env.truncations[a]:
                env.step(None)
            elif a == dqn_agent:
                action, _ = model.predict(obs_dict[dqn_agent], deterministic=True)
                action = int(action)
                idx = env._name_to_idx[a]
                player = env._players[idx]
                if hasattr(player, '_buy_action'):
                    player._buy_action = action % 2
                if hasattr(player, '_build_threshold'):
                    from agents import _BUILD_THRESHOLDS
                    player._build_threshold = _BUILD_THRESHOLDS[action // 2]
                env.step(action)
                obs_dict[dqn_agent] = env.observe(dqn_agent)
            else:
                env.step(0)

        alive = [a for a in player_names
                 if not env._players[env._name_to_idx[a]].is_bankrupt]
        if len(alive) == 1:
            wins[alive[0]] += 1
            sole_winner_games += 1

    print(f'\nTournament: {n_games} games, {sole_winner_games} sole-winner games '
          f'({100*sole_winner_games/n_games:.1f}%)')
    print('Win rates (sole winner only):')
    for name in player_names:
        print(f'  {name:14s}: {100*wins[name]/n_games:5.1f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='models/best_model')
    parser.add_argument('--n',      type=int, default=1000)
    parser.add_argument('--config', default='default_config.yaml')
    args = parser.parse_args()

    cfg = GameConfig.from_yaml(args.config)
    cfg.players = [
        {'name': 'DQN',      'settings': 'StandardPlayerSettings',
         'player_class': 'DQNPlayer',             'starting_money': 1500},
        {'name': 'RuleBased','settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
        {'name': 'Random',   'settings': 'RandomPlayerSettings',
         'player_class': 'RandomPlayer',          'starting_money': 1500},
        {'name': 'Standard', 'settings': 'StandardPlayerSettings',  'starting_money': 1500},
    ]

    print(f'Loading model from {args.model}...')
    model = DQN.load(args.model)
    run_tournament(cfg, model, n_games=args.n)


if __name__ == '__main__':
    main()
