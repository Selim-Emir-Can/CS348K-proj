"""Train a DQN agent against 3 RuleBased opponents.

Usage (from monopoly/):
    python scripts/train_dqn.py                     # 500k steps
    python scripts/train_dqn.py --steps 2000000     # longer run
"""
import argparse
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from config import GameConfig
from monopoly_env import SingleAgentMonopolyEnv


def make_env(cfg, seed=0):
    env = SingleAgentMonopolyEnv(cfg, agent_name='DQN', seed=seed)
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',  type=int, default=500_000)
    parser.add_argument('--config', default='default_config.yaml')
    parser.add_argument('--out',    default='models/dqn_monopoly')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)

    cfg = GameConfig.from_yaml(args.config)
    cfg.players = [
        {'name': 'DQN',       'settings': 'StandardPlayerSettings',
         'player_class': 'DQNPlayer',             'starting_money': 1500},
        {'name': 'RuleBased1','settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
        {'name': 'RuleBased2','settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
        {'name': 'RuleBased3','settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
    ]

    train_env = make_env(cfg, seed=0)
    eval_env  = make_env(cfg, seed=999)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='models/',
        log_path='models/',
        eval_freq=50_000,
        n_eval_episodes=200,
        deterministic=True,
        verbose=1,
    )

    model = DQN(
        'MlpPolicy',
        train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        verbose=1,
    )

    print(f'Training DQN for {args.steps:,} steps...')
    model.learn(total_timesteps=args.steps, callback=eval_callback)
    model.save(args.out)
    print(f'Model saved to {args.out}.zip')


if __name__ == '__main__':
    main()
