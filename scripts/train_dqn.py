"""Train a DQN agent against 3 RuleBased opponents.

Usage (from monopoly/):
    python scripts/train_dqn.py                          # 2M steps, BC warm-start
    python scripts/train_dqn.py --steps 5000000          # longer run
    python scripts/train_dqn.py --run-name my_run        # custom W&B run name
    python scripts/train_dqn.py --no-pretrain            # skip BC warm-start

TensorBoard:
    tensorboard --logdir logs/
"""
import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from agents import _BUILD_THRESHOLDS
from config import GameConfig
from monopoly_env import HistorySingleAgentMonopolyEnv, MonopolyEnv, SingleAgentMonopolyEnv

WANDB_PROJECT = 'monopoly-dqn'


def make_env(cfg, seed=0, history_len=1):
    if history_len > 1:
        env = HistorySingleAgentMonopolyEnv(cfg, agent_name='DQN',
                                             seed=seed, history_len=history_len)
    else:
        env = SingleAgentMonopolyEnv(cfg, agent_name='DQN', seed=seed)
    return Monitor(env)


def behavioral_cloning_pretrain(model, env, n_steps=50_000, n_epochs=5, lr=1e-3):
    """Warm-start the Q-network to mimic RuleBased strategy.

    RuleBased always buys and builds aggressively → action 5 in Discrete(6).
    We collect observations under that policy, supervise the Q-network with
    cross-entropy to prefer action 5, then pre-fill the replay buffer with
    those same transitions so RL starts from useful experience.
    """
    BC_ACTION = 5  # buy + aggressive build

    # --- collect demonstrations ---
    print(f'[BC] Collecting {n_steps:,} RuleBased demonstrations...')
    obs_list, act_list = [], []
    next_obs_list, rew_list, done_list = [], [], []
    obs, _ = env.reset()
    for _ in range(n_steps):
        obs_list.append(obs.copy())
        act_list.append(BC_ACTION)
        next_obs, reward, terminated, truncated, _ = env.step(BC_ACTION)
        done = terminated or truncated
        next_obs_list.append(next_obs.copy())
        rew_list.append(reward)
        done_list.append(done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

    # --- behavioural cloning: supervise Q-net to prefer BC_ACTION ---
    obs_t  = th.FloatTensor(np.array(obs_list)).to(model.device)
    act_t  = th.LongTensor(act_list).to(model.device)
    dataset = th.utils.data.TensorDataset(obs_t, act_t)
    loader  = th.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    optimizer = th.optim.Adam(model.policy.parameters(), lr=lr)

    print(f'[BC] Supervised pre-training ({n_epochs} epochs)...')
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            q_vals = model.policy.q_net(batch_obs)
            loss = F.cross_entropy(q_vals, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f'  epoch {epoch+1}/{n_epochs}  loss={avg:.4f}')
        wandb.log({'pretrain/bc_loss': avg, 'pretrain/epoch': epoch + 1})

    # --- pre-fill replay buffer with the same transitions ---
    print('[BC] Pre-filling replay buffer...')
    for i in range(n_steps):
        model.replay_buffer.add(
            obs_list[i][np.newaxis],
            next_obs_list[i][np.newaxis],
            np.array([[BC_ACTION]]),
            np.array([rew_list[i]]),
            np.array([done_list[i]]),
            [{}],
        )
    print('[BC] Done.')


class WinRateCallback(BaseCallback):
    """Runs a mini tournament every eval_freq steps and logs DQN sole-winner rate to W&B."""

    def __init__(self, cfg, n_games=200, eval_freq=50_000, verbose=1):
        super().__init__(verbose)
        self._cfg = cfg
        self._n_games = n_games
        self._eval_freq = eval_freq
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self._eval_freq:
            self._last_eval = self.num_timesteps
            win_rate = self._tournament()
            wandb.log({'eval/dqn_win_rate': win_rate,
                       'global_step': self.num_timesteps})
            if self.verbose:
                print(f'  [WinRate @ {self.num_timesteps:,}] DQN: {100*win_rate:.1f}%')
        return True

    def _tournament(self) -> float:
        env = MonopolyEnv(self._cfg, seed=77777)
        player_names = env.possible_agents
        dqn_agent = next(a for a in player_names if 'DQN' in a)
        wins = 0
        for i in range(self._n_games):
            env.reset(seed=77777 + i)
            obs = env.observe(dqn_agent)
            while env.agents:
                a = env.agent_selection
                if env.terminations[a] or env.truncations[a]:
                    env.step(None)
                elif a == dqn_agent:
                    action, _ = self.model.predict(obs, deterministic=True)
                    action = int(action)
                    idx = env._name_to_idx[a]
                    player = env._players[idx]
                    if hasattr(player, '_buy_action'):
                        player._buy_action = action % 2
                    if hasattr(player, '_build_threshold'):
                        player._build_threshold = _BUILD_THRESHOLDS[action // 2]
                    env.step(action)
                    obs = env.observe(dqn_agent)
                else:
                    env.step(0)
            alive = [a for a in player_names
                     if not env._players[env._name_to_idx[a]].is_bankrupt]
            if len(alive) == 1 and alive[0] == dqn_agent:
                wins += 1
        return wins / self._n_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',    type=int, default=2_000_000)
    parser.add_argument('--config',   default='default_config.yaml')
    parser.add_argument('--out',      default='models/dqn_monopoly')
    parser.add_argument('--run-name',   default=None,
                        help='W&B run name (default: auto-generated)')
    parser.add_argument('--no-pretrain',  action='store_true',
                        help='Skip behavioral cloning warm-start')
    parser.add_argument('--bc-steps',    type=int, default=50_000,
                        help='Demonstration steps for BC pre-training')
    parser.add_argument('--history-len', type=int, default=1,
                        help='History window length (1=no history, 5=default history)')
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

    run = wandb.init(
        project=WANDB_PROJECT,
        name=args.run_name,
        config={
            'steps': args.steps,
            'batch_size': 512,
            'buffer_size': 500_000,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'gradient_steps': 4,
            'net_arch': [256, 256],
            'action_space': 'Discrete(6)',
            'bc_pretrain': not args.no_pretrain,
            'bc_steps': args.bc_steps,
            'history_len': args.history_len,
        },
        sync_tensorboard=True,
    )

    train_env = make_env(cfg, seed=0,   history_len=args.history_len)
    eval_env  = make_env(cfg, seed=999, history_len=args.history_len)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='models/',
        log_path='models/',
        eval_freq=50_000,
        n_eval_episodes=200,
        deterministic=True,
        verbose=1,
    )

    winrate_callback = WinRateCallback(cfg, n_games=200, eval_freq=50_000, verbose=1)

    wandb_callback = WandbCallback(
        gradient_save_freq=10_000,
        verbose=1,
    )

    model = DQN(
        'MlpPolicy',
        train_env,
        learning_rate=1e-4,
        buffer_size=500_000,
        learning_starts=10_000,
        batch_size=512,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=1_000,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=f'logs/{run.id}',
        verbose=1,
        device='cuda',
    )

    if not args.no_pretrain:
        behavioral_cloning_pretrain(model, make_env(cfg, seed=42,
                                                    history_len=args.history_len),
                                    n_steps=args.bc_steps)

    print(f'Training DQN for {args.steps:,} steps...')
    print(f'W&B run: {run.url}')
    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([eval_callback, winrate_callback, wandb_callback]),
    )
    model.save(args.out)
    print(f'Model saved to {args.out}.zip')
    run.finish()


if __name__ == '__main__':
    main()
