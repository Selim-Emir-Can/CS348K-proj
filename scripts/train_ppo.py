"""Train a RecurrentPPO (LSTM) agent in a 2-player game vs 1 RuleBased opponent.

The LSTM hidden state carries within-episode temporal context so the agent
can track board evolution without manual history concatenation.
Cross-game memory in HistorySingleAgentMonopolyEnv persists across episodes
and provides inter-game strategic statistics.

If --il-checkpoint is passed, we continue training from an imitation-learning
checkpoint (produced by scripts/train_il.py) instead of a fresh init. This is
the intended workflow: IL first to match RuleBased, then PPO fine-tune to
exceed it.

Usage (from monopoly/):
    python scripts/train_ppo.py                                  # 2M, fresh init
    python scripts/train_ppo.py --il-checkpoint models/il_checkpoint
    python scripts/train_ppo.py --steps 5000000

TensorBoard:
    tensorboard --logdir logs/
"""
import argparse
import os
import sys

import numpy as np
import wandb
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from wandb.integration.sb3 import WandbCallback

from agents import _BUILD_THRESHOLDS
from config import GameConfig
from monopoly_env import HistorySingleAgentMonopolyEnv

# Reuse the eval harness so training-time evaluations match CLI eval_ppo exactly.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from eval_ppo import (
    aggregate_debug_stats,
    format_debug_stats,
    run_tournament,
)

WANDB_PROJECT = 'monopoly-dqn'

# LSTM-friendly observation: base + strategic + cross-game (no manual history concat).
# The LSTM hidden state handles within-episode temporal context.
_HISTORY_LEN = 1


def make_env(cfg, seed=0):
    env = HistorySingleAgentMonopolyEnv(cfg, agent_name='DQN',
                                        seed=seed, history_len=_HISTORY_LEN)
    return Monitor(env)


# --------------------------------------------------------------------------- #
# Win-rate callback (LSTM-aware)                                               #
# --------------------------------------------------------------------------- #

class TqdmCallback(BaseCallback):
    """Single tqdm progress bar for the full training run."""

    def __init__(self, total_steps: int):
        super().__init__()
        self._total = total_steps
        self._bar   = None

    def _on_training_start(self):
        self._bar = tqdm(total=self._total, unit='step', dynamic_ncols=True)

    def _on_step(self) -> bool:
        self._bar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        self._bar.close()


class WinRateCallback(BaseCallback):
    """Mini tournament every eval_freq steps; logs detailed stats to W&B.

    Reuses eval_ppo.run_tournament so training-time eval matches the CLI eval.
    Optionally dumps per-game JSONL traces every `trace_every` evals for
    offline debugging (e.g. to see how the action distribution or bankruptcy
    rate evolves across checkpoints).
    """

    def __init__(self, cfg, n_games=200, eval_freq=50_000, verbose=1,
                 trace_dir: str = None, trace_every: int = 4,
                 seed: int = 77777):
        super().__init__(verbose)
        self._cfg         = cfg
        self._n_games     = n_games
        self._eval_freq   = eval_freq
        self._last_eval   = 0
        self._trace_dir   = trace_dir
        self._trace_every = max(trace_every, 1)
        self._eval_idx    = 0
        self._seed        = seed
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self._eval_freq:
            self._last_eval = self.num_timesteps
            self._eval_idx += 1
            self._run_eval()
        return True

    def _run_eval(self):
        trace_path = None
        if self._trace_dir and (self._eval_idx % self._trace_every == 0):
            trace_path = os.path.join(
                self._trace_dir,
                f'eval_step_{self.num_timesteps}.jsonl',
            )

        # Put policy into eval mode for deterministic predict(); restore after.
        policy = self.model.policy
        was_training = policy.training
        policy.set_training_mode(False)
        try:
            wins, clear, counts, names, snaps = run_tournament(
                self._cfg, model=self.model,
                n_games=self._n_games, seed=self._seed,
                track_actions=True,
                trace_path=trace_path,
                verbose_first_n=0,
                collect_snapshots=True,
                show_progress=True,
                progress_leave=False,
                progress_desc=f'eval@{self.num_timesteps:,}',
            )
        finally:
            policy.set_training_mode(was_training)

        stats = aggregate_debug_stats(snaps) if snaps else {}
        log = {f'eval/win_rate/{name}': wins[name] / self._n_games
               for name in names}
        log['eval/draw_rate']           = 1.0 - clear / self._n_games
        log['eval/action_entropy_bits'] = stats.get('action_entropy_bits', 0.0)
        log['eval/avg_rounds']          = stats.get('avg_rounds',        0.0)
        log['eval/dqn_bankruptcy_rate'] = stats.get('dqn_bankruptcy_rate', 0.0)
        log['eval/rb_bankruptcy_rate']  = stats.get('rb_bankruptcy_rate',  0.0)
        log['eval/avg_dqn_net_worth']   = stats.get('avg_dqn_net_worth',   0.0)
        log['eval/avg_rb_net_worth']    = stats.get('avg_rb_net_worth',    0.0)
        log['eval/avg_dqn_monopolies']  = stats.get('avg_dqn_monopolies',  0.0)
        log['eval/avg_dqn_properties']  = stats.get('avg_dqn_properties',  0.0)
        # Per-action frequencies (so the action-space shift is visible in W&B).
        total = max(stats.get('total_actions', 0), 1)
        for i, c in enumerate(stats.get('action_counts', [0] * 6)):
            log[f'eval/action_freq/{i}'] = c / total
        log['global_step'] = self.num_timesteps
        wandb.log(log)

        rates_str = '  '.join(
            f'{n}: {100 * wins[n] / self._n_games:.1f}%' for n in names)
        tqdm.write(f'  [WinRate @ {self.num_timesteps:,}] {rates_str}')
        tqdm.write(format_debug_stats(stats))
        if trace_path:
            tqdm.write(f'    trace → {trace_path}')


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',    type=int, default=2_000_000)
    parser.add_argument('--config',   default='default_config.yaml')
    parser.add_argument('--out',      default='models/ppo_monopoly')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--il-checkpoint', default=None,
                        help='Path to an IL checkpoint (e.g. models/il_checkpoint) to '
                             'warm-start from. If omitted, trains from scratch.')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='LR override. Default: 1e-4 fresh, 3e-5 when resuming from IL.')
    parser.add_argument('--ent-coef', type=float, default=None,
                        help='Entropy coef override. Default: 0.01 fresh, 0.03 when resuming from IL.')
    parser.add_argument('--eval-freq',   type=int, default=50_000)
    parser.add_argument('--eval-games',  type=int, default=200)
    parser.add_argument('--eval-trace-dir', default=None,
                        help='Directory for periodic JSONL traces during training. '
                             'Default: logs/<run_id>/eval_traces (set --no-eval-traces to disable).')
    parser.add_argument('--trace-every', type=int, default=4,
                        help='Dump a JSONL trace every N evals (default: every 4 evals).')
    parser.add_argument('--no-eval-traces', action='store_true',
                        help='Disable periodic trace dumping during training.')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs',   exist_ok=True)

    cfg = GameConfig.from_yaml(args.config)
    # 2-player setup: DQN agent vs 1 RuleBased. DQN uses RuleBasedPlayerSettings so
    # action 5 (buy + aggressive) is behaviourally identical to RuleBased.
    cfg.players = [
        {'name': 'DQN',       'settings': 'RuleBasedPlayerSettings',
         'player_class': 'DQNPlayer',            'starting_money': 1500},
        {'name': 'RuleBased', 'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
    ]

    # Resuming from IL → lower LR + higher entropy to explore off the saturated action 5.
    default_lr  = 3e-5 if args.il_checkpoint else 1e-4
    default_ent = 0.03 if args.il_checkpoint else 0.01
    learning_rate = args.learning_rate if args.learning_rate is not None else default_lr
    ent_coef      = args.ent_coef      if args.ent_coef      is not None else default_ent

    run = wandb.init(
        project=WANDB_PROJECT,
        name=args.run_name,
        config={
            'algo': 'RecurrentPPO',
            'steps': args.steps,
            'n_steps': 4096,
            'batch_size': 128,
            'n_epochs': 10,
            'learning_rate': learning_rate,
            'ent_coef': ent_coef,
            'clip_range': 0.2,
            'vf_coef': 0.5,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'lstm_hidden_size': 256,
            'net_arch': [256, 256],
            'action_space': 'Discrete(6)',
            'opponents': 'RuleBased (2-player)',
            'il_checkpoint': args.il_checkpoint,
            'cross_game_memory': HistorySingleAgentMonopolyEnv._CROSS_MEMORY_GAMES,
        },
        sync_tensorboard=True,
    )

    train_env = make_env(cfg, seed=0)
    eval_env  = make_env(cfg, seed=999)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='models/',
        log_path='models/',
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_games,
        deterministic=True,
        verbose=0,
    )

    if args.no_eval_traces:
        trace_dir = None
    else:
        trace_dir = args.eval_trace_dir or f'logs/{run.id}/eval_traces'

    winrate_callback = WinRateCallback(
        cfg, n_games=args.eval_games, eval_freq=args.eval_freq, verbose=1,
        trace_dir=trace_dir, trace_every=args.trace_every,
    )
    tqdm_callback    = TqdmCallback(total_steps=args.steps)
    wandb_callback   = WandbCallback(gradient_save_freq=10_000, verbose=0)

    if args.il_checkpoint:
        print(f'Loading IL checkpoint from {args.il_checkpoint}...')
        model = RecurrentPPO.load(
            args.il_checkpoint,
            env=train_env,
            tensorboard_log=f'logs/{run.id}',
            device='cuda',
            # Override hyperparams saved in the checkpoint so fine-tuning uses
            # the fresh LR / entropy we want for this run.
            custom_objects={
                'learning_rate': learning_rate,
                'lr_schedule':   lambda _: learning_rate,
                'ent_coef':      ent_coef,
                'clip_range':    lambda _: 0.2,
            },
        )
        model.learning_rate = learning_rate
        model.ent_coef      = ent_coef
    else:
        model = RecurrentPPO(
            'MlpLstmPolicy',
            train_env,
            learning_rate=learning_rate,
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=ent_coef,     # entropy bonus — prevents policy collapse
            clip_range=0.2,
            vf_coef=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                lstm_hidden_size=256,
                n_lstm_layers=1,
                enable_critic_lstm=True,
            ),
            tensorboard_log=f'logs/{run.id}',
            verbose=0,
            device='cuda',
        )

    print(f'Training RecurrentPPO for {args.steps:,} steps...')
    print(f'Obs size: {train_env.observation_space.shape[0]}  |  Action space: {train_env.action_space}')
    print(f'LR: {learning_rate}  |  ent_coef: {ent_coef}  |  IL init: {bool(args.il_checkpoint)}')
    print(f'W&B run: {run.url}')

    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([eval_callback, winrate_callback,
                               tqdm_callback, wandb_callback]),
    )
    model.save(args.out)
    print(f'Model saved to {args.out}.zip')
    run.finish()


if __name__ == '__main__':
    main()
