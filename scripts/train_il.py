"""Imitation-learn a RecurrentPPO policy to mimic RuleBasedPlayer in a 2-player game.

Goal: produce an IL checkpoint that, when used as a policy, has a win rate
statistically indistinguishable from 50% vs RuleBased in a 2-player tournament
(i.e. it matches RuleBased behaviourally). Phase 2 (scripts/train_ppo.py) then
RL-fine-tunes from this checkpoint to exceed 50%.

How the behavioural match works:
  - cfg.players uses RuleBasedPlayerSettings for the DQN slot. This sets
    unspendable_cash=0 and is_willing_to_make_trades=False — exactly matching
    RuleBased — so any action that encodes "buy + $0 build floor" (action 5
    in Discrete(6)) is behaviourally identical to RuleBased.
  - We collect (obs, action=5) demonstrations and supervise the policy via
    cross-entropy on the action logits, running obs through the LSTM as full
    episode sequences (not shuffled transitions) so the recurrent state is
    exercised the same way at train and test time.
  - Optionally regress the value head onto Monte-Carlo returns so Phase 2
    starts with a warm critic.

Usage (from monopoly/):
    python scripts/train_il.py                          # 100k demo steps, 5 epochs
    python scripts/train_il.py --demo-steps 200000
    python scripts/train_il.py --out models/il_checkpoint_v2

Sanity-check after training:
    python scripts/eval_ppo.py --model models/il_checkpoint --n 1000
    # → DQN win rate ≈ 50% vs RuleBased
"""
import argparse
import os
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

from config import GameConfig
from monopoly_env import HistorySingleAgentMonopolyEnv

# Allow `from eval_ppo import ...` whether this script is run from monopoly/
# (PYTHONPATH=.) or as a module — scripts/ isn't a package.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from eval_ppo import (
    aggregate_debug_stats,
    format_debug_stats,
    run_tournament,
)

WANDB_PROJECT = 'monopoly-dqn'

# Lightweight shim so the rest of the module can call wandb.* unconditionally.
class _WandbStub:
    def init(self, **_): return self
    def log(self, *_args, **_kwargs): pass
    def finish(self): pass
    @property
    def id(self): return 'nowandb'
    @property
    def url(self): return '(wandb disabled)'

wandb = _WandbStub()  # replaced in main() if --no-wandb is not passed

BC_ACTION = 5          # buy + aggressive ($0 build floor) → matches RuleBased
_HISTORY_LEN = 1       # LSTM handles temporal state; no manual history concat


def make_env(cfg, seed=0):
    env = HistorySingleAgentMonopolyEnv(cfg, agent_name='DQN',
                                        seed=seed, history_len=_HISTORY_LEN)
    return Monitor(env)


def collect_episodes(env, n_steps: int):
    """Roll fixed-action episodes; return a list of per-episode obs arrays.

    Each returned episode is an ndarray of shape (T_i, obs_dim). We also
    return per-episode rewards so we can compute discounted returns for
    optional critic regression.
    """
    episodes_obs = []
    episodes_rew = []
    cur_obs, cur_rew = [], []

    obs, _ = env.reset()
    steps = 0
    pbar = tqdm(total=n_steps, desc='Collect', unit='step', dynamic_ncols=True)
    while steps < n_steps:
        cur_obs.append(obs.astype(np.float32, copy=True))
        obs, reward, terminated, truncated, _ = env.step(BC_ACTION)
        cur_rew.append(float(reward))
        steps += 1
        pbar.update(1)
        if terminated or truncated:
            episodes_obs.append(np.stack(cur_obs, axis=0))
            episodes_rew.append(np.array(cur_rew, dtype=np.float32))
            cur_obs, cur_rew = [], []
            obs, _ = env.reset()
            pbar.set_postfix(episodes=len(episodes_obs), avg_len=steps / max(len(episodes_obs), 1))
    pbar.close()

    # Keep any trailing partial episode so we don't waste samples.
    if cur_obs:
        episodes_obs.append(np.stack(cur_obs, axis=0))
        episodes_rew.append(np.array(cur_rew, dtype=np.float32))

    return episodes_obs, episodes_rew


def _zero_lstm_states(policy, n_seq: int, device) -> RNNStates:
    """Build zero-initialised RNNStates for n_seq parallel sequences."""
    lstm = policy.lstm_actor
    shape = (lstm.num_layers, n_seq, lstm.hidden_size)
    h0 = th.zeros(shape, device=device)
    c0 = th.zeros(shape, device=device)
    # When enable_critic_lstm=True, lstm_critic exists too; otherwise fall back.
    if policy.lstm_critic is not None:
        return RNNStates(pi=(h0, c0), vf=(h0.clone(), c0.clone()))
    # Shared or feedforward critic: still need a placeholder with matching shape.
    return RNNStates(pi=(h0, c0), vf=(h0.clone(), c0.clone()))


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Monte-Carlo discounted returns, computed backward."""
    g = 0.0
    out = np.zeros_like(rewards, dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def _eval_epoch(model, cfg, n_games: int, seed: int, trace_path: str = None):
    """Quick tournament eval at the end of an IL epoch.

    Returns (win_rate_dqn, stats_dict). stats_dict is from aggregate_debug_stats.
    """
    wins, clear, counts, names, snaps = run_tournament(
        cfg, model=model, n_games=n_games, seed=seed,
        track_actions=True, trace_path=trace_path,
        verbose_first_n=0, collect_snapshots=True,
        show_progress=True,
        progress_leave=False,
        progress_desc='epoch-eval',
    )
    win_rate = wins.get('DQN', 0) / max(n_games, 1)
    stats = aggregate_debug_stats(snaps) if snaps else {}
    return win_rate, wins, stats


def il_train(model: RecurrentPPO, episodes_obs, episodes_rew,
             n_epochs: int, lr: float, value_coef: float, gamma: float,
             eval_cfg=None, eval_games: int = 0, eval_seed: int = 77777,
             eval_trace_dir: str = None):
    """Supervise the policy to output BC_ACTION and (optionally) fit the value head.

    Processes each episode as a single recurrent sequence: n_seq=1, T=len(episode).

    If eval_cfg and eval_games > 0, runs a small DQN-vs-RB tournament at the
    end of each epoch so we can see when the policy becomes behaviourally
    equivalent to RuleBased (win rate and action entropy → stable). If
    eval_trace_dir is set, per-epoch traces are saved as eval_epoch_<N>.jsonl.
    """
    device = model.device
    policy = model.policy
    optimizer = th.optim.Adam(policy.parameters(), lr=lr)

    # Pre-compute per-episode returns (used only if value_coef > 0).
    episodes_ret = [_discounted_returns(r, gamma) for r in episodes_rew]

    for epoch in range(n_epochs):
        # Shuffle episode order each epoch; each episode is a contiguous sequence.
        order = np.random.permutation(len(episodes_obs))
        running_ce, running_vf, n_trans = 0.0, 0.0, 0

        ep_bar = tqdm(order, desc=f'epoch {epoch+1}/{n_epochs}',
                      unit='ep', leave=False, dynamic_ncols=True)
        for ei in ep_bar:
            obs_np = episodes_obs[ei]
            T = obs_np.shape[0]
            if T == 0:
                continue

            obs  = th.as_tensor(obs_np, device=device)                       # (T, obs_dim)
            acts = th.full((T,), BC_ACTION, dtype=th.long, device=device)    # (T,)
            ep_start = th.zeros(T, dtype=th.float32, device=device)          # (T,)
            ep_start[0] = 1.0
            lstm_states = _zero_lstm_states(policy, n_seq=1, device=device)

            values, log_prob, _ = policy.evaluate_actions(
                obs, acts, lstm_states, ep_start
            )

            # Cross-entropy surrogate: minimise -log_prob of the target action.
            ce_loss = -log_prob.mean()

            if value_coef > 0.0:
                ret = th.as_tensor(episodes_ret[ei], device=device)
                vf_loss = F.mse_loss(values.flatten(), ret)
            else:
                vf_loss = th.zeros((), device=device)

            loss = ce_loss + value_coef * vf_loss

            optimizer.zero_grad()
            loss.backward()
            # Gentle grad clip to keep logit magnitudes reasonable
            # (avoids saturating too hard, which would hurt Phase 2 exploration).
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            running_ce += ce_loss.item() * T
            running_vf += vf_loss.item() * T
            n_trans    += T
            ep_bar.set_postfix(ce=f'{running_ce / max(n_trans, 1):.3f}',
                               vf=f'{running_vf / max(n_trans, 1):.3f}')
        ep_bar.close()

        avg_ce = running_ce / max(n_trans, 1)
        avg_vf = running_vf / max(n_trans, 1)
        tqdm.write(f'  epoch {epoch+1}/{n_epochs}  ce={avg_ce:.4f}  vf={avg_vf:.4f}  '
                   f'(trans={n_trans:,})')
        wandb.log({'il/ce_loss': avg_ce, 'il/vf_loss': avg_vf, 'il/epoch': epoch + 1})

        # End-of-epoch tournament eval: tells us whether low CE loss actually
        # translated to RuleBased-equivalent behaviour in real games.
        if eval_cfg is not None and eval_games > 0:
            trace_path = None
            if eval_trace_dir:
                os.makedirs(eval_trace_dir, exist_ok=True)
                trace_path = os.path.join(
                    eval_trace_dir, f'il_epoch_{epoch+1}.jsonl')
            # Put policy in eval mode for deterministic predict(); restore after.
            was_training = policy.training
            policy.set_training_mode(False)
            try:
                win_rate, wins, stats = _eval_epoch(
                    model, eval_cfg,
                    n_games=eval_games, seed=eval_seed + epoch * 10_000,
                    trace_path=trace_path,
                )
            finally:
                policy.set_training_mode(was_training)
            print(f'  [epoch {epoch+1} eval @ {eval_games} games]  '
                  f"DQN={100*win_rate:.1f}%  RB={100*wins.get('RuleBased', 0)/eval_games:.1f}%")
            print(format_debug_stats(stats))
            wandb.log({
                'il/eval/dqn_win_rate':      win_rate,
                'il/eval/rb_win_rate':       wins.get('RuleBased', 0) / eval_games,
                'il/eval/draw_rate':         stats.get('truncation_rate', 0.0),
                'il/eval/action_entropy':    stats.get('action_entropy_bits', 0.0),
                'il/eval/avg_rounds':        stats.get('avg_rounds', 0.0),
                'il/eval/avg_dqn_monopolies':stats.get('avg_dqn_monopolies', 0.0),
                'il/eval/dqn_bankruptcy_rate':stats.get('dqn_bankruptcy_rate', 0.0),
                'il/epoch':                  epoch + 1,
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='default_config.yaml')
    parser.add_argument('--out',        default='models/il_checkpoint')
    parser.add_argument('--run-name',   default=None)
    parser.add_argument('--demo-steps', type=int, default=100_000)
    parser.add_argument('--epochs',     type=int, default=5)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--value-coef', type=float, default=0.5,
                        help='Weight on value-head MSE loss. Set to 0 to skip value supervision.')
    parser.add_argument('--gamma',      type=float, default=0.99)
    parser.add_argument('--no-wandb',   action='store_true',
                        help='Disable W&B logging (useful for smoke tests / offline debug).')
    parser.add_argument('--eval-games', type=int, default=100,
                        help='Games to play for the end-of-epoch sanity-check tournament. 0 disables.')
    parser.add_argument('--eval-seed',  type=int, default=77777)
    parser.add_argument('--eval-trace-dir', default='logs/il_eval_traces',
                        help='Directory for per-epoch JSONL traces. Set to empty string to skip.')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs',   exist_ok=True)

    global wandb
    if not args.no_wandb:
        import wandb as _wandb_real
        wandb = _wandb_real

    cfg = GameConfig.from_yaml(args.config)
    # 2-player: DQN (behavioural twin of RuleBased when action=5) vs RuleBased.
    cfg.players = [
        {'name': 'DQN',       'settings': 'RuleBasedPlayerSettings',
         'player_class': 'DQNPlayer',            'starting_money': 1500},
        {'name': 'RuleBased', 'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
    ]

    run = wandb.init(
        project=WANDB_PROJECT,
        name=args.run_name or 'il_pretrain',
        config={
            'algo':       'IL (supervised) → RecurrentPPO init',
            'demo_steps': args.demo_steps,
            'epochs':     args.epochs,
            'lr':         args.lr,
            'value_coef': args.value_coef,
            'gamma':      args.gamma,
            'bc_action':  BC_ACTION,
            'opponents':  'RuleBased (2-player)',
            'action_space': 'Discrete(6)',
        },
    )

    env = make_env(cfg, seed=0)

    # Build the same RecurrentPPO architecture as scripts/train_ppo.py so the
    # checkpoint is drop-in loadable by the RL fine-tuning stage.
    model = RecurrentPPO(
        'MlpLstmPolicy',
        env,
        learning_rate=args.lr,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            lstm_hidden_size=256,
            n_lstm_layers=1,
            enable_critic_lstm=True,
        ),
        tensorboard_log=f'logs/{run.id}' if args.no_wandb is False else None,
        verbose=0,
        device='cuda' if th.cuda.is_available() else 'cpu',
    )

    print(f'Obs size: {env.observation_space.shape[0]}  |  Action space: {env.action_space}')
    print(f'[IL] Collecting {args.demo_steps:,} demonstration steps '
          f'(BC_ACTION={BC_ACTION}, 2-player vs RuleBased)...')
    episodes_obs, episodes_rew = collect_episodes(env, args.demo_steps)
    n_eps   = len(episodes_obs)
    n_trans = sum(e.shape[0] for e in episodes_obs)
    print(f'[IL] Collected {n_eps} episodes, {n_trans:,} transitions '
          f'(avg episode length: {n_trans / max(n_eps, 1):.1f})')
    wandb.log({'il/episodes': n_eps, 'il/transitions': n_trans})

    print(f'[IL] Supervised training ({args.epochs} epochs, lr={args.lr}, '
          f'value_coef={args.value_coef})...')
    il_train(model, episodes_obs, episodes_rew,
             n_epochs=args.epochs, lr=args.lr,
             value_coef=args.value_coef, gamma=args.gamma,
             eval_cfg=cfg if args.eval_games > 0 else None,
             eval_games=args.eval_games,
             eval_seed=args.eval_seed,
             eval_trace_dir=args.eval_trace_dir or None)

    model.save(args.out)
    print(f'[IL] Saved IL checkpoint to {args.out}.zip')
    print(f'W&B run: {run.url}')
    run.finish()


if __name__ == '__main__':
    main()
