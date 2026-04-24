"""Check action distribution of a trained DQN model."""
from stable_baselines3 import DQN
from config import GameConfig
from monopoly_env import SingleAgentMonopolyEnv

cfg = GameConfig.from_yaml('default_config.yaml')
cfg.players = [
    {'name': 'DQN', 'settings': 'StandardPlayerSettings', 'player_class': 'DQNPlayer', 'starting_money': 1500},
    {'name': 'RuleBased1', 'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
    {'name': 'RuleBased2', 'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
    {'name': 'RuleBased3', 'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
]
model = DQN.load('models/best_model')
env = SingleAgentMonopolyEnv(cfg, agent_name='DQN', seed=0)
obs, _ = env.reset()
counts = [0] * 6
for _ in range(2000):
    a, _ = model.predict(obs, deterministic=True)
    counts[int(a)] += 1
    obs, _, done, _, _ = env.step(int(a))
    if done:
        obs, _ = env.reset()

labels = ['no-buy/hold', 'buy/hold', 'no-buy/mod', 'buy/mod', 'no-buy/aggr', 'buy/aggr']
total = sum(counts)
print('Action distribution (2000 steps):')
for i, (label, count) in enumerate(zip(labels, counts)):
    print(f'  {i} {label:15s}: {count:4d}  ({100*count/total:.1f}%)')
