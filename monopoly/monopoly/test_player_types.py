
from config import GameConfig
from monopoly_env import MonopolyEnv

cfg = GameConfig.from_yaml('default_config.yaml')
cfg.players = [
    {'name': 'RuleBased', 'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500},
    {'name': 'Standard',  'settings': 'StandardPlayerSettings',  'starting_money': 1500},
]
wins = {'RuleBased': 0, 'Standard': 0}
env = MonopolyEnv(cfg)
for i in range(2000):
    env.reset(seed=i)
    while env.agents:
        a = env.agent_selection
        env.step(None if env.terminations[a] or env.truncations[a] else 0)
    alive = [a for a in env.possible_agents if not env._players[env._name_to_idx[a]].is_bankrupt]
    if len(alive) == 1:
        wins[alive[0]] += 1
print(wins)



# # All purchasable property indices on the standard board
# ALL_PROPERTIES = [1,3,5,6,8,9,11,12,13,14,15,16,18,19,21,23,24,25,26,27,28,29,31,32,34,35,37,39]

# cfg = GameConfig.from_yaml('default_config.yaml')
# cfg.players = [
#     {'name': 'RuleBased', 'settings': 'RuleBasedPlayerSettings', 'starting_money': 1500,
#      'starting_properties': ALL_PROPERTIES},
#     {'name': 'Standard',  'settings': 'StandardPlayerSettings',  'starting_money': 1500},
# ]
