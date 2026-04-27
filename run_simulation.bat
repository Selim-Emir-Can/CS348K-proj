cd C:\Users\emir2\Desktop\CS349K_proj\monopoly
set PYTHONPATH=. && python scripts/simulate_from_config.py configs\default
set PYTHONPATH=. && python scripts/compare_env.py --n 200 2>&1
echo.
echo === 10K-game benchmark ===
set PYTHONPATH=. && python scripts/benchmark_env.py

echo.
echo === PettingZoo env smoke test ===
set PYTHONPATH=. && python -c "
from config import GameConfig
from monopoly_env import MonopolyEnv

cfg = GameConfig.from_yaml('default_config.yaml')
env = MonopolyEnv(cfg, seed=0)
env.reset()

steps = 0
while env.agents:
    agent = env.agent_selection
    if env.terminations[agent] or env.truncations[agent]:
        env.step(None)
    else:
        env.step(0)
    steps += 1

print(f'Episode finished in {steps} steps, round {env._round}')
print('Final rewards:', env._cumulative_rewards)
"