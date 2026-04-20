import random
import sys
from functools import partial
from typing import Type

from tqdm.contrib.concurrent import process_map

from monopoly.analytics import AnalyzerFromConfig
from monopoly.core.game import monopoly_game_from_config
from monopoly.log_settings import LogSettings
from config import GameConfig
from settings import SimulationSettings


def run_simulation_from_config(game_config: GameConfig, sim_settings: Type[SimulationSettings] = SimulationSettings) -> None:
    """Simulate N games from a GameConfig in parallel, then print an analysis."""
    LogSettings.init_logs()

    master_rng = random.Random(sim_settings.seed)
    game_seed_pairs = [(i + 1, master_rng.getrandbits(32)) for i in range(sim_settings.n_games)]

    process_map(
        partial(monopoly_game_from_config, game_config=game_config),
        game_seed_pairs,
        max_workers=sim_settings.multi_process,
        total=sim_settings.n_games,
        desc="Simulating Monopoly games",
    )

    AnalyzerFromConfig(game_config, sim_settings).run_all()


if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = GameConfig.from_yaml(yaml_path)
    run_simulation_from_config(cfg, SimulationSettings)
