"""Agent population for the Monopoly optimizer.

Three agents in increasing order of skill:
  RandomPlayer    -- buys each unowned property with fixed probability
  RuleBasedPlayer -- always buy, always build (via RuleBasedPlayerSettings in settings.py)
  (DQN agent in Week 3)

Settings classes live in settings.py alongside StandardPlayerSettings.
Player subclasses (requiring custom logic beyond settings) live here.
"""
import random as _random

from monopoly.core.player import Player
from player_settings import RandomPlayerSettings


class DQNPlayer(Player):
    """Player whose buy decision is injected by an external RL agent each step.

    Set player._buy_action = 0 or 1 before calling make_a_move().
    All other decisions (houses, trading, jail) follow standard settings.
    """

    def __init__(self, name: str, settings=None):
        super().__init__(name, settings or RandomPlayerSettings())
        self._buy_action: int = 1  # default: buy; overwritten each step by the env

    def _should_buy(self, property_to_buy) -> bool:
        if property_to_buy.cost_base > self.money:
            return False
        return bool(self._buy_action)


class RandomPlayer(Player):
    """Buys each unowned property it can afford with probability *buy_probability*.

    All other behaviour (building, trading, bankruptcy) inherits from Player.
    """

    def __init__(self, name: str, settings=None, buy_probability: float = 0.5,
                 seed: int = None):
        super().__init__(name, settings or RandomPlayerSettings())
        self._rng = _random.Random(seed)
        self._buy_probability = buy_probability

    def _should_buy(self, property_to_buy) -> bool:
        if property_to_buy.cost_base > self.money:
            return False
        return self._rng.random() < self._buy_probability
