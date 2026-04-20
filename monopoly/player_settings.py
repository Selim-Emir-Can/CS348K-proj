"""Player behavior settings.

All player strategy settings live here, separate from game/simulation settings
(GameMechanics, SimulationSettings) which live in settings.py.
"""
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class StandardPlayerSettings:
    unspendable_cash: int = 200  # Amount of money the player wants to keep unspent (money safety pillow)
    ignore_property_groups: FrozenSet[str] = frozenset()  # Group of properties do not buy, i.e.{"RED", "GREEN"}

    is_willing_to_make_trades: bool = True
    # agree to trades if the value difference is within these limits:
    trade_max_diff_absolute: int = 200  # More expensive - less expensive
    trade_max_diff_relative: float = 2.0  # More expensive / less expensive


@dataclass(frozen=True)
class HeroPlayerSettings(StandardPlayerSettings):
    """ here you can change the settings of the hero (the Experimental Player) """
    # ignore_property_groups: FrozenSet[str] = frozenset({"GREEN"})


@dataclass(frozen=True)
class RuleBasedPlayerSettings(StandardPlayerSettings):
    """Buy every affordable property, build houses immediately, keep no cash reserve."""
    unspendable_cash: int = 0
    is_willing_to_make_trades: bool = False


@dataclass(frozen=True)
class RandomPlayerSettings(StandardPlayerSettings):
    """Settings for RandomPlayer; actual buy decisions are randomised in the Player subclass."""
    unspendable_cash: int = 0
    is_willing_to_make_trades: bool = False
