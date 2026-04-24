"""Agent population for the Monopoly optimizer.

Three agents in increasing order of skill:
  RandomPlayer    -- buys each unowned property with fixed probability
  RuleBasedPlayer -- always buy, always build (via RuleBasedPlayerSettings in settings.py)
  (DQN agent in Week 3)

Settings classes live in settings.py alongside StandardPlayerSettings.
Player subclasses (requiring custom logic beyond settings) live here.
"""
import random as _random

from monopoly.core.constants import RAILROADS, UTILITIES
from monopoly.core.player import Player
from player_settings import ParametricPlayerSettings, RandomPlayerSettings


_BUILD_THRESHOLDS = [500, 200, 0]  # indexed by action // 2


class DQNPlayer(Player):
    """Player whose buy and build decisions are injected by an external RL agent.

    Action encoding (Discrete 6):
      action % 2  -> buy decision (0=pass, 1=buy)
      action // 2 -> build level  (0=hold $500, 1=moderate $200, 2=aggressive $0)

    Set player._buy_action and player._build_threshold before calling make_a_move().
    """

    def __init__(self, name: str, settings=None):
        super().__init__(name, settings or RandomPlayerSettings())
        self._buy_action: int = 1       # overwritten each step by the env
        self._build_threshold: int = 200  # overwritten each step by the env

    @property
    def _improve_cash_floor(self) -> int:
        return self._build_threshold

    def _should_buy(self, property_to_buy) -> bool:
        if property_to_buy.cost_base > self.money:
            return False
        if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
            return False
        return bool(self._buy_action)


class ParametricPlayer(Player):
    """Rule-based player with the full parametric knob-set for the optimiser.

    Every decision point the optimiser might want to vary is gated by a field on
    ``ParametricPlayerSettings``. Behaviours not yet expressible in the base
    Player (aggressive-vs-one-at-a-time building, skip-utilities/skip-railroads,
    early jail exit) are implemented here; everything else (trading thresholds,
    unspendable_cash, ignore_property_groups) is inherited from
    StandardPlayerSettings semantics via base-class logic.
    """

    def __init__(self, name: str, settings=None):
        super().__init__(name, settings or ParametricPlayerSettings())

    # ------------------------------------------------------------------ #
    # Buy decision                                                          #
    # ------------------------------------------------------------------ #
    def _should_buy(self, property_to_buy) -> bool:
        # Base affordability + ignored-groups check from parent
        if property_to_buy.cost_base > self.money:
            return False
        if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
            return False
        if property_to_buy.group in self.settings.ignore_property_groups:
            return False
        # Parametric knobs
        if property_to_buy.group == UTILITIES and not getattr(
                self.settings, 'buy_utilities', True):
            return False
        if property_to_buy.group == RAILROADS and not getattr(
                self.settings, 'buy_railroads', True):
            return False
        return True

    # ------------------------------------------------------------------ #
    # Build decision                                                        #
    # ------------------------------------------------------------------ #
    @property
    def _improve_cash_floor(self) -> int:
        return getattr(self.settings, 'build_cash_floor',
                       self.settings.unspendable_cash)

    def improve_properties(self, board, log):
        """If aggressive_build is False, build at most 1 house per turn.

        This is a targeted override of ``Player.improve_properties`` (core/player.py).
        When aggressive, we just delegate to super(). Otherwise we inline the
        one-iteration version, kept in sync with core/player.py:528-602.
        """
        if getattr(self.settings, 'aggressive_build', True):
            return super().improve_properties(board, log)

        # Non-aggressive: exactly one build attempt per turn.
        from monopoly.core.constants import RAILROADS as _RR, UTILITIES as _UT
        can_be_improved = []
        for cell in self.owned:
            if (cell.has_hotel == 0
                    and not cell.is_mortgaged
                    and cell.monopoly_multiplier == 2
                    and cell.group not in (_RR, _UT)):
                eligible = True
                for other_cell in board.groups[cell.group]:
                    if ((other_cell.has_houses < cell.has_houses
                         and not other_cell.has_hotel)
                            or other_cell.is_mortgaged):
                        eligible = False
                        break
                if eligible:
                    if cell.has_houses != 4 and board.available_houses > 0 or \
                            cell.has_houses == 4 and board.available_hotels > 0:
                        can_be_improved.append(cell)
        if not can_be_improved:
            return
        can_be_improved.sort(key=lambda x: x.cost_house)
        cell_to_improve = can_be_improved[0]
        if self.money - cell_to_improve.cost_house < self._improve_cash_floor:
            return
        ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        if cell_to_improve.has_houses != 4:
            cell_to_improve.has_houses += 1
            board.available_houses -= 1
            self.money -= cell_to_improve.cost_house
            log.add(f"{self} built {ordinal[cell_to_improve.has_houses]} "
                    f"house on {cell_to_improve} for ${cell_to_improve.cost_house}")
        else:
            cell_to_improve.has_houses = 0
            cell_to_improve.has_hotel = 1
            board.available_houses += 4
            board.available_hotels -= 1
            self.money -= cell_to_improve.cost_house
            log.add(f"{self} built a hotel on {cell_to_improve}")

    # ------------------------------------------------------------------ #
    # Jail decision                                                         #
    # ------------------------------------------------------------------ #
    def handle_jail(self, dice_roll_is_double, board, log):
        """Extend base jail handling with an early-pay option.

        Parent (`core/player.py:handle_jail`) only pays the fine after the
        mandatory 2-turn wait. If ``jail_pay_threshold`` is set and the player
        has comfortably more cash than that threshold, pay on day 0 or 1 too
        — it's strategically useful when opponents have few developed cells.
        """
        from settings import GameMechanics
        threshold = getattr(self.settings, 'jail_pay_threshold', None)
        if (threshold is not None
                and self.in_jail
                and not self.get_out_of_jail_chance
                and not self.get_out_of_jail_comm_chest
                and not dice_roll_is_double
                and self.days_in_jail < 2
                and self.money > threshold + GameMechanics.exit_jail_fine):
            log.add(f"{self} chooses to pay ${GameMechanics.exit_jail_fine} "
                    f"and exit jail early (cash > threshold={threshold})")
            self.pay_money(GameMechanics.exit_jail_fine, "bank", board, log)
            self.in_jail = False
            self.days_in_jail = 0
            return False
        return super().handle_jail(dice_roll_is_double, board, log)


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
