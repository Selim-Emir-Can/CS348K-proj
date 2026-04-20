"""PettingZoo AEC environment wrapping the Monopoly game loop.

Typical usage::

    from config import GameConfig
    from monopoly_env import MonopolyEnv

    env = MonopolyEnv(GameConfig.from_yaml("default_config.yaml"), seed=42)
    env.reset()
    while env.agents:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
        else:
            env.step(env.action_space(agent).sample())
"""
import functools

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from monopoly.core.game import setup_game_from_config, setup_players_from_config
from monopoly.core.move_result import MoveResult
from monopoly.core.cell import Property
from settings import SimulationSettings

# Board always has exactly 40 cells in the standard layout.
N_CELLS = 40
# Soft normalisation ceiling for money / net-worth features.
_MAX_MONEY = 10_000.0


# --------------------------------------------------------------------------- #
# Observation helper                                                            #
# --------------------------------------------------------------------------- #

def _make_obs(players, board, self_idx: int) -> np.ndarray:
    """Build a flat float32 observation vector from *self_idx*'s perspective.

    Layout (n = number of players)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [0 .. 5n-1]      per-player block (self first, others follow in index order)
        slot*5 + 0   money        / _MAX_MONEY   (clipped to [-1, 1])
        slot*5 + 1   net_worth()  / _MAX_MONEY
        slot*5 + 2   position     / N_CELLS
        slot*5 + 3   in_jail      (0 / 1)
        slot*5 + 4   is_bankrupt  (0 / 1)
    [5n .. 5n+2*N_CELLS-1]   per-cell block
        cell*2 + 0   owner encoding: -1=unowned, 0=self, k/n=opponent-slot-k
        cell*2 + 1   development:  (has_houses + 5*has_hotel) / 5
    """
    n = len(players)
    obs = np.zeros(5 * n + 2 * N_CELLS, dtype=np.float32)

    # Player features — reorder so self is always slot 0
    order = [self_idx] + [i for i in range(n) if i != self_idx]
    for slot, pidx in enumerate(order):
        p = players[pidx]
        base = slot * 5
        obs[base]     = float(np.clip(p.money / _MAX_MONEY, -1.0, 1.0))
        obs[base + 1] = float(np.clip(p.net_worth() / _MAX_MONEY, -1.0, 1.0))
        obs[base + 2] = p.position / N_CELLS
        obs[base + 3] = float(p.in_jail)
        obs[base + 4] = float(p.is_bankrupt)

    # Cell features
    cell_base = 5 * n
    other_indices = [i for i in range(n) if i != self_idx]
    for i, cell in enumerate(board.cells):
        if not isinstance(cell, Property):
            obs[cell_base + 2 * i] = -1.0   # non-purchasable → treat as unowned
            continue
        if cell.owner is None:
            obs[cell_base + 2 * i] = -1.0
        elif cell.owner is players[self_idx]:
            obs[cell_base + 2 * i] = 0.0
        else:
            opp_slot = next(
                (s + 1 for s, oi in enumerate(other_indices)
                 if cell.owner is players[oi]),
                -1,
            )
            obs[cell_base + 2 * i] = opp_slot / n if opp_slot >= 0 else -1.0
        obs[cell_base + 2 * i + 1] = (cell.has_houses + 5 * cell.has_hotel) / 5.0

    return obs


# --------------------------------------------------------------------------- #
# Environment                                                                   #
# --------------------------------------------------------------------------- #

class MonopolyEnv(AECEnv):
    """PettingZoo AEC environment for the Monopoly simulator.

    Each call to step() runs one player's complete turn using their built-in
    strategy (Player.make_a_move).  The *action* argument is currently unused
    and reserved for future decision-point hooks (buy/no-buy, jail choices, etc.).
    """

    metadata = {"render_modes": ["human"], "name": "monopoly_v0"}

    def __init__(
        self,
        game_config=None,
        seed: int = 0,
        max_turns: int = None,
        render_mode=None,
    ):
        super().__init__()
        from config import GameConfig

        self._cfg = game_config if game_config is not None else GameConfig()
        self._base_seed = seed
        self._max_turns = (
            max_turns if max_turns is not None else SimulationSettings.n_moves
        )
        self.render_mode = render_mode

        # Derive agent names from the config (needed before the first reset)
        if self._cfg.players:
            self.possible_agents = [p["name"] for p in self._cfg.players]
        else:
            self.possible_agents = [
                name for name, _ in self._cfg.settings.players_list
            ]

        n = len(self.possible_agents)
        obs_size = 5 * n + 2 * N_CELLS

        self.observation_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for a in self.possible_agents
        }
        # Discrete(1) is a placeholder; will expand to real decision actions later.
        self.action_spaces = {
            a: spaces.Discrete(1) for a in self.possible_agents
        }

    # ------------------------------------------------------------------ #
    # PettingZoo AEC interface                                             #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._base_seed = seed

        # Reuse existing setup helpers; disable both logs to avoid file I/O
        # during training (KEEP_GAME_LOG defaults to True in LogSettings).
        self.board, self.dice, self._elog, self._blog = setup_game_from_config(
            0, self._base_seed, self._cfg
        )
        self._elog.disabled = True
        self._blog.disabled = True

        self._players = setup_players_from_config(self.board, self.dice, self._cfg)

        self.agents = list(self.possible_agents)
        # Rebuild after shuffle: _players order may differ from possible_agents order.
        self._name_to_idx = {p.name: i for i, p in enumerate(self._players)}
        self._round = 0

        self.terminations        = {a: False for a in self.possible_agents}
        self.truncations         = {a: False for a in self.possible_agents}
        self.rewards             = {a: 0.0   for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0   for a in self.possible_agents}
        self.infos               = {a: {}    for a in self.possible_agents}

        self._prev_nw = {
            a: self._players[self._name_to_idx[a]].net_worth()
            for a in self.possible_agents
        }

        self.agent_selection = self.possible_agents[0]

    def observe(self, agent: str) -> np.ndarray:
        return _make_obs(self._players, self.board, self._name_to_idx[agent])

    def step(self, action):
        agent = self.agent_selection

        # Null-step pattern: caller signals a dead agent is acknowledged.
        if self.terminations[agent] or self.truncations[agent]:
            self.rewards[agent] = 0.0
            if agent in self.agents:
                self.agents.remove(agent)
            self._advance_agent()
            return

        idx = self._player_idx(agent)
        player = self._players[idx]
        prev_nw = self._prev_nw[agent]

        # Run the player's full turn via existing game logic.
        move_result = player.make_a_move(
            self.board, self._players, self.dice, self._elog
        )
        if move_result == MoveResult.BANKRUPT:
            self._blog.add(f"0\t{player}\t{self._round}")

        # Sync bankruptcy flags for all players — a single turn can cascade
        # (e.g. the acting player cannot pay rent and goes bankrupt).
        for a in self.possible_agents:
            p = self._players[self._name_to_idx[a]]
            if p.is_bankrupt and not self.terminations[a]:
                self.terminations[a] = True

        # Shaped reward: change in net worth this turn.
        cur_nw = player.net_worth()
        self.rewards[agent] = (cur_nw - prev_nw) / 1000.0
        if self.terminations[agent] and player.is_bankrupt:
            self.rewards[agent] -= 5.0   # bankruptcy penalty
        self._prev_nw[agent] = cur_nw
        self._cumulative_rewards[agent] += self.rewards[agent]

        # Win condition: fewer than 2 players remain.
        alive = [p for p in self._players if not p.is_bankrupt]
        if len(alive) < 2:
            for a in self.possible_agents:
                p = self._players[self._name_to_idx[a]]
                if not p.is_bankrupt and not self.terminations[a]:
                    self.rewards[a] += 10.0          # winner bonus
                    self._cumulative_rewards[a] += 10.0
                    self.terminations[a] = True

        # Hard turn-limit truncation.
        if self._round >= self._max_turns:
            for a in self.possible_agents:
                if not self.terminations[a]:
                    self.truncations[a] = True

        self._advance_agent()

        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode != "human":
            return
        alive = [p for p in self._players if not p.is_bankrupt]
        print(f"[Round {self._round}] Alive: {len(alive)}/{len(self._players)}")
        for p in self._players:
            if p.is_bankrupt:
                print(f"  {p.name}: BANKRUPT")
            else:
                print(
                    f"  {p.name}: ${p.money}  net ${p.net_worth()}"
                    f"  pos {p.position}"
                )

    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self.action_spaces[agent]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _player_idx(self, agent: str) -> int:
        return self._name_to_idx[agent]

    def _advance_agent(self):
        """Advance agent_selection to the next agent still in self.agents.

        A round completes whenever the turn order wraps past the end of
        possible_agents (i.e. current_idx + offset >= n).  This fires correctly
        even when some agents are dead and only receive null steps, because the
        wrap is detected by index position rather than a step counter.
        """
        if not self.agents:
            return
        n = len(self.possible_agents)
        try:
            start = self.possible_agents.index(self.agent_selection)
        except ValueError:
            start = 0
        for offset in range(1, n + 1):
            candidate = self.possible_agents[(start + offset) % n]
            if candidate in self.agents:
                if start + offset >= n:   # wrapped around → new round
                    self._on_round_complete()
                self.agent_selection = candidate
                return

    def _on_round_complete(self):
        self._round += 1
        # "All rich" stalemate: if every surviving player is above the cash
        # threshold the game will never end; truncate all.
        alive = [p for p in self._players if not p.is_bankrupt]
        if alive and all(
            p.money > SimulationSettings.never_bankrupt_cash for p in alive
        ):
            for a in self.possible_agents:
                if not self.terminations[a]:
                    self.truncations[a] = True


# --------------------------------------------------------------------------- #
# Single-agent Gym wrapper for SB3 training                                    #
# --------------------------------------------------------------------------- #

class SingleAgentMonopolyEnv(gym.Env):
    """Gym wrapper around MonopolyEnv for training one agent with SB3.

    One named agent (the learner) faces N-1 opponents that act via their
    built-in strategy.  Each step() corresponds to one full turn for the
    learner.

    action_space  : Discrete(2) — 0 = don't buy, 1 = buy
    observation   : same flat float32 vector as MonopolyEnv.observe()
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, game_config=None, agent_name: str = None,
                 seed: int = 0, max_turns: int = None, render_mode=None):
        super().__init__()
        self._multi = MonopolyEnv(game_config, seed=seed,
                                  max_turns=max_turns, render_mode=render_mode)
        self._agent = agent_name or self._multi.possible_agents[0]
        self.render_mode = render_mode

        n = len(self._multi.possible_agents)
        obs_size = 5 * n + 2 * N_CELLS
        self.observation_space = spaces.Box(-1.0, 1.0,
                                            shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0=pass, 1=buy

    # ------------------------------------------------------------------ #
    # Gym interface                                                         #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        self._multi.reset(seed=seed)
        self._advance_opponents()
        return self._multi.observe(self._agent), {}

    def step(self, action: int):
        # Inject the buy decision into the learner before its turn runs.
        idx = self._multi._name_to_idx[self._agent]
        player = self._multi._players[idx]
        if hasattr(player, '_buy_action'):
            player._buy_action = int(action)

        self._multi.step(action)
        self._advance_opponents()

        terminated = self._multi.terminations.get(self._agent, False)
        truncated  = self._multi.truncations.get(self._agent, False)
        done       = terminated or truncated or not self._multi.agents
        reward     = self._multi.rewards.get(self._agent, 0.0)
        obs        = self._multi.observe(self._agent)
        info       = self._multi.infos.get(self._agent, {})

        return obs, reward, done, False, info

    def render(self):
        self._multi.render()

    def close(self):
        self._multi.close()

    # ------------------------------------------------------------------ #
    # Internal                                                              #
    # ------------------------------------------------------------------ #

    def _advance_opponents(self):
        """Step all agents until it is the learner's turn (or game over)."""
        env = self._multi
        while env.agents and env.agent_selection != self._agent:
            a = env.agent_selection
            if env.terminations[a] or env.truncations[a]:
                env.step(None)
            else:
                env.step(0)
        # If the learner is now terminated/truncated, consume its null step too.
        if (env.agents and env.agent_selection == self._agent and
                (env.terminations.get(self._agent) or
                 env.truncations.get(self._agent))):
            env.step(None)
