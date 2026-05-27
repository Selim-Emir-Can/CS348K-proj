"""Human-playable Monopoly on an (optimised) board, via a local web UI.

A variant of the simulation where the buy / build / jail decisions are made by
humans in a browser instead of by an agent. The full game is recorded to a log
file (every engine event plus every human decision).

It reuses the existing engine unchanged: a normal game loop runs in a background
thread, and a ``HumanPlayer`` blocks that thread on each decision until the
browser POSTs an answer. A stdlib HTTP server (no Flask dependency) serves the
UI, the current state (polled as JSON), and accepts decisions.

Usage (from monopoly/, with the conda env active and PYTHONPATH=.):

    # The 2-player GA-optimised winner board
    python scripts/play_web.py --runs logs/optimizer_v3/ga_2p_mask.jsonl \
        --n-players 2 --port 8000 --log logs/human_play/ga2p_game1.log

    # The length-only ablation 2p board
    python scripts/play_web.py --runs logs/optimizer_v3/abl_len_2p_mask.jsonl \
        --n-players 2 --port 8000 --log logs/human_play/ablen2p_game1.log

    # The default (unmodified) board, for comparison
    python scripts/play_web.py --identity --n-players 2 --port 8000

Then open http://localhost:8000 in a browser. Two players share one screen
(hot-seat); the page shows whose turn it is and what decision is pending.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# Make ``monopoly/`` (parent) and ``scripts/`` (self) importable regardless of
# how this file is launched (``python scripts/play_web.py`` or imported).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from config import GameConfig
from optimizer.design_space import DesignSpace
from monopoly.core.board import Board
from monopoly.core.dice import Dice
from monopoly.core.cell import Property, FreeParking
from monopoly.core.player import Player
from monopoly.core.move_result import MoveResult
from monopoly.core.game_utils import _check_end_conditions
from player_settings import StandardPlayerSettings
from settings import GameMechanics


# ----------------------------------------------------------------------------- #
# Game session: shared state between the game thread and the web threads.       #
# ----------------------------------------------------------------------------- #

class GameSession:
    """Thread-safe bridge between the (blocking) game loop and the web UI."""

    def __init__(self, log_path: str):
        self._lock = threading.Lock()
        self._answer_event = threading.Event()
        self._answer = None

        self.snapshot: dict = {}      # latest board/player state for the UI
        self.pending: dict | None = None  # current decision request, or None
        self.events: list[str] = []   # rolling event log (also written to file)
        self.finished = False
        self.result_text = ""
        self.board_fracs = None   # [[left%, top%], ...] per cell, for token overlay
        self.center_frac = None   # [left%, top%] of board centre, for badge offset
        self.last_roll = None     # {"roller": name, "cast": [...], "total": int, "double": bool}
        self.active_player = None  # whose make_a_move is currently executing
        self.decision_seq = 0      # counts real (buy/build/jail) human decisions

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any previous log at this path.
        self._fh = open(self.log_path, "w", encoding="utf-8")

    # -- logging ------------------------------------------------------------- #
    def log(self, line: str):
        line = str(line).rstrip("\n")
        with self._lock:
            self.events.append(line)
            self._fh.write(line + "\n")
            self._fh.flush()

    # -- snapshot ------------------------------------------------------------ #
    def set_snapshot(self, snap: dict):
        with self._lock:
            self.snapshot = snap

    def set_active_player(self, name):
        with self._lock:
            self.active_player = name

    def clear_dice(self):
        with self._lock:
            self.last_roll = None

    def set_dice(self, cast, total, is_double):
        with self._lock:
            self.last_roll = {"roller": self.active_player, "cast": list(cast),
                              "total": int(total), "double": bool(is_double)}

    # -- decision request/response (called from the game thread) ------------- #
    def request_decision(self, decision: dict):
        """Publish a pending decision and BLOCK the game thread until answered."""
        with self._lock:
            self.pending = decision
            self._answer = None
            self._answer_event.clear()
        self._answer_event.wait()
        with self._lock:
            ans = self._answer
            self.pending = None
        return ans

    # -- answer submission (called from a web thread) ------------------------ #
    def request_player_decision(self, decision: dict):
        """Like request_decision, but marks that a *real* human decision
        (buy / build / jail) was made this turn — used to decide whether a
        turn needs a between-turn recap pause."""
        with self._lock:
            self.decision_seq += 1
        return self.request_decision(decision)

    def submit_answer(self, value):
        with self._lock:
            self._answer = value
        self._answer_event.set()

    # -- state for the UI ---------------------------------------------------- #
    def get_state(self) -> dict:
        with self._lock:
            return {
                "snapshot": self.snapshot,
                "pending": self.pending,
                "events": self.events[-60:],
                "finished": self.finished,
                "result": self.result_text,
                "layout": self.board_fracs,
                "center": self.center_frac,
                "dice": self.last_roll,
            }

    def finish(self, result_text: str):
        with self._lock:
            self.finished = True
            self.result_text = result_text
        self.log("=== GAME OVER: " + result_text + " ===")


# ----------------------------------------------------------------------------- #
# Log proxy: lets the engine's ``log.add(...)`` calls flow into the session.    #
# ----------------------------------------------------------------------------- #

class LogProxy:
    """Minimal stand-in for monopoly.log.Log; mirrors every event into the
    GameSession (and thus the log file + the UI)."""

    def __init__(self, session: GameSession):
        self.session = session
        self.content = True  # satisfies callers that check ``if log.content``

    def add(self, line=""):
        if str(line).strip():
            self.session.log(line)

    def save(self):
        pass


class DiceUI:
    """Wraps the engine ``Dice`` so each roll is surfaced to the UI; delegates
    everything else (shuffle, attributes) to the real dice object."""

    def __init__(self, dice, session: GameSession):
        self._dice = dice
        self._session = session

    def roll(self):
        cast, total, is_double = self._dice.roll()
        try:
            self._session.set_dice(cast, total, is_double)
        except Exception:
            pass
        return cast, total, is_double

    def __getattr__(self, name):
        # Only reached for attributes not found on DiceUI itself (e.g. shuffle).
        return getattr(self._dice, name)


# ----------------------------------------------------------------------------- #
# Snapshot of board + players for the UI.                                       #
# ----------------------------------------------------------------------------- #

def _cell_dict(cell, index):
    tn = type(cell).__name__
    d = {"index": index, "name": cell.name, "type": tn}
    if isinstance(cell, Property):
        d.update({
            "group": cell.group,
            "cost": int(cell.cost_base),
            "rent": int(cell.rent_base),
            "owner": cell.owner.name if cell.owner is not None else None,
            "houses": getattr(cell, "has_houses", 0),
            "hotel": getattr(cell, "has_hotel", 0),
            "mortgaged": getattr(cell, "is_mortgaged", False),
        })
    return d


def build_snapshot(board, players, current_player_name, turn_n):
    cells = [_cell_dict(c, i) for i, c in enumerate(board.cells)]
    pdata = []
    for p in players:
        pos = p.position % len(board.cells)
        pdata.append({
            "name": p.name,
            "money": int(p.money),
            "position": pos,
            "cell": board.cells[pos].name,
            "in_jail": bool(p.in_jail),
            "on_jail_cell": bool(getattr(board, "jail_index", None) is not None
                                 and pos == board.jail_index),
            "bankrupt": bool(p.is_bankrupt),
            "net_worth": int(p.net_worth()),
            "owned": sorted(c.name for c in p.owned),
        })
    return {
        "cells": cells,
        "players": pdata,
        "current": current_player_name,
        "turn": turn_n,
        "n_cells": len(board.cells),
    }


# ----------------------------------------------------------------------------- #
# Human player: routes decisions to the web UI.                                 #
# ----------------------------------------------------------------------------- #

class HumanPlayer(Player):
    """Player whose buy / build / jail choices come from the web UI.

    Auto-trading is disabled (the user opted out of interactive trades), so
    ``do_a_two_way_trade`` always declines.
    """

    def __init__(self, name, settings, session: GameSession):
        super().__init__(name, settings)
        self.session = session
        self._board_ref = None
        self._players_ref = None
        self._turn_n = 0

    # No automatic trading in human games.
    def do_a_two_way_trade(self, players, board, log):
        return False

    def _publish(self, extra_pending=None):
        snap = build_snapshot(self._board_ref, self._players_ref,
                              self.name, self._turn_n)
        self.session.set_snapshot(snap)

    def make_a_move(self, board, players, dice, log):
        self._board_ref = board
        self._players_ref = players
        # Blank the dice readout at the start of this player's turn so the
        # pre-roll build phase doesn't display a stale roll from someone else;
        # the roll that follows is tagged with this player's name.
        self.session.set_active_player(self.name)
        self.session.clear_dice()
        self._publish()
        return super().make_a_move(board, players, dice, log)

    # -- buy ----------------------------------------------------------------- #
    def _should_buy(self, property_to_buy) -> bool:
        prop = property_to_buy
        # Can't afford it: no choice to offer.
        if prop.cost_base > self.money:
            self.session.log(f"[decision] {self.name}: cannot afford "
                             f"{prop.name} (${int(prop.cost_base)} > "
                             f"${int(self.money)}); auto-pass")
            return False
        self._publish()
        decision = {
            "type": "buy",
            "player": self.name,
            "prompt": (f"{self.name}: buy {prop.name} ({prop.group}) for "
                       f"${int(prop.cost_base)}?  (rent ${int(prop.rent_base)}, "
                       f"you have ${int(self.money)})"),
            "options": [
                {"label": f"Buy for ${int(prop.cost_base)}", "value": "buy"},
                {"label": "Pass", "value": "pass"},
            ],
        }
        ans = self.session.request_player_decision(decision)
        bought = (ans == "buy")
        self.session.log(f"[decision] {self.name}: "
                         f"{'BUY' if bought else 'PASS'} {prop.name} "
                         f"(${int(prop.cost_base)})")
        return bought

    # -- build --------------------------------------------------------------- #
    def _eligible_to_build(self, board):
        """Cells this player may build one more house/hotel on right now
        (same eligibility rule as Player.improve_properties)."""
        from monopoly.core.constants import RAILROADS as _RR, UTILITIES as _UT
        out = []
        for cell in self.owned:
            if (cell.has_hotel == 0 and not cell.is_mortgaged
                    and cell.monopoly_multiplier == 2
                    and cell.group not in (_RR, _UT)):
                ok = True
                for other in board.groups[cell.group]:
                    if ((other.has_houses < cell.has_houses and not other.has_hotel)
                            or other.is_mortgaged):
                        ok = False
                        break
                if ok and ((cell.has_houses != 4 and board.available_houses > 0)
                           or (cell.has_houses == 4 and board.available_hotels > 0)):
                    out.append(cell)
        out.sort(key=lambda c: c.cost_house)
        return out

    def _do_build(self, cell, board, log):
        ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        if cell.has_houses != 4:
            cell.has_houses += 1
            board.available_houses -= 1
            self.money -= cell.cost_house
            log.add(f"{self} built {ordinal[cell.has_houses]} house on "
                    f"{cell} for ${cell.cost_house}")
        else:
            cell.has_houses = 0
            cell.has_hotel = 1
            board.available_houses += 4
            board.available_hotels -= 1
            self.money -= cell.cost_house
            log.add(f"{self} built a hotel on {cell} for ${cell.cost_house}")

    def improve_properties(self, board, log):
        """Let the human build houses/hotels, one at a time, until they finish
        or can no longer afford / have no eligible property.

        Called at the SAME point in the turn as the engine calls it for the
        agents (before the dice roll), so the human game mirrors the
        simulation's turn structure exactly — only the decision-maker differs.
        """
        while True:
            eligible = self._eligible_to_build(board)
            # Only offer cells the player can actually afford right now.
            affordable = [c for c in eligible if self.money - c.cost_house >= 0]
            if not affordable:
                return
            self._publish()
            opts = []
            for c in affordable:
                what = "hotel" if c.has_houses == 4 else f"house #{c.has_houses + 1}"
                opts.append({
                    "label": f"Build {what} on {c.name} (${int(c.cost_house)})",
                    "value": c.name,
                })
            opts.append({"label": "Finish building", "value": "__done__"})
            decision = {
                "type": "build",
                "player": self.name,
                "prompt": (f"{self.name}: build houses? (cash ${int(self.money)}) "
                           f"— pick a property or finish"),
                "options": opts,
            }
            ans = self.session.request_player_decision(decision)
            if ans == "__done__" or ans is None:
                return
            chosen = next((c for c in affordable if c.name == ans), None)
            if chosen is None:
                return
            self._do_build(chosen, board, log)

    # -- jail ---------------------------------------------------------------- #
    def is_player_stay_in_jail(self, dice_roll_is_double, board, log):
        """Offer an early-pay choice when in jail and not auto-released."""
        forced_out = (self.get_out_of_jail_chance or self.get_out_of_jail_comm_chest
                      or dice_roll_is_double or self.days_in_jail == 2)
        can_pay = self.money > GameMechanics.exit_jail_fine
        if self.in_jail and not forced_out and can_pay:
            self._publish()
            decision = {
                "type": "jail",
                "player": self.name,
                "prompt": (f"{self.name}: you're in jail (day {self.days_in_jail + 1}). "
                           f"Pay ${GameMechanics.exit_jail_fine} to leave now, or "
                           f"wait and try for a double?"),
                "options": [
                    {"label": f"Pay ${GameMechanics.exit_jail_fine} and leave",
                     "value": "pay"},
                    {"label": "Stay in jail", "value": "stay"},
                ],
            }
            ans = self.session.request_player_decision(decision)
            if ans == "pay":
                self.pay_money(GameMechanics.exit_jail_fine, "bank", board, log)
                self.in_jail = False
                self.days_in_jail = 0
                self.session.log(f"[decision] {self.name}: paid jail fine, leaving")
                return False
            self.session.log(f"[decision] {self.name}: staying in jail")
        return super().is_player_stay_in_jail(dice_roll_is_double, board, log)


# ----------------------------------------------------------------------------- #
# Board loading.                                                                #
# ----------------------------------------------------------------------------- #

def _best_vec_from_run(run_path):
    best = None
    with open(run_path) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if best is None or e["score"] < best["score"]:
                best = e
    return np.array(best["vec"])


def load_board_config(args):
    base = GameConfig.from_yaml(args.config)
    space = DesignSpace(base, removal_direction=args.removal_direction)
    if args.identity:
        return space.decode(space.identity_vec()), "default board"
    if args.runs:
        vec = _best_vec_from_run(args.runs)
        return space.decode(vec), Path(args.runs).stem
    if args.vec:
        vec = np.array([float(x) for x in args.vec.split(",")])
        return space.decode(vec), "custom vec"
    raise SystemExit("Pass --identity / --runs / --vec.")


def render_board_png(decoded_cfg, out_path):
    """Render the board to a PNG for the UI (reusing render_board's drawing).

    Returns ``(ok, cell_fracs)`` where ``cell_fracs[i] = [left_pct, top_pct]``
    is the centre of board cell ``i`` as a percentage of the saved image, so
    the browser can overlay live player tokens at the right spot without
    re-rendering. ``bbox_inches`` is NOT used (so the figure maps 1:1 onto the
    image and the fractions stay exact); an explicit axes rectangle leaves room
    for the colour legend at the bottom.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import render_board  # sibling module (scripts/ is on sys.path[0])
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_axes([0.03, 0.08, 0.94, 0.90])
        render_board.draw_board(ax, decoded_cfg, title="", font_scale=1.5)
        fig.legend(handles=render_board._legend_patches(), loc="lower center",
                   ncol=10, frameon=False, fontsize=14)
        fig.canvas.draw()  # realise transforms before querying them

        corners = render_board._find_corners(decoded_cfg.cells)
        n = len(decoded_cfg.cells)
        inv = fig.transFigure.inverted()
        cell_fracs = []
        for i in range(n):
            x, y, w, h, _side, _is_corner = render_board._cell_rect(i, corners, n)
            disp = ax.transData.transform((x + w / 2.0, y + h / 2.0))
            fx, fy = inv.transform(disp)             # figure fraction, origin bottom-left
            cell_fracs.append([float(round(fx * 100, 3)),
                               float(round((1 - fy) * 100, 3))])

        # Board centre as a figure fraction, so the UI can push ownership
        # badges inward (off the cell, into the open middle of the board).
        bot_n   = (corners[1] - corners[0] - 1) % n
        left_n  = (corners[2] - corners[1] - 1) % n
        top_n   = (corners[3] - corners[2] - 1) % n
        right_n = (corners[0] - corners[3] - 1) % n
        S = 2 * 1.5 + max(bot_n, left_n, top_n, right_n) * 1.0
        dcx, dcy = inv.transform(ax.transData.transform((S / 2.0, S / 2.0)))
        center_frac = [float(round(dcx * 100, 3)), float(round((1 - dcy) * 100, 3))]

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150)          # no bbox_inches → fractions exact
        plt.close(fig)
        return True, cell_fracs, center_frac
    except Exception as exc:  # rendering is a nice-to-have, never fatal
        print(f"[play_web] board PNG render failed ({exc}); UI will omit it.")
        return False, None, None


# ----------------------------------------------------------------------------- #
# The game thread.                                                              #
# ----------------------------------------------------------------------------- #

def run_game(session, board, dice, players, log, n_moves, board_label, seed=0):
    log.add(f"= HUMAN GAME on {board_label} ({len(players)} players) [seed={seed}] =")
    try:
        for turn_n in range(1, n_moves + 1):
            for p in players:
                p._turn_n = turn_n
            session.set_snapshot(build_snapshot(board, players, None, turn_n))
            if _check_end_conditions(players, log, 1, turn_n):
                break
            for player in players:
                if player.is_bankrupt:
                    continue
                ev0 = len(session.events)
                seq0 = session.decision_seq
                player.make_a_move(board, players, dice, log)
                session.set_snapshot(
                    build_snapshot(board, players, player.name, turn_n))
                # If this turn involved no human decision (e.g. a roll that
                # drew a "Go to Jail" card, or landed on a tax cell), pause
                # with a recap so the players can see what happened before
                # play moves on. Skipped once the game is effectively over.
                alive = sum(1 for q in players if not q.is_bankrupt)
                if session.decision_seq == seq0 and alive > 1:
                    session.request_decision({
                        "type": "continue",
                        "player": player.name,
                        "prompt": f"{player.name}'s turn (no decisions to make):",
                        "recap": list(session.events[ev0:]),
                        "options": [{"label": "Continue \\u25B6", "value": "ok"}],
                    })
        # Determine result.
        alive = [p for p in players if not p.is_bankrupt]
        if len(alive) == 1:
            session.finish(f"{alive[0].name} wins (last player standing).")
        else:
            ranked = sorted(alive, key=lambda p: p.net_worth(), reverse=True)
            board_state = ", ".join(f"{p.name}: ${p.net_worth()}" for p in ranked)
            leader = ranked[0].name if ranked else "nobody"
            session.finish(f"Turn limit reached. Leader by net worth: "
                           f"{leader}. ({board_state})")
    except Exception as exc:
        import traceback
        session.log("[error] game thread crashed:\n" + traceback.format_exc())
        session.finish(f"Game ended on error: {exc}")


# ----------------------------------------------------------------------------- #
# Web server.                                                                   #
# ----------------------------------------------------------------------------- #

PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Beta-Testing Monopoly — human play</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background:#fafafa; color:#111; }
  #wrap { display:flex; flex-wrap:wrap; gap:16px; padding:16px; align-items:flex-start; }
  #left { flex:0 0 auto; }
  #right { flex:1 1 360px; min-width:340px; }
  img#board { width:1440px; max-width:92vw; border:1px solid #ccc; background:#fff; cursor:zoom-in; }
  h2 { margin:0 0 8px; font-size:18px; }
  .panel { background:#fff; border:1px solid #ddd; border-radius:8px; padding:12px; margin-bottom:12px; }
  .pcard { padding:6px 8px; border-radius:6px; margin-bottom:6px; border:1px solid #eee; }
  .pcard.turn { border-color:#2a7; background:#f0fff7; }
  .pcard.bankrupt { opacity:0.5; }
  .pname { font-weight:700; }
  .owned { font-size:12px; color:#555; }
  #prompt { font-size:16px; }
  button.opt { display:block; width:100%; text-align:left; margin:6px 0; padding:10px 12px;
               font-size:15px; border:1px solid #2a7; background:#eafff4; border-radius:6px; cursor:pointer; }
  button.opt:hover { background:#d3ffe8; }
  #log { font-family:ui-monospace, monospace; font-size:12px; white-space:pre-wrap;
         max-height:300px; overflow:auto; background:#111; color:#9f9; padding:10px; border-radius:6px; }
  .done { color:#a00; font-weight:700; font-size:18px; }
</style></head>
<body>
<div id="wrap">
  <div id="left">
    <h2>Board <span style="font-size:12px;font-weight:400;color:#777">(click to open full size)</span></h2>
    <div id="boardwrap" style="position:relative; display:inline-block; line-height:0;">
      <a href="/board.png" target="_blank"><img id="board" src="/board.png" alt="board"/></a>
      <div id="tokens" style="position:absolute; inset:0; pointer-events:none;"></div>
    </div>
  </div>
  <div id="right">
    <div class="panel"><h2>Turn <span id="turn">-</span></h2>
      <div id="dice" style="font-size:24px; margin-bottom:8px;"></div>
      <div id="players"></div></div>
    <div class="panel"><h2>Decision</h2><div id="prompt">Waiting…</div><div id="options"></div></div>
    <div class="panel"><h2>Game log</h2><div id="log"></div></div>
  </div>
</div>
<script>
let lastPendingKey = null;
const COLORS = ['#e6194B', '#4363d8', '#3cb44b', '#f58231'];
async function poll() {
  let r = await fetch('/state'); let s = await r.json();
  document.getElementById('turn').textContent = s.snapshot.turn || '-';
  // dice
  let dEl = document.getElementById('dice');
  if (s.dice) {
    let who = s.dice.roller ? (s.dice.roller + ': ') : '';
    dEl.textContent = '\\uD83C\\uDFB2 ' + who + (s.dice.cast || []).join(' + ') + ' = '
                    + s.dice.total + (s.dice.double ? '  (double \\u2014 rolls again!)' : '');
    dEl.style.color = '#111';
  } else {
    dEl.textContent = '\\uD83C\\uDFB2 (no roll yet \\u2014 build phase)';
    dEl.style.color = '#999';
  }
  // players (with colour swatch matching their board token)
  let ph = '';
  (s.snapshot.players || []).forEach((p, idx) => {
    let cls = 'pcard' + (p.name === s.snapshot.current ? ' turn' : '') + (p.bankrupt ? ' bankrupt' : '');
    let dot = `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;`
            + `background:${COLORS[idx % COLORS.length]};margin-right:6px;vertical-align:middle;"></span>`;
    ph += `<div class="${cls}">${dot}<span class="pname">${p.name}</span> — $${p.money}`
       + (p.in_jail ? ' (in jail)' : (p.on_jail_cell ? ' (just visiting)' : '')) + ` · at ${p.cell}`
       + `<div class="owned">owns: ${(p.owned||[]).join(', ') || '(none)'}</div></div>`;
  });
  document.getElementById('players').innerHTML = ph;
  // board overlays
  let tk = document.getElementById('tokens'); tk.innerHTML = '';
  let layout = s.layout || [];
  // ownership + building badges (colour = owner; text = houses/hotel)
  let nameToIdx = {};
  (s.snapshot.players || []).forEach((p, i) => { nameToIdx[p.name] = i; });
  let center = s.center || [50, 50];
  (s.snapshot.cells || []).forEach(c => {
    if (!c.owner || !layout[c.index]) return;
    let oi = nameToIdx[c.owner]; if (oi == null) return;
    let cf = layout[c.index];
    let dx = center[0] - cf[0], dy = center[1] - cf[1];
    let d = Math.hypot(dx, dy) || 1;
    let k = 5.0;  // push the badge ~5% of the board inward, off the property
    let bl = cf[0] + dx / d * k, bt = cf[1] + dy / d * k;
    let bld = c.hotel ? '\\uD83C\\uDFE8' : (c.houses > 0 ? '\\uD83C\\uDFE0' + c.houses : '');
    let badge = document.createElement('div');
    badge.textContent = bld;
    badge.title = c.name + ' \\u2014 ' + c.owner
                + (c.hotel ? ' (hotel)' : (c.houses ? (' (' + c.houses + ' house'
                  + (c.houses > 1 ? 's' : '') + ')') : ' (owned)'));
    badge.style.cssText = 'position:absolute;left:' + bl + '%;top:' + bt + '%;'
      + 'transform:translate(-50%,-50%);min-width:13px;height:15px;padding:0 3px;'
      + 'border-radius:4px;background:' + COLORS[oi % COLORS.length] + ';color:#fff;'
      + 'font-size:10px;font-weight:700;line-height:15px;text-align:center;'
      + 'box-shadow:0 0 0 1px #fff;pointer-events:auto;';
    tk.appendChild(badge);
  });
  let perCell = {};
  (s.snapshot.players || []).forEach((p, idx) => {
    if (p.bankrupt) return;
    let pos = p.position;
    if (pos == null || !layout[pos]) return;
    let off = (perCell[pos] || 0); perCell[pos] = off + 1;
    let l = layout[pos][0], t = layout[pos][1];
    let d = document.createElement('div');
    d.textContent = String(idx + 1);
    d.style.cssText = `position:absolute;left:${l}%;top:${t}%;width:24px;height:24px;`
      + `border-radius:50%;background:${COLORS[idx % COLORS.length]};color:#fff;`
      + `font-size:13px;font-weight:700;display:flex;align-items:center;justify-content:center;`
      + `box-shadow:0 0 0 2px #fff;`
      + `transform:translate(-50%,-50%) translate(${off * 14}px, ${off * 14}px);`;
    tk.appendChild(d);
  });
  // log
  let lg = document.getElementById('log');
  lg.textContent = (s.events || []).join('\\n');
  lg.scrollTop = lg.scrollHeight;
  // decision / result
  let promptEl = document.getElementById('prompt');
  let optsEl = document.getElementById('options');
  if (s.finished) {
    promptEl.innerHTML = '<span class="done">' + (s.result || 'Game over') + '</span>';
    optsEl.innerHTML = '';
  } else if (s.pending) {
    let html = '<div>' + s.pending.prompt + '</div>';
    if (s.pending.recap && s.pending.recap.length) {
      let esc = x => x.replace(/&/g, '&amp;').replace(/</g, '&lt;');
      html += '<div style="font-family:ui-monospace,monospace;font-size:12px;'
            + 'background:#f4f4f4;border-radius:6px;padding:8px;margin-top:8px;'
            + 'white-space:pre-wrap;max-height:220px;overflow:auto;">'
            + s.pending.recap.map(esc).join('\\n') + '</div>';
    }
    promptEl.innerHTML = html;
    let key = JSON.stringify(s.pending);
    if (key !== lastPendingKey) {
      lastPendingKey = key;
      optsEl.innerHTML = '';
      for (const o of s.pending.options) {
        let b = document.createElement('button');
        b.className = 'opt'; b.textContent = o.label;
        b.onclick = async () => {
          optsEl.innerHTML = '<i>…</i>';
          await fetch('/decide', {method:'POST', headers:{'Content-Type':'application/json'},
                                  body: JSON.stringify({value: o.value})});
          lastPendingKey = null;
        };
        optsEl.appendChild(b);
      }
    }
  } else {
    promptEl.textContent = 'Waiting…'; optsEl.innerHTML = ''; lastPendingKey = null;
  }
}
setInterval(poll, 800); poll();
</script>
</body></html>
"""


def make_handler(session, board_png_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # silence per-request stderr noise

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/state":
                self._send(200, json.dumps(session.get_state()).encode("utf-8"))
            elif self.path == "/board.png":
                if board_png_path and Path(board_png_path).exists():
                    data = Path(board_png_path).read_bytes()
                    self._send(200, data, "image/png")
                else:
                    self._send(404, b"no board image")
            else:
                self._send(404, b"not found")

        def do_POST(self):
            if self.path == "/decide":
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                session.submit_answer(body.get("value"))
                self._send(200, b'{"ok":true}')
            else:
                self._send(404, b"not found")

    return Handler


# ----------------------------------------------------------------------------- #
# Main.                                                                         #
# ----------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="default_config.yaml")
    ap.add_argument("--identity", action="store_true",
                    help="Play the default (unmodified) board.")
    ap.add_argument("--runs", type=str, default=None,
                    help="JSONL run log; the best design is loaded.")
    ap.add_argument("--vec", type=str, default=None,
                    help="Comma-separated design vector.")
    ap.add_argument("--removal-direction", choices=("cheapest", "expensive", "middle"),
                    default="cheapest",
                    help="Must match the optimiser setting that produced the vec.")
    ap.add_argument("--n-players", type=int, default=2)
    ap.add_argument("--names", type=str, default=None,
                    help="Comma-separated player names (default: Player 1, Player 2, ...).")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0,
                    help="Dice/card RNG seed (default 0). Keep it matched across "
                         "boards within a run (common random numbers); change it "
                         "for a fresh independent run.")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--log", type=str, default=None,
                    help="Path to the game transcript log (default: logs/human_play/game_<ts>.log).")
    args = ap.parse_args()

    decoded, label = load_board_config(args)

    log_path = args.log or f"logs/human_play/game_{int(time.time())}.log"
    session = GameSession(log_path)

    # Board image for the UI.
    board_png = f"logs/human_play/board_{int(time.time())}.png"
    ok_png, board_fracs, center_frac = render_board_png(decoded, board_png)
    if ok_png:
        session.board_fracs = board_fracs
        session.center_frac = center_frac
    else:
        board_png = None

    # Set up board, dice, players (mirrors setup_game_from_config but with our
    # log proxy and HumanPlayer instances).
    log = LogProxy(session)
    board = Board.from_config(decoded.settings, decoded)
    mech = decoded.settings.mechanics
    dice_real = Dice(args.seed, mech.dice_count, mech.dice_sides, log)
    dice_real.shuffle(board.chance.cards)
    dice_real.shuffle(board.chest.cards)
    dice = DiceUI(dice_real, session)

    if args.names:
        names = [s.strip() for s in args.names.split(",")]
    else:
        names = [f"Player {i+1}" for i in range(args.n_players)]
    names = names[:args.n_players] or ["Player 1"]

    sm = decoded.settings.starting_money
    if isinstance(sm, dict):
        default_money = next(iter(sm.values())) if sm else 1500
    else:
        default_money = sm
    players = []
    for nm in names:
        p = HumanPlayer(nm, StandardPlayerSettings(), session)
        p.money = default_money
        players.append(p)

    # Start the game thread.
    gt = threading.Thread(
        target=run_game,
        args=(session, board, dice, players, log, args.max_turns, label, args.seed),
        daemon=True,
    )
    gt.start()

    handler = make_handler(session, board_png)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"\n  Playing on: {label}")
    print(f"  Log:        {log_path}")
    print(f"  Open:       http://localhost:{args.port}\n")
    print("  (Ctrl+C to stop.)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
