"""Parse a human web-UI game log into per-game metrics for the report.

The web UI (scripts/play_web.py) writes a plain-text transcript. This script
reconstructs the same objective metrics the automated experiments record
(see optimizer/objectives.py + optimizer/simulate.py) from that transcript,
so a human playtest can be compared against the agent predictions.

Metric definitions (kept consistent with the experiments):
  - rounds          : full turn cycles. Doubles recurse *within* a turn, so a
                      "go again" does NOT add a round. rounds = fresh turns of
                      the first player (= its turn-blocks minus its go-agains).
  - transfer_total  : player->player cash flow ($), matching
                      simulate._track_interplayer_transfers: rent + card pays
                      to a player + the bankruptcy cash cascade.
  - transfer_rate   : transfer_total / rounds  (objective target: $100/round).
  - truncated       : True if nobody was bankrupted (a turn-limit / draw game).

Usage:
  python scripts/human_game_stats.py --log <game.log> --out-dir <dir> \
      [--game-id game1] [--board ga_2p_winner]

Writes <out-dir>/<game-id>_stats.json and <out-dir>/<game-id>_stats.md.
Display/analysis only; does not touch the simulation or any experiment files.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_log(text: str) -> dict:
    lines = text.splitlines()

    # --- header: "= HUMAN GAME on <board> (<n> players) =" -------------------
    board, n_players = None, None
    m = re.search(r"HUMAN GAME on (\S+) \((\d+) players?\)", text)
    if m:
        board, n_players = m.group(1), int(m.group(2))
    # seed appears as "[seed=N]" for games logged after that header change;
    # older transcripts omit it (those were all run at the default seed 0).
    ms = re.search(r"\[seed=(\d+)\]", text)
    seed = int(ms.group(1)) if ms else None

    # --- turn blocks --------------------------------------------------------
    turn_re = re.compile(r"^=== (Player \d+) \(.*goes: ===")
    turn_owner = [m.group(1) for m in (turn_re.match(l) for l in lines) if m]
    blocks = Counter(turn_owner)
    players = sorted(blocks, key=lambda p: int(p.split()[1]))
    if n_players is None:
        n_players = len(players)

    # "go again" only printed when a double actually grants an extra turn
    go_again = Counter(re.findall(r"(Player \d+) rolled a double", text))

    # rounds = fresh turns of the first player (it leads every cycle)
    first = players[0] if players else None
    rounds = (blocks[first] - go_again.get(first, 0)) if first else 0

    # --- inter-player cash flow (matches simulate instrumentation) ----------
    rent = re.findall(r"(Player \d+) pays (Player \d+) rent \$(\d+(?:\.\d+)?)", text)
    # card pays straight to a player, e.g. Chairman "$50" (no 'rent', no tax)
    card = re.findall(r"(Player \d+) pays (Player \d+) \$(\d+(?:\.\d+)?)", text)
    bank = re.findall(
        r"(Player \d+) gave (Player \d+) all their remaining money \(\$(\d+(?:\.\d+)?)\)",
        text,
    )

    flow = Counter()
    for a, b, amt in rent:
        flow[(a, b)] += float(amt)
    for a, b, amt in card:
        flow[(a, b)] += float(amt)
    for a, b, amt in bank:
        flow[(a, b)] += float(amt)
    transfer_total = sum(flow.values())
    transfer_rate = (transfer_total / rounds) if rounds else 0.0

    # --- outcome ------------------------------------------------------------
    winner = None
    mw = re.search(r"GAME OVER: (Player \d+) wins", text)
    if mw:
        winner = mw.group(1)
    bankrupt = sorted(set(re.findall(r"(Player \d+) is bankrupt", text)))
    truncated = winner is None and not bankrupt
    terminated_by = "bankruptcy" if bankrupt else ("turn_limit" if winner is None else "last_standing")

    # --- descriptive engagement stats --------------------------------------
    bought = Counter(re.findall(r"(Player \d+) bought ", text))
    houses = Counter(re.findall(r"(Player \d+) built \d+\w+ house", text))
    hotels = Counter(re.findall(r"(Player \d+) built a hotel", text))
    jail_card = Counter(re.findall(r"(Player \d+) got GTJ", text))
    jail_cell = Counter(re.findall(r"(Player \d+) landed on Go To Jail", text))
    jail_cc = Counter(re.findall(r"(Player \d+) drew Community Chest card: 'Go to Jail", text))
    jail_total = Counter()
    for c in (jail_card, jail_cell, jail_cc):
        jail_total.update(c)

    return {
        "board": board,
        "seed": seed,
        "n_players": n_players,
        "winner": winner,
        "truncated": truncated,
        "terminated_by": terminated_by,
        "rounds": rounds,
        "total_turn_blocks": sum(blocks.values()),
        "doubles_go_again": dict(go_again),
        "transfer_total": round(transfer_total, 2),
        "transfer_rate_per_round": round(transfer_rate, 2),
        "transfer_breakdown": {f"{a}->{b}": round(v, 2) for (a, b), v in flow.items()},
        "properties_bought": dict(bought),
        "houses_built": dict(houses),
        "hotels_built": dict(hotels),
        "jail_entries": dict(jail_total),
        "players": players,
    }


def to_markdown(s: dict, game_id: str) -> str:
    def row(k, v):
        return f"| {k} | {v} |\n"

    md = f"# Human playtest stats — {game_id}\n\n"
    md += f"Board: **{s['board']}**  |  Players: **{s['n_players']}**  "
    md += f"|  Seed: **{s.get('seed')}**  "
    md += f"|  Winner: **{s['winner'] or 'none (draw)'}**\n\n"
    md += "## Objective metrics (report axes)\n\n"
    md += "| metric | value |\n|---|---|\n"
    md += row("rounds (length; target 60)", s["rounds"])
    md += row("transfer_total ($)", s["transfer_total"])
    md += row("transfer_rate ($/round; target 100)", s["transfer_rate_per_round"])
    md += row("decisive?", "yes" if not s["truncated"] else "no (draw / turn-limit)")
    md += row("terminated_by", s["terminated_by"])
    md += "\n### transfer breakdown\n\n| flow | $ |\n|---|---|\n"
    for k, v in s["transfer_breakdown"].items():
        md += row(k, v)
    md += "\n## Descriptive / engagement\n\n"
    md += "| stat | value |\n|---|---|\n"
    md += row("total turn-blocks (incl. doubles)", s["total_turn_blocks"])
    md += row("doubles (go-again)", s["doubles_go_again"])
    md += row("properties bought", s["properties_bought"])
    md += row("houses built (gross)", s["houses_built"])
    md += row("hotels built (gross)", s["hotels_built"])
    md += row("jail entries", s["jail_entries"])
    md += "\n_Generated from the game log by scripts/human_game_stats.py. "
    md += "Objective only — subjective ratings live in playtest_notes.md._\n"
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to the web-UI game log")
    ap.add_argument("--out-dir", required=True, help="folder to write stats into")
    ap.add_argument("--game-id", default="game1", help="basename for output files")
    ap.add_argument("--board", default=None, help="override board name in output")
    ap.add_argument("--seed", type=int, default=None,
                    help="override/record the RNG seed (use for older logs whose "
                         "header predates the [seed=N] tag).")
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8")
    stats = parse_log(text)
    if args.board:
        stats["board"] = args.board
    if args.seed is not None:
        stats["seed"] = args.seed

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.game_id}_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    (out / f"{args.game_id}_stats.md").write_text(
        to_markdown(stats, args.game_id), encoding="utf-8"
    )

    print(f"[human_game_stats] wrote {args.game_id}_stats.json / .md to {out}")
    print(f"  board={stats['board']} winner={stats['winner']} "
          f"rounds={stats['rounds']} transfer_rate={stats['transfer_rate_per_round']}")


if __name__ == "__main__":
    main()
