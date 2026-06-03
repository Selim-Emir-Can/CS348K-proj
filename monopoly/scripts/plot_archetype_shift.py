"""Slope chart: win-rate shift of cash-conserving vs aggressive archetypes
between the default board and the GA-2p winner.

Numbers are the per-archetype mean win-rate-vs-field values reported in the
"What the GA fixed" table (slides_final.tex / report). Two groups:

  - Cash-conserving (CashHoarder, RiskAverse, Passive): middling -> top-tier.
  - Aggressive      (AggressiveBuilder, Trader, Bully): top-tier -> middling.

The bold lines are the group averages; the faint lines are the individual
archetypes that make up each group, for context.

Run from monopoly/:
  python scripts/plot_archetype_shift.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_OUT = _ROOT / "report" / "figures"
_OUT.mkdir(parents=True, exist_ok=True)

# (archetype, WR_default, WR_ga2p) straight from the table.
CASH = [
    ("CashHoarder", 0.46, 0.63),
    ("RiskAverse",  0.50, 0.60),
    ("Passive",     0.48, 0.54),
]
AGGR = [
    ("AggressiveBuilder", 0.73, 0.66),
    ("Trader",            0.66, 0.60),
    ("Bully",             0.63, 0.62),
]
# RailroadKing is a board-robust dud: near-zero win-rate under both boards.
RAIL = ("RailroadKing", 0.01, 0.00)
# AggressiveBuilder is the board-robust top: best on default, still strong on GA.
ABUILD = ("AggressiveBuilder", 0.73, 0.66)

CASH_C = "#1b9e77"   # teal-green
AGGR_C = "#d95f02"   # warm orange
RAIL_C = "#7570b3"   # muted purple
ABUILD_C = "#a6261d"  # dark red


def _avg(group, idx):
    return sum(g[idx] for g in group) / len(group)


def main():
    x = [0, 1]
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    # Faint per-archetype lines for context.
    for name, d, g in CASH:
        ax.plot(x, [d, g], color=CASH_C, alpha=0.28, lw=1.4, zorder=1)
    for name, d, g in AGGR:
        ax.plot(x, [d, g], color=AGGR_C, alpha=0.28, lw=1.4, zorder=1)

    # Bold group-average lines.
    cash_avg = [_avg(CASH, 1), _avg(CASH, 2)]
    aggr_avg = [_avg(AGGR, 1), _avg(AGGR, 2)]
    ax.plot(x, cash_avg, color=CASH_C, lw=3.2, marker="o", ms=9,
            zorder=3, label="Cash-conserving (avg)")
    ax.plot(x, aggr_avg, color=AGGR_C, lw=3.2, marker="o", ms=9,
            zorder=3, label="Aggressive (avg)")

    # RailroadKing: single board-robust line near zero.
    rail = [RAIL[1], RAIL[2]]
    ax.plot(x, rail, color=RAIL_C, lw=3.2, marker="o", ms=9,
            zorder=3, label="RailroadKing")

    # AggressiveBuilder: single board-robust line near the top.
    abuild = [ABUILD[1], ABUILD[2]]
    ax.plot(x, abuild, color=ABUILD_C, lw=3.2, marker="o", ms=9,
            zorder=3, label="AggressiveBuilder")

    # Endpoint value labels on the averages.
    ax.text(-0.04, cash_avg[0], f"{cash_avg[0]:.2f}", ha="right", va="center",
            fontsize=11, fontweight="bold", color=CASH_C)
    ax.text(1.04, cash_avg[1], f"{cash_avg[1]:.2f}", ha="left", va="center",
            fontsize=11, fontweight="bold", color=CASH_C)
    ax.text(-0.04, aggr_avg[0], f"{aggr_avg[0]:.2f}", ha="right", va="center",
            fontsize=11, fontweight="bold", color=AGGR_C)
    ax.text(1.04, aggr_avg[1] - 0.012, f"{aggr_avg[1]:.2f}", ha="left",
            va="center", fontsize=11, fontweight="bold", color=AGGR_C)
    ax.text(-0.04, abuild[0], f"{abuild[0]:.2f}", ha="right", va="center",
            fontsize=11, fontweight="bold", color=ABUILD_C)
    ax.text(1.04, abuild[1] + 0.018, f"{abuild[1]:.2f}", ha="left",
            va="center", fontsize=11, fontweight="bold", color=ABUILD_C)
    ax.text(-0.04, rail[0], f"{rail[0]:.2f}", ha="right", va="center",
            fontsize=11, fontweight="bold", color=RAIL_C)
    ax.text(1.04, rail[1], f"{rail[1]:.2f}", ha="left", va="center",
            fontsize=11, fontweight="bold", color=RAIL_C)

    # 0.50 reference (even match vs field).
    ax.axhline(0.50, ls="--", lw=1, color="#999999", zorder=0)
    ax.text(0.5, 0.505, "even vs field (0.50)", ha="center", va="bottom",
            fontsize=8.5, color="#777777")

    ax.set_xticks(x)
    ax.set_xticklabels(["Default board", "GA-2p winner"], fontsize=12)
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(-0.05, 0.80)
    ax.set_ylabel("mean win-rate vs field", fontsize=12)
    ax.set_title("The GA board reshuffles who wins:\n"
                 "cash-conserving play rises, aggression no longer dominates",
                 fontsize=12.5)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.30), fontsize=10,
              frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = _OUT / "fig_archetype_shift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_archetype_shift] wrote {out}")
    print(f"  cash-conserving avg: {cash_avg[0]:.3f} -> {cash_avg[1]:.3f}")
    print(f"  aggressive      avg: {aggr_avg[0]:.3f} -> {aggr_avg[1]:.3f}")


if __name__ == "__main__":
    main()
