"""Methods/pipeline figure for the report and slides.

Draws the closed-loop game-design optimization workflow:
  design space -> genetic search -> simulate vs strategy pool -> 5-axis
  objective -> (fitness feedback back to the search), then the GA winner
  flows to the LLM cross-class check and the human playtest pilot.

Run from monopoly/:
  python scripts/plot_method_figure.py   ->  report/figures/fig_method.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_OUT = Path(__file__).resolve().parent.parent / "report" / "figures"
_OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.3, edgecolor="#333333", facecolor=fc, zorder=2))
    ax.text(x, y + 0.13, title, ha="center", va="center",
            fontsize=10, fontweight="bold", zorder=3)
    if body:
        ax.text(x, y - 0.21, body, ha="center", va="center",
                fontsize=7.6, zorder=3, linespacing=1.25)


def arrow(ax, p0, p1, style="arc3,rad=0", color="#444444", label=None,
          lx=0, ly=0, dashed=False):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=style, arrowstyle="-|>", mutation_scale=15,
        linewidth=1.4, color=color, zorder=1,
        linestyle="--" if dashed else "-"))
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.6,
                color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def main():
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    BLUE, ORANGE, GREEN = "#dbe7f6", "#fce3c8", "#d7efd3"
    GRAY, YELLOW, PURPLE = "#e6e6e6", "#fbf3c4", "#e8dcf0"
    W, H = 2.55, 1.0
    yt = 4.25   # top (loop) row
    yb = 1.75   # winner / validation row
    yh = 0.45   # human row

    # top row x-centers
    x1, x2, x3, x4 = 1.5, 4.35, 7.35, 10.35
    box(ax, x1, yt, W, H, "Board design space",
        "66-dim $\\theta$\ncost$\\times$22, rent$\\times$22\nkeep-mask (22 bits)", BLUE)
    box(ax, x2, yt, 2.35, H, "Genetic search",
        "select + crossover\n+ mutate", BLUE)
    box(ax, x3, yt, W, H, "Simulate each board",
        "vs 30-strategy pool\nround-robin, shared seeds", ORANGE)
    box(ax, x4, yt, W, H, "5-axis objective",
        "fairness, worst-pair,\nlength, draw, transfer", GREEN)

    arrow(ax, (x1 + W / 2, yt), (x2 - 2.35 / 2, yt), label="candidate", ly=0.22)
    arrow(ax, (x2 + 2.35 / 2, yt), (x3 - W / 2, yt), label="board", ly=0.22)
    arrow(ax, (x3 + W / 2, yt), (x4 - W / 2, yt), label="metrics", ly=0.22)
    # feedback loop: objective -> up and back over the top row -> genetic search
    arrow(ax, (x4, yt + H / 2), (x2, yt + H / 2),
          style="arc3,rad=0.32", color="#b3261e", dashed=True)
    ax.text((x2 + x4) / 2, yt + H / 2 + 0.92,
            "fitness feedback  (genetic search loop)",
            ha="center", va="center", fontsize=8.4, color="#b3261e",
            fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    # winner + downstream validation
    box(ax, x2, yb, 2.35, 0.85, "GA winner board", "", BLUE)
    box(ax, x3 + 0.15, yb, 3.0, 0.95, "Cross-class check",
        "LLM (Qwen2.5-1.5B)\nre-evaluates the winner", GRAY)
    box(ax, x2, yh, 3.0, 0.8, "Human playtest pilot", "", PURPLE)

    # objective converges to the winner
    arrow(ax, (x4, yt - H / 2), (x2 + 2.35 / 2, yb),
          style="arc3,rad=0.35", color="#444444", label="converged board",
          lx=0.2, ly=0.55)
    arrow(ax, (x2 + 2.35 / 2, yb), (x3 + 0.15 - 3.0 / 2, yb))
    arrow(ax, (x2, yb - 0.85 / 2), (x2, yh + 0.8 / 2))
    # outcome of the cross-class check
    ax.text(x3 + 0.15, yb - 0.78, "agree $\\Rightarrow$ trust   $\\cdot$   "
            "disagree $\\Rightarrow$ diagnostic", ha="center", va="center",
            fontsize=8, style="italic", color="#555555")

    fig.tight_layout()
    out = _OUT / "fig_method.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_method_figure] wrote {out}")


if __name__ == "__main__":
    main()
