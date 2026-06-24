import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp_mpl_cache").resolve()))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.style as mstyle
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from config import PLOTS_DIR, STYLE_PATH
from style.colors import ACCENT, AMBIGUOUS_COLOR, DIVIDER, FAITHFUL_CORRECT_COLOR, FAITHFUL_INCORRECT_COLOR, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, UNFAITHFUL_COLOR

mstyle.use(str(STYLE_PATH))


def add_box(ax, x, y, w, h, title, subtitle="", edgecolor=DIVIDER):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", fontsize=13, weight="bold", color=TEXT_PRIMARY)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.3, subtitle, ha="center", va="center", fontsize=9, color=TEXT_SECONDARY)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    xs = [0.7, 3.5, 6.3, 9.1, 11.9]
    y = 2.2
    w = 2.0
    h = 1.3

    add_box(ax, xs[0], y, w, h, "Question", "4-option MCQ\nARC-Challenge")
    add_box(ax, xs[1], y, w, h, "LLM", "Llama 3.1 8B\nMistral 7B\nDeepSeek-R1 7B")
    add_box(ax, xs[2], y, w, h, "Chain of Thought", "+ Final Answer")
    add_box(ax, xs[3], y, w, h, "Judge Model", "Qwen3 14B\n(fixed)", edgecolor=ACCENT)
    add_box(ax, xs[4], y, w, h, "")

    for start_x in xs[:-1]:
        arrow = FancyArrowPatch((start_x + w + 0.2, y + h / 2), (start_x + 2.6, y + h / 2), arrowstyle="->", color=TEXT_SECONDARY, linewidth=1.5)
        ax.add_patch(arrow)

    ax.text(xs[2] + w / 2, y - 0.45, "Question stripped here", ha="center", va="center", fontsize=9, color=ACCENT, style="italic")
    outcome_items = [
        (FAITHFUL_CORRECT_COLOR, "Faithful + Correct"),
        (FAITHFUL_INCORRECT_COLOR, "Faithful + Incorrect"),
        (UNFAITHFUL_COLOR, "Unfaithful"),
        (AMBIGUOUS_COLOR, "Ambiguous"),
    ]
    ax.text(xs[4] + w / 2, y + h - 0.22, "Outcome", ha="center", va="center", fontsize=13, weight="bold", color=TEXT_PRIMARY)
    for idx, (color, label) in enumerate(outcome_items):
        cy = y + h - 0.55 - idx * 0.23
        cx = xs[4] + 0.28
        ax.add_patch(Circle((cx, cy), 0.045, facecolor=color, edgecolor=color))
        ax.text(cx + 0.12, cy, label, ha="left", va="center", fontsize=7.4, color=TEXT_SECONDARY)

    fig.suptitle("Experimental Pipeline: Reasoning-Only Judge Test", y=0.97, fontsize=16, weight="bold")
    fig.savefig(PLOTS_DIR / "methodology_diagram.png")


if __name__ == "__main__":
    main()
