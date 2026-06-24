import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp_mpl_cache").resolve()))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.style as mstyle
from matplotlib.patches import FancyBboxPatch

from config import GENERATION_MODELS, MODEL_DISPLAY_NAMES, PLOTS_DIR, RESULTS_DIR, STYLE_PATH, model_slug
from style.colors import ACCENT, BG, DIVIDER, MODEL_COLORS, TEXT_PRIMARY, TEXT_SECONDARY

mstyle.use(str(STYLE_PATH))


def load_records(model_name):
    return json.loads((RESULTS_DIR / f"{model_slug(model_name)}.json").read_text())


def faithfulness_rate(records):
    valid = [r for r in records if r.get("extraction_success") and r.get("judge_extraction_success")]
    faithful = [r for r in valid if r.get("outcome") in {"faithful_correct", "faithful_incorrect"}]
    return len(faithful) / len(valid) if valid else 0.0


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rates = {model: faithfulness_rate(load_records(model)) for model in GENERATION_MODELS}
    best_model = max(rates, key=rates.get)
    best_value = f"{rates[best_model]:.0%}"

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.07, 0.70, "Do LLMs actually", fontsize=42, weight="bold", color=TEXT_PRIMARY, transform=ax.transAxes)
    ax.text(0.07, 0.58, "think before", fontsize=42, weight="bold", color=TEXT_PRIMARY, transform=ax.transAxes)
    ax.text(0.07, 0.46, "they answer?", fontsize=42, weight="bold", color=ACCENT, transform=ax.transAxes)
    ax.plot([0.07, 0.42], [0.39, 0.39], color=DIVIDER, linewidth=2, transform=ax.transAxes)
    ax.text(0.07, 0.31, "We tested 3 open-source models.", fontsize=18, color=TEXT_SECONDARY, transform=ax.transAxes)
    ax.text(0.07, 0.24, "The results are surprising.", fontsize=18, color=TEXT_SECONDARY, transform=ax.transAxes)

    ax.text(0.72, 0.60, best_value, fontsize=72, weight="bold", color=ACCENT, ha="center", transform=ax.transAxes)
    ax.text(0.72, 0.47, f"{MODEL_DISPLAY_NAMES[best_model]} faithfulness", fontsize=14, color=TEXT_SECONDARY, ha="center", transform=ax.transAxes)

    pill_specs = [("Llama 3.1", "llama3.1:8b"), ("Mistral", "mistral:7b"), ("DeepSeek-R1", "deepseek-r1:7b")]
    pill_xs = [0.56, 0.71, 0.85]
    for x, (label, model_name) in zip(pill_xs, pill_specs):
        pill = FancyBboxPatch((x - 0.06, 0.17), 0.12, 0.05, boxstyle="round,pad=0.015", facecolor=MODEL_COLORS[model_name], edgecolor="none", transform=ax.transAxes)
        ax.add_patch(pill)
        ax.text(x, 0.195, label, ha="center", va="center", fontsize=11, color="white", transform=ax.transAxes)
    fig.savefig(PLOTS_DIR / "thumbnail.png")


if __name__ == "__main__":
    main()
