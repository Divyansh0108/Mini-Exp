import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp_mpl_cache").resolve()))
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.style as mstyle
from matplotlib.patches import FancyBboxPatch, Rectangle

from config import GENERATION_MODELS, MODEL_DISPLAY_NAMES, PLOTS_DIR, RESULTS_DIR, STYLE_PATH, model_slug
from style.colors import ACCENT, BG, DIVIDER, OUTCOME_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, UNFAITHFUL_COLOR, FAITHFUL_CORRECT_COLOR

mstyle.use(str(STYLE_PATH))


def load_records(model_name):
    return json.loads((RESULTS_DIR / f"{model_slug(model_name)}.json").read_text())


def metrics(records):
    valid = [r for r in records if r.get("extraction_success") and r.get("judge_extraction_success")]
    faithful = [r for r in valid if r.get("outcome") in {"faithful_correct", "faithful_incorrect"}]
    correct = [r for r in valid if r.get("is_correct")]
    return {
        "faithfulness_rate": len(faithful) / len(valid) if valid else 0.0,
        "accuracy": len(correct) / len(valid) if valid else 0.0,
        "outcomes": {
            "faithful_correct": sum(1 for r in records if r.get("outcome") == "faithful_correct"),
            "faithful_incorrect": sum(1 for r in records if r.get("outcome") == "faithful_incorrect"),
            "unfaithful": sum(1 for r in records if r.get("outcome") == "unfaithful"),
            "ambiguous": sum(1 for r in records if r.get("outcome") == "ambiguous"),
        },
    }


def headline_from(metrics_by_model):
    mistral = metrics_by_model["mistral:7b"]["faithfulness_rate"]
    deepseek = metrics_by_model["deepseek-r1:7b"]["faithfulness_rate"]
    gap = mistral - deepseek
    return f"{gap:.0%}", "faithfulness gap between Mistral 7B and DeepSeek-R1 7B"


def rank_color(rate, rates):
    if rate == max(rates):
        return FAITHFUL_CORRECT_COLOR
    if rate == min(rates):
        return UNFAITHFUL_COLOR
    return ACCENT


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    records_by_model = {model: load_records(model) for model in GENERATION_MODELS}
    metrics_by_model = {model: metrics(records) for model, records in records_by_model.items()}
    headline_value, headline_caption = headline_from(metrics_by_model)
    faith_rates = [metrics_by_model[m]["faithfulness_rate"] for m in GENERATION_MODELS]

    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.05, 0.92, "CoT Faithfulness in Open-Source LLMs", fontsize=20, weight="bold", color=TEXT_PRIMARY, transform=ax.transAxes)
    ax.text(0.05, 0.875, "Does the reasoning actually lead to the answer?", fontsize=11, color=TEXT_SECONDARY, transform=ax.transAxes)

    card = FancyBboxPatch((0.04, 0.14), 0.92, 0.68, boxstyle="round,pad=0.01", facecolor=BG, edgecolor=DIVIDER, linewidth=1.2, transform=ax.transAxes)
    ax.add_patch(card)
    ax.plot([0.04, 0.96], [0.74, 0.74], color=DIVIDER, linewidth=1, transform=ax.transAxes)
    ax.plot([0.04, 0.96], [0.28, 0.28], color=DIVIDER, linewidth=1, transform=ax.transAxes)
    for x in [0.24, 0.49, 0.72]:
        ax.plot([x, x], [0.28, 0.74], color=DIVIDER, linewidth=1, transform=ax.transAxes)

    ax.text(0.14, 0.61, headline_value, fontsize=36, weight="bold", color=ACCENT, ha="center", transform=ax.transAxes)
    ax.text(0.14, 0.52, "faithfulness gap", fontsize=10, color=TEXT_SECONDARY, ha="center", transform=ax.transAxes)
    ax.text(0.14, 0.44, "between Mistral 7B\nand DeepSeek-R1 7B", fontsize=9.5, color=TEXT_SECONDARY, ha="center", linespacing=1.35, transform=ax.transAxes)

    column_centers = [0.365, 0.605, 0.845]
    for center, model_name in zip(column_centers, GENERATION_MODELS):
        model_metrics = metrics_by_model[model_name]
        ax.text(center, 0.64, MODEL_DISPLAY_NAMES[model_name], ha="center", fontsize=13, weight="bold", color=TEXT_PRIMARY, transform=ax.transAxes)
        rate = model_metrics["faithfulness_rate"]
        ax.text(center, 0.53, f"{rate:.0%}", ha="center", fontsize=28, weight="bold", color=rank_color(rate, faith_rates), transform=ax.transAxes)
        ax.text(center, 0.47, "faithful", ha="center", fontsize=10, color=TEXT_SECONDARY, transform=ax.transAxes)
        ax.text(center, 0.39, f"{model_metrics['accuracy']:.0%} acc.", ha="center", fontsize=11, color=TEXT_SECONDARY, transform=ax.transAxes)
        bar_x = center - 0.085
        total = sum(model_metrics["outcomes"].values())
        left = bar_x
        for key in ["faithful_correct", "faithful_incorrect", "unfaithful", "ambiguous"]:
            width = 0.17 * (model_metrics["outcomes"][key] / total if total else 0.0)
            ax.add_patch(Rectangle((left, 0.325), width, 0.018, transform=ax.transAxes, facecolor=OUTCOME_COLORS[key], edgecolor="none"))
            left += width

    ax.text(0.05, 0.205, "Key finding:", fontsize=11, weight="bold", color=TEXT_PRIMARY, transform=ax.transAxes)
    ax.text(0.16, 0.205, "DeepSeek was highly accurate on valid rows, but its reasoning aligned with the final answer far less often than Mistral or Llama.", fontsize=10.2, color=TEXT_SECONDARY, transform=ax.transAxes)
    ax.add_patch(Rectangle((0.04, 0.08), 0.92, 0.05, transform=ax.transAxes, facecolor=SURFACE, edgecolor=DIVIDER, linewidth=0.8))
    ax.text(0.76, 0.095, "ARC-Challenge · n=50 · Ollama", fontsize=9, color=TEXT_SECONDARY, transform=ax.transAxes)
    fig.savefig(PLOTS_DIR / "results_card.png")


if __name__ == "__main__":
    main()
