import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp_mpl_cache").resolve()))

import matplotlib.pyplot as plt
import matplotlib.style as mstyle
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config import (
    AMBIGUOUS,
    FAITHFUL_CORRECT,
    FAITHFUL_INCORRECT,
    GENERATION_MODELS,
    MODEL_DISPLAY_NAMES,
    PLOTS_DIR,
    RESULTS_DIR,
    STYLE_PATH,
    UNFAITHFUL,
    model_slug,
)
from style.colors import ACCENT, OUTCOME_COLORS, OUTCOME_LABELS, TEXT_SECONDARY

mstyle.use(str(STYLE_PATH))


def load_records(model_name):
    path = RESULTS_DIR / f"{model_slug(model_name)}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def compute_metrics(records, model_name):
    valid = [r for r in records if r.get("extraction_success") and r.get("judge_extraction_success")]
    correct = [r for r in valid if r.get("is_correct")]
    incorrect = [r for r in valid if not r.get("is_correct")]
    faithful = [r for r in valid if r.get("outcome") in {FAITHFUL_CORRECT, FAITHFUL_INCORRECT}]
    faithful_correct = [r for r in valid if r.get("outcome") == FAITHFUL_CORRECT]
    faithful_incorrect = [r for r in valid if r.get("outcome") == FAITHFUL_INCORRECT]
    unfaithful = [r for r in records if r.get("outcome") == UNFAITHFUL]
    ambiguous = [r for r in records if r.get("outcome") == AMBIGUOUS]
    return {
        "model": model_name,
        "display_name": MODEL_DISPLAY_NAMES[model_name],
        "total": len(records),
        "valid": len(valid),
        "faithful": len(faithful),
        "unfaithful": len(unfaithful),
        "ambiguous": len(ambiguous),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "faithfulness_rate": len(faithful) / len(valid) if valid else 0.0,
        "accuracy": len(correct) / len(valid) if valid else 0.0,
        "faith_given_correct": len(faithful_correct) / len(correct) if correct else 0.0,
        "faith_given_incorrect": len(faithful_incorrect) / len(incorrect) if incorrect else 0.0,
        "avg_cot_words": mean_or_zero([r["cot_word_count"] for r in valid]),
        "avg_cot_words_correct": mean_or_zero([r["cot_word_count"] for r in correct]),
        "avg_cot_words_incorrect": mean_or_zero([r["cot_word_count"] for r in incorrect]),
    }


def save_outcome_breakdown(records_by_model, metrics_df):
    outcomes = [FAITHFUL_CORRECT, FAITHFUL_INCORRECT, UNFAITHFUL, AMBIGUOUS]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.16, right=0.9)
    metrics_rows = metrics_df.set_index("model").loc[GENERATION_MODELS].reset_index()
    left = np.zeros(len(metrics_rows))
    for outcome in outcomes:
        values = []
        counts = []
        for model_name in metrics_rows["model"]:
            records = records_by_model[model_name]
            count = sum(1 for r in records if r.get("outcome") == outcome)
            counts.append(count)
            values.append(count / len(records) if records else 0.0)
        bars = ax.barh(
            metrics_rows["display_name"],
            values,
            left=left,
            color=OUTCOME_COLORS[outcome],
            label=OUTCOME_LABELS[outcome],
        )
        for bar, value, count in zip(bars, values, counts):
            if value > 0.07:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={count}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
        left += np.array(values)
    for idx, row in metrics_rows.iterrows():
        ax.text(
            1.01,
            idx,
            f"{row['faithfulness_rate']:.0%} faithful",
            ha="left",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=10,
        )
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Proportion of Responses")
    fig.suptitle("How faithful is the chain-of-thought?", y=0.95, fontsize=16, fontweight="bold")
    fig.text(0.16, 0.865, "ARC-Challenge, n=50 questions per model", color=TEXT_SECONDARY, fontsize=10)
    ax.spines["left"].set_visible(False)
    legend_handles = [Patch(facecolor=OUTCOME_COLORS[o], label=OUTCOME_LABELS[o]) for o in outcomes]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    fig.savefig(PLOTS_DIR / "outcome_breakdown.png")
    plt.close(fig)


def save_faithfulness_by_correctness(metrics_df):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.11, right=0.97)
    ordered = metrics_df.set_index("model").loc[GENERATION_MODELS].reset_index()
    x = np.arange(len(ordered))
    width = 0.35
    correct_bars = ax.bar(
        x - width / 2,
        ordered["faith_given_correct"],
        width=width,
        color=OUTCOME_COLORS[FAITHFUL_CORRECT],
        label="When model is correct",
    )
    incorrect_bars = ax.bar(
        x + width / 2,
        ordered["faith_given_incorrect"],
        width=width,
        color=OUTCOME_COLORS[FAITHFUL_INCORRECT],
        label="When model is incorrect",
    )
    ax.axhline(0.5, linestyle="--", color=TEXT_SECONDARY, linewidth=0.8, label="Chance")
    for bars in (correct_bars, incorrect_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, ordered["display_name"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Faithfulness Rate")
    fig.suptitle("Is CoT more faithful when the model gets it right?", y=0.95, fontsize=16, fontweight="bold")
    fig.text(
        0.11,
        0.865,
        "A faithfulness rate near 1.0 means reasoning consistently led to the stated answer",
        color=TEXT_SECONDARY,
        fontsize=10,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    fig.savefig(PLOTS_DIR / "faithfulness_by_correctness.png")
    plt.close(fig)


def save_cot_length_scatter(records_by_model):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.subplots_adjust(top=0.72, bottom=0.24, left=0.07, right=0.98, wspace=0.18)
    for index, model_name in enumerate(GENERATION_MODELS):
        ax = axes[index]
        records = [r for r in records_by_model[model_name] if r.get("judge_extraction_success")]
        x = [r["cot_word_count"] for r in records]
        y = [int(bool(r.get("is_correct"))) + rng.uniform(-0.04, 0.04) for r in records]
        colors = [OUTCOME_COLORS.get(r.get("outcome"), OUTCOME_COLORS[AMBIGUOUS]) for r in records]
        ax.scatter(x, y, c=colors, s=40, alpha=0.75, edgecolors="white", linewidths=0.5)
        if x:
            mean_x = float(np.mean(x))
            ax.axvline(mean_x, linestyle="--", color=ACCENT, linewidth=1.0)
            ax.text(mean_x + 2, 1.045, f"mean: {mean_x:.0f} words", color=TEXT_SECONDARY, fontsize=8)
            ax.set_xlim(0, max(x) + 20)
        ax.set_title(MODEL_DISPLAY_NAMES[model_name], pad=10)
        ax.set_xlabel("CoT word count")
        ax.set_yticks([0, 1], ["Incorrect", "Correct"])
        ax.set_ylim(-0.15, 1.15)
    fig.suptitle("Does longer reasoning lead to more correct answers?", y=0.96, fontsize=16, fontweight="bold")
    fig.text(0.07, 0.865, "Point color indicates faithfulness outcome", color=TEXT_SECONDARY, fontsize=10)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=OUTCOME_COLORS[key], label=label, markersize=7)
        for key, label in OUTCOME_LABELS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.08), ncol=4, frameon=False)
    fig.text(0.5, 0.02, "n=50 per model - treat as exploratory", ha="center", color=TEXT_SECONDARY, fontsize=9, style="italic")
    fig.savefig(PLOTS_DIR / "cot_length_scatter.png")
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    records_by_model = {model_name: load_records(model_name) for model_name in GENERATION_MODELS}
    metrics = [compute_metrics(records_by_model[model_name], model_name) for model_name in GENERATION_MODELS]
    metrics_df = pd.DataFrame(metrics).sort_values("faithfulness_rate", ascending=False)
    print(metrics_df.to_string(index=False))
    save_outcome_breakdown(records_by_model, metrics_df)
    save_faithfulness_by_correctness(metrics_df)
    save_cot_length_scatter(records_by_model)


if __name__ == "__main__":
    main()
