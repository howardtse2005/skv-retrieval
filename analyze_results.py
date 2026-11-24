"""Summarise benchmark results and create experiment figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

POSITION_LABELS = [
    "beginning",
    "25th_percentile",
    "50th_percentile",
    "75th_percentile",
    "end",
]
POSITION_TICKS = ["Beginning", "25th", "50th", "75th", "End"]
SIMILARITY_PALETTE = {"high": "#4c72b0", "low": "#dd8452"}
MODEL_GROUPS = {
    "1B": ("gemma3:1b", "llama3.2:1b", "qwen3:1.7b"),
    "3B": ("llama3.2:3b", "ministral-3:3b", "qwen2.5:3b"),
}


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("results_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(f"No results_*.jsonl files found in {results_dir}")
    frame = pd.DataFrame(rows)
    frame["position_label"] = pd.Categorical(
        frame["position_label"],
        categories=POSITION_LABELS,
        ordered=True,
    )
    return frame


def write_analysis(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("accuracy_by_position*.png"):
        path.unlink()
    summary = (frame.groupby(["model", "size", "similarity", "position_label"],
                             observed=True)["correct"]
               .agg(["mean", "count", "sum"])
               .reset_index()
               .rename(columns={"mean": "accuracy", "sum": "correct_trials"}))
    summary.to_csv(output_dir / "accuracy_summary.csv", index=False)
    errors = (frame.assign(is_error=frame["correct"].eq(0))
              .groupby(["model", "size", "similarity", "error_type"],
                       observed=True)["is_error"].mean()
              .reset_index(name="rate"))
    errors.to_csv(output_dir / "error_type_rates.csv", index=False)

    sns.set_theme(style="darkgrid")
    for size in sorted(frame["size"].dropna().unique()):
        size_frame = frame[frame["size"] == size]
        for group_name, group_models in MODEL_GROUPS.items():
            group_frame = size_frame[size_frame["model"].isin(group_models)]
            models = [model for model in group_models if model in set(group_frame["model"])]
            if not models:
                continue
            fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5),
                                     sharey=False, squeeze=False)
            axes = axes[0]
            for axis, model in zip(axes, models):
                subset = group_frame[group_frame["model"] == model]
                sns.lineplot(
                    data=subset, x="position_label", y="correct",
                    hue="similarity", estimator="mean", errorbar=None,
                    marker="o", palette=SIMILARITY_PALETTE, ax=axis,
                )
                axis.set_ylim(0.2, 1.01)
                axis.yaxis.set_major_formatter(PercentFormatter(1.0))
                axis.set_title(model)
                axis.set_xlabel("Target position")
                axis.set_ylabel("Accuracy (%)")
                axis.set_xticks(range(len(POSITION_LABELS)))
                axis.set_xticklabels(POSITION_TICKS)
                if axis is not axes[0] and axis.legend_:
                    axis.legend_.remove()
            handles, labels = axes[0].get_legend_handles_labels()
            if axes[0].legend_:
                axes[0].legend_.remove()
            fig.suptitle(f"Accuracy by target position ({size} key-value pairs, {group_name} models)")
            fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                       bbox_to_anchor=(0.5, 0.01))
            fig.tight_layout(rect=(0, 0.1, 1, 0.95))
            fig.savefig(
                output_dir / f"accuracy_by_position_{size}_{group_name.lower()}.png",
                dpi=180,
            )
            plt.close(fig)

    heatmap = (frame.groupby(["size", "similarity", "position_label"], observed=True)["correct"]
               .mean().reset_index()
               .pivot_table(index=["size", "similarity"], columns="position_label",
                            values="correct", observed=True))
    plt.figure(figsize=(9, 5))
    sns.heatmap(heatmap, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis")
    plt.title("Accuracy by size, similarity, and target position")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_heatmap.png", dpi=180)
    plt.close()

    # Keep the inferential result reproducible without making analysis fail if a
    # local statsmodels build lacks a mixed-effects implementation.
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        model = smf.glm("correct ~ C(position_label) * C(similarity) * C(size)",
                        data=frame, family=sm.families.Binomial()).fit()
        (output_dir / "glm_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    except (ImportError, ValueError, KeyError) as exc:
        (output_dir / "glm_summary.txt").write_text(
            f"GLM unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    frame = load_results(args.results_dir)
    write_analysis(frame, args.output_dir)
    print(f"Analysed {len(frame)} trials; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
