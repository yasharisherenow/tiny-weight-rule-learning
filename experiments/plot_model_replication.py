"""Plot matched parcel results for Qwen and SmolLM2."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.result_dir / "results.json").read_text(encoding="utf-8"))
    labels = ["2,000 scalars · 6 epochs", "2,000 scalars · 12 epochs",
              "2,000 scalars · 18 epochs", "8,000 scalars · up to 6 epochs",
              "Conventional LoRA · 6 epochs"]
    keys = ["2000-6", "2000-12", "2000-18", "8000-6", "conventional"]
    n = report["test_count"]

    fig, ax = plt.subplots(figsize=(8.9, 4.4))
    fig.subplots_adjust(left=.304, right=.962, top=.72, bottom=.21)
    fig.text(.04, .91, "Rule learning in a second model", fontsize=19, weight="bold", color="#162b39")
    fig.text(.04, .855, "Both copies correct on the same unseen parcel, with the rulebooks withheld",
             fontsize=10, color="#425565")
    arms = (
        ("Qwen2.5 0.5B", -.16, "#3a7184", "qwen"),
        ("SmolLM2 360M", .16, "#6e5696", "smol"),
    )
    for name, offset, color, model in arms:
        hits = [report["paired"][model][k] for k in keys]
        x = [h / n for h in hits]
        bars = ax.barh([i + offset for i in range(len(keys))], x,
                       height=.28, color=color, label=name)
        ax.bar_label(bars, [f"{h}/{n}" for h in hits], padding=3, fontsize=8)
    ax.axvline(report["shortcut"] / n, color="#a45a31", linestyle="--",
               linewidth=1.3, label="Best tested shortcut")
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_yticks(range(len(keys)), labels)
    ax.tick_params(length=0)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dfe5e9", linewidth=.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    handles, legend = ax.get_legend_handles_labels()
    fig.legend(handles, legend, ncols=3, loc="lower center", bbox_to_anchor=(.55, .073),
               frameon=False, fontsize=8.5)
    fig.text(.04, .03, "One rulebook pair · familiar record format · 8,000-scalar checkpoints selected on validation.",
             fontsize=8.5, color="#425565")
    fig.savefig(args.result_dir / "model-replication.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
