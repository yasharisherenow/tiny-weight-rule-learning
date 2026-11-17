"""Plot the first matched-pair run using its saved result counts."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.run_dir / "protocol.json").read_text(encoding="utf-8"))
    lr = protocol["training"]["learning_rate"]
    shortcut = max(report["shortcut_baselines"].values(),
                   key=lambda s: (s["disagreement"]["both_correct"], s["agreement"]["both_correct"]))
    conditions = (
        ("Frozen, rules withheld", report["frozen_hidden"], "#788694"),
        ("Best tested shortcut", shortcut, "#788694"),
        ("Frozen, rules shown", report["frozen_visible"], "#a65d31"),
        ("LoRA, rules withheld · familiar wording", report["trained_hidden_seen_wording"], "#397480"),
        ("LoRA, rules withheld · new wording", report["trained_hidden_novel_wording"], "#225461"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.7), sharey=True)
    fig.subplots_adjust(left=0.31, right=0.96, top=0.76, bottom=0.20, wspace=0.27)
    fig.text(0.03, 0.92, "Matched Qwen2.5 0.5B LoRA", fontsize=18, weight="bold")
    fig.text(0.03, 0.045,
             f"Learning rate {lr:g}. Both answers must be correct on the same parcel. One rulebook pair.",
             fontsize=9, color="#3f4e5b")
    labels = [name for name, _, _ in conditions]
    for ax, (split, title) in zip(axes, (("disagreement", "Correct answers differ"),
                                       ("agreement", "Correct answers agree")), strict=True):
        values = [counts[split] for _, counts, _ in conditions]
        rates = [row["both_correct"] / row["n"] for row in values]
        bars = ax.barh(range(len(labels)), rates, color=[color for _, _, color in conditions], height=0.61)
        ax.bar_label(bars, [f"{r['both_correct']}/{r['n']}" for r in values], padding=4, fontsize=9)
        ax.set_xlim(0, 1.19)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dfe5e9", linewidth=0.6)
        ax.tick_params(length=0)
        ax.set_yticks(range(len(labels)), labels)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].invert_yaxis()
    for label in axes[1].get_yticklabels():
        label.set_visible(False)
    fig.savefig(args.run_dir / "matched-pair.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
