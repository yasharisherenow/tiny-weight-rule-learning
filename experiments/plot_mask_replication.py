"""Plot paired accuracy near the transition for separate coordinate masks."""

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
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11, 4.4))
    fig.subplots_adjust(left=.085, right=.97, top=.72, bottom=.24, wspace=.18)
    fig.text(.045, .91, "Does the jump repeat with new coordinate masks?",
             fontsize=18, weight="bold", color="#162b39")
    fig.text(.045, .85, "Qwen2.5 0.5B · 128 disagreement parcels per rulebook pair · rules withheld",
             fontsize=10, color="#425565")
    colors = ("#267884", "#654894")
    ticks = (1000, 2000, 4000, 8000, 10000)
    for i, (ax, (seed, draw)) in enumerate(zip(axes, report["draws"].items(), strict=True)):
        for j, (mask, counts) in enumerate(draw["fresh_masks"].items()):
            budgets = sorted(int(k) for k in counts)
            rate = [counts[str(k)] / draw["test_count"] for k in budgets]
            ax.plot(budgets, rate, color=colors[j], marker=("o", "s")[j],
                    markerfacecolor=(colors[0], "white")[j], linewidth=1.7, label=f"Fresh mask {j + 1}")
        anchors = draw["original_mask"]
        ax.scatter([int(k) for k in anchors], [v / draw["test_count"] for v in anchors.values()],
                   marker="D", facecolors="#536475", label="Original mask", zorder=3, s=27)
        ax.axhline(draw["shortcut"] / draw["test_count"], color="#a45a31", linestyle="--",
                   linewidth=1.3, label="Best tested shortcut")
        ax.set_title(f"Rulebook pair {i + 1}", loc="left", weight="bold", pad=9)
        ax.set_xscale("log")
        ax.set_xlim(820, 11900)
        ax.set_ylim(-.035, 1.06)
        ax.set_xticks(ticks, [f"{k // 1000}k" for k in ticks], fontsize=8.5)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(length=0)
        ax.tick_params(which="minor", bottom=False)
        ax.grid(axis="y", color="#dfe5e9", linewidth=.65)
        ax.set_axisbelow(True)
        ax.set_xlabel("Trainable adapter scalars")
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("Both copies correct")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="lower center", bbox_to_anchor=(.54, .07),
               frameon=False, fontsize=8.5)
    fig.text(.045, .028, "All three masks use the same frozen A matrices and training schedule.",
             fontsize=8.5, color="#425565")
    fig.savefig(args.run_dir / "mask-replication.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
