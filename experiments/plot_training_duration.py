"""Plot how fixed-checkpoint held-out accuracy changes with training duration."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference-run", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    reference = json.loads((args.reference_run / "results.json").read_text(encoding="utf-8"))
    if len(report) != 2 or report.keys() != reference.keys():
        raise ValueError("Both runs must cover the same two rulebook pairs.")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.25), sharey=True)
    fig.subplots_adjust(left=.092, right=.962, top=.72, bottom=.245, wspace=.18)
    fig.text(.045, .91, "Can 2,000 scalars learn the rules with more training?",
             fontsize=18, weight="bold", color="#162b39")
    fig.text(.045, .85, "Qwen2.5 0.5B · same coordinate mask, records, and optimizer settings",
             fontsize=10, color="#425565")
    for i, (ax, (seed, draw)) in enumerate(zip(axes, report.items(), strict=True)):
        fixed = draw["budgets"]["2000"]["test_by_epoch"]
        epochs = sorted(int(e) for e in fixed)
        points = [fixed[str(e)]["disagreement"] for e in epochs]
        rates = [p["both_correct"] / p["n"] for p in points]
        ax.plot(epochs, rates, color="#237584", marker="o", linewidth=1.7, label="2,000 scalars")
        for epoch, rate, point in zip(epochs, rates, points, strict=True):
            offset = (5, -13) if epoch == epochs[0] else (5, 6)
            ax.annotate(f"{point['both_correct']}/{point['n']}", (epoch, rate), fontsize=8,
                        textcoords="offset points", xytext=offset, color="#24404e")
        control = reference[seed]["budgets"]["4000"]["test"]["disagreement"]
        ax.scatter([6], [control["both_correct"] / control["n"]], marker="D",
                   color="#6961a0", label="4,000 scalars at six epochs", zorder=4, s=27)
        shortcut = draw["best_fixed_shortcut"]["disagreement"]
        ax.axhline(shortcut["both_correct"] / shortcut["n"], color="#a45a31", linestyle="--",
                   linewidth=1.3, label="Best tested shortcut")
        ax.set_title(f"Rulebook pair {i + 1}", loc="left", weight="bold", pad=9)
        ax.set_xlim(min(epochs) - 1.5, max(epochs) + 2.6)
        ax.set_ylim(-.04, 1.105)
        ax.set_xticks(epochs)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(length=0)
        ax.grid(axis="y", color="#dfe5e9", linewidth=.65)
        ax.set_axisbelow(True)
        ax.set_xlabel("Completed training epochs")
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("Both copies correct")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="lower center", bbox_to_anchor=(.54, .072),
               frameon=False, fontsize=8.5)
    fig.text(.045, .028, "Test parcels use a familiar format. The rulebooks are withheld at every checkpoint.",
             fontsize=8.5, color="#425565")
    fig.savefig(args.run_dir / "training-duration.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
