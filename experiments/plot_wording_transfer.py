"""Plot template-level results from the matched wording experiment."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


LABELS = {
    "familiar": "Familiar wording",
    "original-unseen": "Earlier unseen wording",
    "email": "New · email",
    "answer-before-record": "New · question first",
    "inspection-notes": "New · inspection notes",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    results = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    if len(results) != 2:
        raise ValueError("Both rulebook pairs must finish before plotting the comparison.")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11, 4.8))
    fig.subplots_adjust(left=.24, right=.97, bottom=.19, top=.70, wspace=.27)
    fig.text(.037, .91, "Does more training wording help rule transfer?",
             fontsize=19, weight="bold", color="#162b39")
    fig.text(.037, .855, "Paired accuracy on the same unseen parcels, with rulebooks withheld",
             fontsize=10, color="#425565")
    panels = ("First rulebook pair", "New rulebook pair")
    for ax, panel, draw in zip(axes, panels, results.values(), strict=True):
        for arm, offset, color, label in (
            ("two-templates", -.17, "#788694", "2 training templates"),
            ("eight-templates", .17, "#237584", "8 training templates"),
        ):
            counts = [draw[arm]["per_template"][s]["disagreement"] for s in LABELS]
            rates = [x["both_correct"] / x["n"] for x in counts]
            bars = ax.barh([i + offset for i in range(len(counts))], rates, height=.29,
                           color=color, label=label)
            ax.bar_label(bars, [f"{x['both_correct']}/{x['n']}" for x in counts], padding=3, fontsize=8)
        ax.axvline(draw["best_shortcut_disagreement_count"] / counts[0]["n"],
                   color="#995d2e", linestyle="--", linewidth=1.3, label="Best tested shortcut")
        ax.set_title(panel, loc="left", weight="bold", pad=10)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_yticks(range(len(LABELS)), list(LABELS.values()))
        ax.tick_params(length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e1e7eb", linewidth=.65)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="lower center", frameon=False,
               bbox_to_anchor=(.59, .085))
    fig.text(.037, .035, "Each prompt style reuses the same 128 disagreement parcels within its rulebook pair.",
             fontsize=9, color="#425565")
    fig.savefig(args.run_dir / "wording-transfer.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
