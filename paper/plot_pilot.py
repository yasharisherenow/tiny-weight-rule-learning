"""Render Figure 1 of the Tiny Weight Rule Learning draft."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent
RUNS = (
    ("qwen-full-book", "Qwen3 4B\nFull rulebook"),
    ("ministral-full-book", "Ministral 3 3B\nFull rulebook"),
    ("ministral-applicable-rule", "Ministral 3 3B\nApplicable rule only"),
)


def main():
    with (ROOT / "pilot-data.csv").open(newline="", encoding="utf-8") as source:
        by_run = defaultdict(list)
        for row in csv.DictReader(source):
            by_run[row["run"]].append(row)

    labels = []
    paired_disagreement = []
    paired_agreement = []
    hidden_agreement = []
    for name, label in RUNS:
        labels.append(label)
        records = by_run[name]
        disagreements = [r for r in records if r["expected_a"] != r["expected_b"]]
        agreements = [r for r in records if r["expected_a"] == r["expected_b"]]
        if len(disagreements) != 16 or len(agreements) != 16:
            raise ValueError(f"{name} must include 16 agreement and 16 disagreement items")

        for subset, counts in ((disagreements, paired_disagreement),
                               (agreements, paired_agreement)):
            counts.append(sum(r["answer_a"] == r["expected_a"]
                              and r["answer_b"] == r["expected_b"] for r in subset))
        hidden_agreement.append(sum(r["answer_hidden"] == r["expected_a"] for r in agreements))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": "#253345",
        "axes.titlecolor": "#142338",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#a9b4c1",
        "xtick.color": "#253345",
        "ytick.color": "#253345",
        "svg.fonttype": "none",
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    fig.subplots_adjust(left=0.079, right=0.975, top=0.71, bottom=0.21, wspace=0.24)
    fig.text(0.08, 0.91, "Parcel-routing pilot", fontsize=21, weight="bold", color="#142338")
    fig.text(0.08, 0.853, "Frozen models · 16 records per panel · one rulebook pair",
             fontsize=10, color="#435369")

    positions = list(range(len(RUNS)))
    shown = "#206b78"
    withheld = "#a95a27"
    left_bars = left.bar(positions, paired_disagreement, 0.48, color=shown)
    right_bars = right.bar([x - 0.17 for x in positions], paired_agreement, 0.32,
                           color=shown, label="Correct under both shown rules")
    hidden_bars = right.bar([x + 0.17 for x in positions], hidden_agreement, 0.32,
                            color=withheld, label="Correct with no rule shown")

    for ax, title in ((left, "A   Correct answers differ"), (right, "B   Correct answers agree")):
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=12)
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, 18)
        ax.set_xlim(-0.55, len(RUNS) - 0.45)
        ax.yaxis.set_major_locator(MultipleLocator(4))
        ax.grid(axis="y", color="#dce3ea")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, pad=7)
    left.set_ylabel("Correct records out of 16")
    for ax, bars in ((left, left_bars), (right, right_bars), (right, hidden_bars)):
        ax.bar_label(bars, padding=4, color="#142338", fontsize=10)
    handles, descriptions = right.get_legend_handles_labels()
    fig.legend(handles, descriptions, loc="lower center", ncols=2, frameon=False,
               bbox_to_anchor=(0.53, 0.025))

    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(destination / f"pilot.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Paired disagreements: {paired_disagreement}")
    print(f"Paired agreements: {paired_agreement}")
    print(f"Agreement without a rule: {hidden_agreement}")


if __name__ == "__main__":
    main()
