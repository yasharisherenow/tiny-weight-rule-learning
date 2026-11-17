"""Render the two main figures from the saved parcel-routing experiments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


PAPER = Path(__file__).resolve().parent
RUNS = PAPER.parent / "output" / "experiments"
COLORS = ("#126e78", "#9a4e23")
SHORTCUT = "#6b6573"


def read_results(run):
    return json.loads((RUNS / run / "results.json").read_text(encoding="utf-8"))


def accuracy(metrics):
    counts = metrics["disagreement"]
    return counts["both_correct"] / counts["n"]


def format_axis(ax):
    ax.set_ylim(-0.035, 1.045)
    ax.set_yticks((0, .25, .5, .75, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color="#dce3e7", linewidth=.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#bbc6ce")


def save(fig, name):
    destination = PAPER / "figures"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_budgets():
    original = read_results("qwen-05b-scalar-sweep")
    bridge = read_results("qwen-05b-bridge-sweep")
    replications = read_results("qwen-05b-mask-replication")["draws"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.25), sharey=True)
    fig.subplots_adjust(left=.09, right=.973, top=.74, bottom=.242, wspace=.2)
    fig.text(.09, .93, "Paired accuracy across trainable weight budgets", fontsize=17, weight="bold")
    fig.text(.09, .867, "Qwen2.5 0.5B · six-epoch training schedule · familiar wording", fontsize=10)
    for i, (ax, (seed, draw)) in enumerate(zip(axes, original.items(), strict=True), start=1):
        format_axis(ax)
        ax.set_title(f"Rulebook pair {i}", loc="left", weight="bold", pad=9)
        ax.set_xscale("log")
        for mask, style, marker in (("120", "--", "s"), ("121", ":", "^")):
            values = replications[seed]["fresh_masks"][mask]
            ax.plot([int(budget) for budget in values],
                    [count / replications[seed]["test_count"] for count in values.values()],
                    color=COLORS[1], linestyle=style, marker=marker, markersize=5,
                    markerfacecolor="white", label=f"Mask {mask}")
        nested = {**draw["budgets"],
                  **{k: v for k, v in bridge[seed]["budgets"].items() if int(k) <= 100000}}
        ax.plot(list(map(int, nested)), [accuracy(v["test"]) for v in nested.values()],
                color=COLORS[0], marker="o", markersize=4.8, label="Mask 119")
        ax.axhline(accuracy(draw["best_fixed_shortcut"]), color=SHORTCUT, linestyle="-.",
                   linewidth=1.1, label="Best fixed shortcut")
        ax.axhline(accuracy(draw["conventional_lora"]), color="#a4abb4", linewidth=1.05,
                   zorder=1, label="Ordinary LoRA")
        ax.set_xticks([10, 100, 1000, 10000, 100000], ["10", "100", "1k", "10k", "100k"])
        ax.set_xlabel("Trainable adapter weights")
    axes[0].set_ylabel("Both copies correct")
    handles, labels = axes[0].get_legend_handles_labels()
    order = (2, 0, 1, 3, 4)
    fig.legend([handles[i] for i in order], [labels[i] for i in order], ncols=5,
               loc="lower center", bbox_to_anchor=(.53, .055), frameon=False, fontsize=9)
    save(fig, "weight-budget")


def plot_duration():
    first = read_results("qwen-05b-1000-scalar-duration")
    second = read_results("qwen-05b-training-duration")
    confirmation = read_results("qwen-05b-budget-confirmation")
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.25), sharey=True)
    fig.subplots_adjust(left=.09, right=.974, top=.745, bottom=.252, wspace=.25)
    fig.text(.09, .93, "Paired accuracy as training continues", fontsize=17, weight="bold")
    fig.text(.09, .867, "Qwen2.5 0.5B · fixed checkpoints · familiar wording", fontsize=10)
    prior = [(first[seed]["budgets"]["1000"], second[seed]["budgets"]["2000"],
              first[seed]["best_fixed_shortcut"]) for seed in first]
    current = confirmation["assignment"]
    draws = [*prior, (current["budgets"]["1000"], current["budgets"]["2000"],
                      current["best_fixed_shortcut"])]
    for i, (ax, (small, large, shortcut)) in enumerate(zip(axes, draws, strict=True)):
        format_axis(ax)
        title = ("Pair 1 · mask 120", "Pair 2 · mask 120", "Pair 3 · mask 122")[i]
        ax.set_title(title, loc="left", weight="bold", pad=9)
        ax.axhline(accuracy(shortcut), color=SHORTCUT, linestyle="-.", linewidth=1.1)
        for result, color, marker in zip((small, large), COLORS, ("o", "s"), strict=True):
            metrics = result["test_by_epoch"]
            ax.plot(list(map(int, metrics)), [accuracy(v) for v in metrics.values()],
                    color=color, marker=marker, linewidth=1.6, markersize=5)
        if i == 2:
            for budget, color, marker in zip(("1000", "2000"), COLORS, ("o", "s"), strict=True):
                metrics = confirmation["format-only"]["budgets"][budget]["test_by_epoch"]
                ax.plot(list(map(int, metrics)), [accuracy(v) for v in metrics.values()],
                        color=color, linestyle=":", marker=marker, markerfacecolor="white",
                        linewidth=1.35, markersize=5)
        ax.set_xticks([6, 12, 18])
        ax.set_xlim(4.6, 19.4)
        ax.set_xlabel("Completed epochs")
    axes[0].set_ylabel("Both copies correct")
    handles = [
        Line2D([], [], color=COLORS[0], marker="o", label="1,000 weights"),
        Line2D([], [], color=COLORS[1], marker="s", label="2,000 weights"),
        Line2D([], [], color="#51535e", linestyle=":", marker="o", markerfacecolor="white",
               label="Format-only labels"),
        Line2D([], [], color=SHORTCUT, linestyle="-.", label="Best fixed shortcut"),
    ]
    fig.legend(handles=handles, ncols=4, loc="lower center", bbox_to_anchor=(.53, .055),
               frameon=False, fontsize=9)
    save(fig, "training-duration")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#1d2b37", "axes.labelcolor": "#1d2b37",
                         "xtick.color": "#344655", "ytick.color": "#344655"})
    plot_budgets()
    plot_duration()
