"""Plot the held-out results of an exact scalar sweep."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--previous-run", type=Path, help="Include the earlier points from the same nested sweep.")
    args = parser.parse_args()
    report = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.run_dir / "protocol.json").read_text(encoding="utf-8"))
    if args.previous_run:
        prior = json.loads((args.previous_run / "results.json").read_text(encoding="utf-8"))
        if protocol["reference_sweep"] != args.previous_run.name or prior.keys() != report.keys():
            raise ValueError("The runs must share the same draws and nested mask.")
        for seed, draw in report.items():
            if set(prior[seed]["budgets"]) & draw["budgets"].keys():
                raise ValueError("The two runs contain overlapping budgets.")
            draw["budgets"] = prior[seed]["budgets"] | draw["budgets"]
    if len(report) != 2:
        raise ValueError("Both rulebook pairs must complete before plotting.")
    budgets = sorted(int(x) for x in next(iter(report.values()))["budgets"])
    dense = [n for n in budgets if next(iter(report.values()))["budgets"][str(n)].get("dense_b")]
    sparse = [n for n in budgets if n not in dense]
    fig, panels = plt.subplots(1, 2, figsize=(10.2, 4.15), sharey=True)
    fig.subplots_adjust(left=.095, right=.98, top=.73, bottom=.245, wspace=.22)
    fig.text(.046, .92, "Exact trainable-scalar sweep", fontsize=19, weight="bold", color="#162b39")
    fig.text(.046, .855, "Qwen2.5 0.5B · new parcels in a familiar format · rules withheld",
             fontsize=10, color="#425565")
    palette = ("#237584", "#574894")
    for ax, stratum, title in zip(panels, ("disagreement", "agreement"),
                                 ("Correct answers differ", "Correct answers agree"), strict=True):
        for i, (seed, draw) in enumerate(report.items()):
            rates = {}
            for n in budgets:
                count = draw["budgets"][str(n)]["test"][stratum]
                rates[n] = count["both_correct"] / count["n"]
            marker = {"marker": ("o", "s")[i], "color": palette[i],
                      "markersize": (5, 7)[i], "markerfacecolor": (palette[0], "none")[i]}
            ax.plot(sparse, [rates[n] for n in sparse], **marker,
                    label=f"Rulebook pair {i + 1}", linewidth=1.7)
            if dense:
                ax.plot(dense, [rates[n] for n in dense], **marker, linestyle="none")
        controls = report[next(iter(report))]
        for key, color, style, label in (
            ("conventional_lora", "#7c8794", ":", "Conventional LoRA"),
            ("best_fixed_shortcut", "#a45a31", "--", "Best shortcut on disagreements"),
        ):
            if key == "best_fixed_shortcut" and stratum == "agreement":
                continue
            count = controls[key][stratum]
            ax.axhline(count["both_correct"] / count["n"], color=color,
                       linestyle=style, linewidth=1.4, label=label)
        ax.set_title(title, loc="left", weight="bold", pad=10)
        ax.set_xscale("log")
        labels = [f"{n / 1e6:.2f}M" if n in dense else f"{n:,}" for n in budgets]
        ax.set_xticks(budgets, labels)
        ax.set_xlim(min(budgets) / 1.5, max(budgets) * 1.5)
        ax.set_ylim(-.04, 1.06)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Trainable adapter scalars")
        ax.grid(axis="y", color="#dfe5e9", linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, axis="both")
        ax.tick_params(which="minor", bottom=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
    panels[0].set_ylabel("Both copies correct")
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=4, loc="lower center",
               bbox_to_anchor=(.53, .065), fontsize=8.5)
    footnote = ("The 2.43M point trains all B entries at a separate learning rate. Each point counts paired correctness."
                if dense else "One nested mask over LoRA B. Each point requires both copies to be correct on the same record.")
    fig.text(.046, .025, footnote, fontsize=8.5, color="#425565")
    fig.savefig(args.run_dir / "scalar-sweep.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
