"""Create the confirmatory tables and figures from verified run artifacts."""

import csv
import platform
from importlib.metadata import version
from statistics import fmean

from replay_confirmatory import load_verified_replay
from run_confirmatory import ANALYSIS, BUDGETS, EDL_CONDITIONS, MASK_SEEDS, RULEBOOK_SEEDS, write_json
from verify_confirmatory import load_verified_draws, require


WORDINGS = ("familiar", "novel")
STEPS = ANALYSIS["sparse_steps"]
CONTRASTS = (("format-only", "prior-rulebook"), ("opposite-rulebook", "format-only"),
             ("opposite-rulebook", "prior-rulebook"))


def score_columns(scores):
    return {f"{stratum}_{key}": value for stratum, counts in scores.items() for key, value in counts.items()}


def fit_row(draw, method, copy, measurements, **settings):
    return {"draw": draw, "method": method, "copy": copy,
            **dict.fromkeys(("budget", "mask_seed", "edl_rate", "condition"), ""), **settings,
            **{key: measurements[key] for key in ("seconds_train_and_dev", "peak_allocated_bytes", "peak_reserved_bytes")}}


def build_tables(draws):
    sparse, duration, ordinary, edl, contrasts, fits = [], [], [], [], [], []
    for seed, draw in draws.items():
        require({(item["budget"], item["mask_seed"]) for item in draw["sparse"]}
                == {(budget, mask) for budget in BUDGETS for mask in MASK_SEEDS},
                f"Missing sparse conditions for {seed}.")
        for condition in ("frozen_hidden", "frozen_visible", "trained_hidden_seen_wording", "trained_hidden_novel_wording"):
            ordinary.append({"draw": seed, "condition": condition, **score_columns(draw["ordinary"][condition])})
        for copy, fit in draw["ordinary"]["training"].items():
            fits.append(fit_row(seed, "ordinary-lora", copy, fit))
        for item in draw["sparse"]:
            group = {"draw": seed, "mask_seed": item["mask_seed"], "budget": item["budget"]}
            for wording in WORDINGS:
                points = [score_columns(item["points"][str(step)][wording]) for step in STEPS]
                for step, counts in zip(STEPS, points, strict=True):
                    sparse.append({**group, "wording": wording, "steps": step, **counts})
                first, final = (p["disagreement_both_correct"] for p in (points[0], points[-1]))
                duration.append({**group, "wording": wording, "paired_at_144": first, "paired_at_432": final,
                                 "paired_change_144_to_432": final - first})
            for i, copy in enumerate("AB"):
                fits.append(fit_row(seed, "sparse", copy, {key: item[key][i] for key in (
                    "seconds_train_and_dev", "peak_allocated_bytes", "peak_reserved_bytes",
                )}, budget=item["budget"], mask_seed=item["mask_seed"]))
        for run in draw["edl"]:
            group = {"draw": seed, "rate": run["rate"], "endpoints_comparable": run["endpoints_comparable"],
                     "final_loss_span_bits_per_label": run["final_loss_span_bits_per_label"]}
            for condition, item in run["conditions"].items():
                edl.append({**group, "condition": condition, **score_columns(item["paired_accuracy"]),
                            "final_bits_per_label": item["mean_final_bits_per_label"],
                            **{f"edl_{k}_bits_per_label": v for k, v in item["mean_edl_bits_per_label"].items()}})
                for copy, fit in item["training"].items():
                    fits.append(fit_row(seed, "edl-adaptation", copy, fit, edl_rate=run["rate"], condition=condition))
            for copy, fit in run.get("format_exposure_training", {}).items():
                fits.append(fit_row(seed, "format-exposure", copy, fit, condition="format-only"))
            for left, right in CONTRASTS:
                a, b = [run["conditions"][k]["mean_edl_bits_per_label"] for k in (left, right)]
                contrasts.append({**group, "contrast": f"{left} minus {right}",
                                  **{f"edl_{k}_difference_bits_per_label": a[k] - b[k] for k in a}})
    means, ranges = draw_means(sparse, duration)
    tables = {"sparse-checkpoints": sparse, "sparse-duration": duration, "sparse-draw-means": means,
              "sparse-draw-ranges": ranges, "ordinary": ordinary, "edl-conditions": edl,
              "edl-contrasts": contrasts, "training-fits": fits}
    require(tuple(map(len, tables.values())) == (72, 24, 36, 16, 12, 12, 12, 58),
            "Unexpected coverage in study tables.")
    return tables


def draw_means(sparse, duration):
    means, ranges = [], []
    for budget in BUDGETS:
        for wording in WORDINGS:
            for step in (*STEPS, "432 minus 144"):
                by_draw = []
                for draw in RULEBOOK_SEEDS:
                    if isinstance(step, int):
                        counts = [r["disagreement_both_correct"] for r in sparse if
                                  (r["draw"], r["budget"], r["wording"], r["steps"]) == (draw, budget, wording, step)]
                    else:
                        counts = [r["paired_change_144_to_432"] for r in duration if
                                  (r["draw"], r["budget"], r["wording"]) == (draw, budget, wording)]
                    require(len(counts) == len(MASK_SEEDS), "A draw mean must include both frozen masks.")
                    value = fmean(counts)
                    by_draw.append(value)
                    if isinstance(step, int):
                        means.append({"budget": budget, "wording": wording, "steps": step, "draw": draw,
                                      "mean_paired_count": value, "denominator": 128})
                ranges.append({"budget": budget, "wording": wording, "steps": step,
                               "mean_of_draw_means": fmean(by_draw), "min_draw_mean": min(by_draw),
                               "max_draw_mean": max(by_draw), "denominator": 128})
    return means, ranges


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_tables(tables):
    text = ["# Confirmatory results", "", "## Sparse paired scores on disagreement parcels", "",
            "| Draw | Mask | Weights | Familiar 144 | Familiar 288 | Familiar 432 | Unseen 144 | Unseen 288 | Unseen 432 | Familiar change | Unseen change |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    scores = {(r["draw"], r["mask_seed"], r["budget"], r["wording"], r["steps"]): r["disagreement_both_correct"]
              for r in tables["sparse-checkpoints"]}
    for draw in RULEBOOK_SEEDS:
        for mask in MASK_SEEDS:
            for budget in BUDGETS:
                points = [[scores[draw, mask, budget, wording, step] for step in STEPS] for wording in WORDINGS]
                changes = [f"{counts[-1] - counts[0]:+d}" for counts in points]
                values = [str(draw), str(mask), f"{budget:,}", *(f"{n}/128" for counts in points for n in counts), *changes]
                text.append("| " + " | ".join(values) + " |")
    text += ["", "## EDL on all labels and paired accuracy on disagreements", "",
             "| Draw | Learning rate | Condition | EDL (bits/label) | Final loss (bits/label) | Paired | Comparable endpoints |",
             "| --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for r in tables["edl-conditions"]:
        text.append(f"| {r['draw']} | {r['rate']:.5f} | {r['condition']} | {r['edl_all_bits_per_label']:.6f} | "
                    f"{r['final_bits_per_label']:.6f} | {r['disagreement_both_correct']}/288 | "
                    f"{'Yes' if r['endpoints_comparable'] else 'No'} |")
    return "\n".join(text) + "\n"


def make_figures(output, tables):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = tables["sparse-checkpoints"]
    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(9.2, 3.6), layout="constrained")
    for wording, axis in zip(WORDINGS, axes, strict=True):
        for budget, color in zip(BUDGETS, ("#17699b", "#98620b"), strict=True):
            for draw in RULEBOOK_SEEDS:
                for mask in MASK_SEEDS:
                    values = [r["disagreement_both_correct"] for r in rows if
                              (r["wording"], r["budget"], r["draw"], r["mask_seed"]) == (wording, budget, draw, mask)]
                    axis.plot(STEPS, values, color=color, alpha=0.22, linewidth=0.9)
            values = [r["mean_of_draw_means"] for r in tables["sparse-draw-ranges"] if
                      (r["wording"], r["budget"]) == (wording, budget) and isinstance(r["steps"], int)]
            axis.plot(STEPS, values, marker="o", color=color, linewidth=2.4, label=f"{budget:,} weights")
        axis.axhline(64, color="#555555", linestyle="--", linewidth=1, label="Fixed shortcut")
        axis.set(title="Familiar wording" if wording == "familiar" else "Unseen wording",
                 xlabel="Optimizer steps", xticks=STEPS, ylim=(-3, 131), yticks=(0, 32, 64, 96, 128))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Paired correct out of 128")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    figure.savefig(output / "sparse-duration.png", dpi=220)
    plt.close(figure)

    edl_rows = tables["edl-conditions"]
    groups = sorted({(r["draw"], r["rate"]) for r in edl_rows}, key=lambda k: (k[0], -k[1]))
    figure, axes = plt.subplots(2, 2, sharey=True, figsize=(8.6, 5.7), layout="constrained")
    for (draw, rate), axis in zip(groups, axes.flat, strict=True):
        rows = [r for condition in EDL_CONDITIONS for r in edl_rows if
                (r["draw"], r["rate"], r["condition"]) == (draw, rate, condition)]
        comparable = rows[0]["endpoints_comparable"]
        containers = axis.bar(["Target rule", "Format only", "Opposite rule"],
                              [r["edl_all_bits_per_label"] for r in rows],
                              color=("#17699b", "#777777", "#98620b"), edgecolor="#444444", linewidth=0.3)
        if not comparable:
            for bar in containers:
                bar.set_hatch("//")
        status = "Comparable endpoints" if comparable else "Endpoint criteria failed"
        axis.set_title(f"{draw}, learning rate {rate:.5f}\n{status}", fontsize=10)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.supylabel("EDL (bits per label)")
    figure.savefig(output / "edl-conditions.png", dpi=220)
    plt.close(figure)


def report_study(root):
    inputs_hash, draws = load_verified_draws(root)
    replay = load_verified_replay(root, inputs_hash, draws)
    tables = build_tables(draws)
    output = root / "report"
    output.mkdir(exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    (output / "tables.md").write_text(markdown_tables(tables), encoding="utf-8", newline="\n")
    write_json(output / "summary.json", {"inputs_sha256": inputs_hash, "study_fits": 58, "replay_fits": 2,
                                        "replay": replay, "sparse_draw_ranges": tables["sparse-draw-ranges"],
                                        "edl_conditions": tables["edl-conditions"], "edl_contrasts": tables["edl-contrasts"],
                                        "report_runtime": {"python": platform.python_version(),
                                                           "matplotlib": version("matplotlib"), "numpy": version("numpy")}})
    make_figures(output, tables)
    print(f"Wrote verified results for 58 study fits and 2 replay fits to {output}.", flush=True)
