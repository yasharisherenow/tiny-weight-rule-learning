"""Confirm the Qwen training-duration result on a new rulebook pair and coordinate mask."""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from run_matched_pair import (
    CHOICES, CHOICE_SUFFIX, MODEL_ID, REVISION, TEMPLATES, Experiment,
    build_data, check_data, write_json,
)
from run_scalar_sweep import ALPHA, RANK, eligible_modules, run_draw, sample_masks


RULE_SEED = 81720
MASK_SEED = 122
BUDGETS = (1000, 2000)


def format_only_labels(rows, *, rule_seed=RULE_SEED, separate_templates=False):
    labels = [row.answer("printed") for row in rows]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row.printed != row.scanned:
            key = (row.depot, labels[i], row.template) if separate_templates else (row.depot, labels[i])
            groups[key].append(i)

    arms = {}
    for copy in CHOICES:
        label_seed = f"{rule_seed}:format-only:{copy}"
        rng = random.Random(label_seed)
        targets = labels.copy()
        for indices in groups.values():
            if len(indices) % 2:
                raise ValueError("Format-only labels require even disagreement group sizes.")
            for i in rng.sample(indices, len(indices) // 2):
                targets[i] = rows[i].answer("scanned")
        arms[copy] = {"seed": label_seed, "labels": targets}
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wording-run", required=True, type=Path)
    parser.add_argument("--duration-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    wording = json.loads((args.wording_run / "protocol.json").read_text(encoding="utf-8"))
    prior = json.loads((args.duration_run / "protocol.json").read_text(encoding="utf-8"))
    if prior["model"] != MODEL_ID or prior["revision"] != REVISION:
        raise ValueError("The reference duration run must use the pinned Qwen2.5 0.5B checkpoint.")
    if prior["epochs"] != 18 or prior["fixed_test_epochs"] != [6, 12, 18]:
        raise ValueError("The reference duration run must report fixed epochs 6, 12, and 18.")
    if (prior["rank"], prior["alpha"]) != (RANK, ALPHA) or MASK_SEED == prior["mask_seed"]:
        raise ValueError("The confirmation must preserve the LoRA architecture and use a new mask.")

    args.model_id, args.revision = MODEL_ID, REVISION
    args.epochs = prior["epochs"]
    args.test_epochs = prior["fixed_test_epochs"]
    args.batch_size = prior["batch_size"]
    args.learning_rate = prior["learning_rate"]
    args.basis_seed = prior["basis_seed"]
    training_seed = prior["training_seed"]

    rules_a, rules_b, rows = build_data(RULE_SEED, test_lanes=range(900, 1000))
    rows = {split: [replace(row, id=f"confirmation-{row.id}") for row in records]
            for split, records in rows.items()}
    check_data(rows, rules_a, rules_b)
    test_lanes = {lane for row in rows["test"] for lane in (row.option_a, row.option_b)}
    for previous in wording["draws"].values():
        if rules_a in previous["rulebooks"]:
            raise ValueError("The confirmation must use a previously untested rulebook pair.")
        if any(test_lanes.intersection((row["option_a"], row["option_b"]))
               for split in previous["examples"].values() for row in split):
            raise ValueError("Confirmation test lane values must be new.")

    draw = {"rulebooks": [rules_a, rules_b],
            "examples": {split: [asdict(r) for r in records] for split, records in rows.items()}}
    controls = format_only_labels(rows["train"])
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output
    exp = Experiment(args)
    model = exp.model().requires_grad_(False)
    eligible, masks = sample_masks(eligible_modules(model), BUDGETS, MASK_SEED)
    write_json(root / "protocol.json", {
        "model": MODEL_ID,
        "revision": REVISION,
        "runtime": exp.config(),
        "reference_duration_run": args.duration_run.name,
        "training_seed": training_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "fixed_test_epochs": args.test_epochs,
        "training_templates": TEMPLATES[:2],
        "choice_suffix": CHOICE_SUFFIX,
        "draws": {RULE_SEED: draw},
        "mask_seed": MASK_SEED,
        "basis_seed": args.basis_seed,
        "rank": RANK,
        "alpha": ALPHA,
        "eligible_B_coordinates": eligible,
        "budgets": masks,
        "format_only_labels_in_training_record_order": controls,
        "primary": "At fixed epochs 6, 12, and 18, report paired correctness on disagreement "
                   "records in familiar wording with the rulebooks withheld.",
        "comparison": "Compare each budget with the best fixed shortcut at each epoch and "
                      "with its format-only control at the same epoch.",
    })

    results = {}
    arms = {"assignment": None, "format-only": [controls[copy]["labels"] for copy in CHOICES]}
    for name, training_labels in arms.items():
        args.output = root / name
        args.output.mkdir()
        print(f"Training condition: {name}", flush=True)
        results[name] = run_draw(exp, model, RULE_SEED, draw, None, masks, args, training_seed,
                                 training_labels=training_labels)
        write_json(root / "results.json", results)


if __name__ == "__main__":
    main()
