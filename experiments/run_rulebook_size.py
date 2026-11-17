"""Compare eight parcel assignments with the matched four-depot sparse reference."""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from run_matched_pair import (
    CHOICE_SUFFIX, FIELDS, TEMPLATES, Experiment, Parcel, build_split, check_data, write_json,
)


ADDITIONAL_DEPOTS = ("Elm", "Fir", "Grove", "Hazel")
DATA_PROTOCOLS = {"sparse": "matched-total-data.json", "practice": "matched-per-depot-data.json"}


def balanced_half(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row.depot, row.template, row.printed != row.scanned, row.answer("printed")].append(row)
    rng = random.Random(seed)
    selected = []
    for group in groups.values():
        if len(group) % 2:
            raise ValueError("Each depot, template, agreement, and option stratum must divide in half.")
        selected.extend(rng.sample(group, len(group) // 2))
    rng.shuffle(selected)
    return selected


def prepare(study, reference):
    raw = (reference / "protocol.json").read_bytes()
    prior = json.loads(raw)
    (seed, source), = prior["draws"].items()
    first, second = source["rulebooks"]
    additions = list(FIELDS) * 2
    random.Random(f"{seed}:eight-depot-rules").shuffle(additions)
    extra_a = dict(zip(ADDITIONAL_DEPOTS, additions, strict=True))
    extra_b = {name: FIELDS[1 - FIELDS.index(field)] for name, field in extra_a.items()}
    books = [first | extra_a, second | extra_b]
    if len(books[0]) != 8 or any(list(book.values()).count("printed") != 4 for book in books):
        raise ValueError("The reference must extend to a balanced eight-depot pair.")

    full = {}
    for split, raw_rows in source["examples"].items():
        prior_rows = [Parcel(**row) for row in raw_rows]
        exemplar = [row for row in prior_rows if row.depot == next(iter(first))]
        diff = sum(row.printed != row.scanned for row in exemplar)
        old_lanes = {lane for row in prior_rows for lane in (row.option_a, row.option_b)}
        lanes = range(min(old_lanes), max(old_lanes) + 1)
        additional = build_split(f"{seed}:eight-depot-records", split, lanes, ADDITIONAL_DEPOTS,
                                 diff, len(exemplar) - diff)
        merged = prior_rows + [replace(row, id=f"eight-new-{row.id}") for row in additional]
        random.Random(f"{seed}:{split}:full-order").shuffle(merged)
        full[split] = merged
    matched_total = {split: balanced_half(rows, f"{seed}:{split}:matched-total")
                     for split, rows in full.items()}
    matched_practice = {**matched_total, "train": full["train"]}
    for comparison in (matched_total, matched_practice):
        check_data(comparison, *books)
        ids = [row.id for rows in comparison.values() for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Parcel IDs must be unique across splits.")
    if any(len(matched_total[split]) != len(rows) for split, rows in source["examples"].items()):
        raise ValueError("The eight-depot comparison must match the four-depot split sizes.")

    study.mkdir(parents=True, exist_ok=False)
    for stage, rows in (("sparse", matched_total), ("practice", matched_practice)):
        draw = {"rulebooks": books,
                "examples": {split: [asdict(row) for row in parcels] for split, parcels in rows.items()}}
        write_json(study / DATA_PROTOCOLS[stage], {
            "source_run": str(reference), "source_protocol_sha256": hashlib.sha256(raw).hexdigest(),
            "model": prior["model"], "revision": prior["revision"],
            "training_templates": TEMPLATES[:2], "choice_suffix": CHOICE_SUFFIX,
            "draws": {seed: draw},
        })
        print(stage, {split: len(parcels) for split, parcels in rows.items()}, flush=True)


def run_sparse(args):
    from run_scalar_sweep import ALPHA, RANK, eligible_modules, run_draw, sample_masks

    study = args.output
    data_bytes = (study / DATA_PROTOCOLS[args.stage]).read_bytes()
    data = json.loads(data_bytes)
    (seed, draw), = data["draws"].items()
    reference = Path(data["source_run"])
    prior_bytes = (reference / "protocol.json").read_bytes()
    if hashlib.sha256(prior_bytes).hexdigest() != data["source_protocol_sha256"]:
        raise ValueError("The four-depot reference protocol changed.")
    prior = json.loads(prior_bytes)
    four = json.loads((reference / "results.json").read_text(encoding="utf-8"))["assignment"]
    ordinary = args.ordinary_run or study / "ordinary"
    ordinary_protocol = json.loads((ordinary / "protocol.json").read_text(encoding="utf-8"))
    total_sha = hashlib.sha256((study / DATA_PROTOCOLS["sparse"]).read_bytes()).hexdigest()
    if (ordinary_protocol["data_protocol_sha256"] != total_sha
            or (ordinary_protocol["model"], ordinary_protocol["revision"]) != (prior["model"], prior["revision"])):
        raise ValueError("Ordinary LoRA must use the eight-depot matched-total protocol and checkpoint.")
    ordinary_results = json.loads((ordinary / "results.json").read_text(encoding="utf-8"))
    conventional = ordinary_results["trained_hidden_seen_wording"]
    best_shortcut = max(source["disagreement"]["both_correct"]
                        for source in ordinary_results["shortcut_baselines"].values())
    if conventional["disagreement"]["both_correct"] <= best_shortcut:
        raise ValueError("Ordinary LoRA must beat the shortcut before running the sparse comparison.")
    budgets = sorted(map(int, four["budgets"]))
    saved = {budget: prior["budgets"][str(budget)] for budget in budgets}
    args.epochs = prior["epochs"]
    args.test_epochs = prior["fixed_test_epochs"]
    if args.stage == "practice":
        total = json.loads((study / "sparse" / "results.json").read_text(encoding="utf-8"))
        endpoint = str(args.test_epochs[-1])
        def paired(result, budget):
            return result["budgets"][str(budget)]["test_by_epoch"][endpoint]["disagreement"]["both_correct"]
        budgets = [budget for budget in budgets if paired(total, budget) < paired(four, budget)]
        if not budgets:
            print("Neither eight-depot budget fell below its four-depot reference at the final checkpoint.",
                  flush=True)
            return
        if len(draw["examples"]["train"]) != 2 * len(prior["draws"][seed]["examples"]["train"]):
            raise ValueError("Matched examples per depot must double the eight-depot training set.")
        args.epochs //= 2
        args.test_epochs = [epoch // 2 for epoch in args.test_epochs]

    args.model_id, args.revision = prior["model"], prior["revision"]
    args.batch_size, args.learning_rate = prior["batch_size"], prior["learning_rate"]
    args.basis_seed = prior["basis_seed"]
    updates = math.ceil(len(draw["examples"]["train"]) / args.batch_size)
    matched_steps = [epoch * updates for epoch in args.test_epochs]
    reference_updates = math.ceil(len(prior["draws"][seed]["examples"]["train"]) / args.batch_size)
    if matched_steps != [epoch * reference_updates for epoch in prior["fixed_test_epochs"]]:
        raise ValueError("The evaluation checkpoints must match the four-depot optimizer steps.")
    exp = Experiment(args)
    model = exp.model().requires_grad_(False)
    eligible, masks = sample_masks(eligible_modules(model), sorted(saved), prior["mask_seed"])
    if masks != saved or eligible != prior["eligible_B_coordinates"] or (RANK, ALPHA) != (prior["rank"], prior["alpha"]):
        raise ValueError("The eight-depot runs must use the exact four-depot adapter coordinates and basis.")
    args.output = study / args.stage
    args.output.mkdir(exist_ok=False)
    write_json(args.output / "protocol.json", {
        "data_protocol_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "data_protocol": DATA_PROTOCOLS[args.stage], "source_run": str(reference),
        "source_protocol_sha256": data["source_protocol_sha256"], "ordinary_run": str(ordinary),
        "model": args.model_id, "revision": args.revision, "runtime": exp.config(),
        "training_seed": prior["training_seed"], "basis_seed": args.basis_seed, "mask_seed": prior["mask_seed"],
        "rank": RANK, "alpha": ALPHA, "eligible_B_coordinates": eligible,
        "epochs": args.epochs, "fixed_test_epochs": args.test_epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "budgets": {budget: masks[budget] for budget in budgets},
        "matched_optimizer_steps": matched_steps,
    })
    result = run_draw(exp, model, seed, draw, conventional, {budget: masks[budget] for budget in budgets},
                      args, prior["training_seed"])
    write_json(args.output / "results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "sparse", "practice"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ordinary-run", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.reference is None:
            parser.error("Preparing the design requires the four-depot --reference.")
        prepare(args.output, args.reference)
    else:
        if args.checkpoint is None:
            parser.error("Training requires the pinned local --checkpoint.")
        run_sparse(args)


if __name__ == "__main__":
    main()
