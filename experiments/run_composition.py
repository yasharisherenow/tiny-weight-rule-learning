"""Train a matched LoRA pair and test held-out depot and package-class combinations."""

import argparse
import gc
import random
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from run_matched_pair import (
    CHOICES, CHOICE_SUFFIX, DEPOTS, FIELDS, TARGET_MODULES, Experiment, Parcel,
    checkpoint_arguments, pair_metrics, resolve_revision, select, shortcut_metrics, write_json,
)


CLASSES = ("Amber", "Cobalt", "Jade", "Slate")
TEMPLATE = (
    "Depot: {depot}\nPackage class: {package_class}\nPrinted lane: {printed}\n"
    "Scanned lane: {scanned}\nWhere should the parcel go?"
)
TEST_SPLITS = ("test_seen", "test_held_out")


@dataclass(frozen=True)
class ComposedParcel(Parcel):
    package_class: str

    def prompt(self):
        return (TEMPLATE + CHOICE_SUFFIX).format(**asdict(self))

    @property
    def cell(self):
        return self.depot, self.package_class


def field(start, reverse):
    return FIELDS[FIELDS.index(start) ^ reverse]


def make_records(seed, split, cells, lanes, disagreement, agreement):
    rng = random.Random(f"{seed}:{split}")
    profiles, used = [], set()
    for i in range(disagreement + agreement):
        printed, alternative = rng.sample(lanes, 2)
        key = tuple(sorted((printed, alternative)))
        while key in used:
            printed, alternative = rng.sample(lanes, 2)
            key = tuple(sorted((printed, alternative)))
        used.add(key)
        option_a, option_b = (printed, alternative) if i % 2 == 0 else (alternative, printed)
        profiles.append((printed, alternative if i < disagreement else printed, option_a, option_b))
    # Match all numeric profiles across cells, including across the two test conditions.
    return [
        ComposedParcel(
            id=f"{split}-{depot}-{package_class}-{i + 1:02}",
            depot=depot, package_class=package_class, printed=printed, scanned=scanned,
            option_a=option_a, option_b=option_b, template=0,
        )
        for depot, package_class in sorted(cells)
        for i, (printed, scanned, option_a, option_b) in enumerate(profiles)
    ]


def make_design(seed):
    rng = random.Random(seed)
    depots, classes = list(DEPOTS), list(CLASSES)
    rng.shuffle(depots)
    rng.shuffle(classes)
    start_offset, class_offset = rng.randrange(2), rng.randrange(2)
    depot_a = {depot: FIELDS[(i + start_offset) % 2] for i, depot in enumerate(depots)}
    depot_b = {depot: field(start, True) for depot, start in depot_a.items()}
    reversals = {package_class: bool((i + class_offset) % 2) for i, package_class in enumerate(classes)}
    # The eight training cells form a connected cycle with balanced one-factor marginals.
    seen = {(depot, classes[j % len(classes)])
            for i, depot in enumerate(depots) for j in (i, i + 1)}
    held_out = set(product(DEPOTS, CLASSES)) - seen
    rows = {
        "train": make_records(seed, "train", seen, range(100, 300), 36, 12),
        "dev": make_records(seed, "dev", seen, range(300, 500), 8, 4),
        "test_seen": make_records(seed, "test", seen, range(500, 700), 16, 8),
        "test_held_out": make_records(seed, "test", held_out, range(500, 700), 16, 8),
    }
    for name, records in rows.items():
        random.Random(f"{seed}:order:{name}").shuffle(records)
    return (depot_a, depot_b), reversals, seen, held_out, rows


def answers(rows, book, reversals):
    return [row.answer(field(book[row.depot], reversals[row.package_class])) for row in rows]


def validate_design(books, reversals, seen, held_out, rows):
    expected_cells = {"train": seen, "dev": seen, "test_seen": seen, "test_held_out": held_out}
    if len(seen) != 8 or len(held_out) != 8 or seen & held_out:
        raise ValueError("The design must split the sixteen cells into eight seen and eight held out.")
    ids = [row.id for split in rows.values() for row in split]
    if len(ids) != len(set(ids)):
        raise ValueError("Record IDs must be unique.")
    pools = [{lane for row in rows[name] for lane in (row.option_a, row.option_b)}
             for name in ("train", "dev", "test_seen")]
    held_lanes = {lane for row in rows["test_held_out"] for lane in (row.option_a, row.option_b)}
    if any(pools[i] & pools[j] for i, j in ((0, 1), (0, 2), (1, 2))) or held_lanes != pools[2]:
        raise ValueError("Lane profiles must be shared at test and disjoint across train, dev, and test.")

    for name, records in rows.items():
        if {row.cell for row in records} != expected_cells[name]:
            raise ValueError(f"Unexpected combinations in {name}.")
        labels = [answers(records, book, reversals) for book in books]
        groups = Counter()
        for row, label_a, label_b in zip(records, *labels, strict=True):
            disagreement = row.printed != row.scanned
            if (label_a != label_b) != disagreement:
                raise ValueError("The copies must disagree exactly when the lane fields differ.")
            groups[row.cell, disagreement, label_a] += 1
        if any(groups[cell, disagreement, "A"] != groups[cell, disagreement, "B"]
               or not groups[cell, disagreement, "A"]
               for cell in expected_cells[name] for disagreement in (False, True)):
            raise ValueError(f"Answer positions must be balanced in each cell of {name}.")

    cells = sorted(seen | held_out)
    target = {cell: field(books[0][cell[0]], reversals[cell[1]]) for cell in cells}
    consistent = set()
    for starts in product(FIELDS, repeat=len(DEPOTS)):
        for flags in product((False, True), repeat=len(CLASSES)):
            depot_rules = dict(zip(DEPOTS, starts, strict=True))
            class_rules = dict(zip(CLASSES, flags, strict=True))
            candidate = {cell: field(depot_rules[cell[0]], class_rules[cell[1]]) for cell in cells}
            if all(candidate[cell] == target[cell] for cell in seen):
                consistent.add(tuple(candidate[cell] for cell in cells))
    if consistent != {tuple(target[cell] for cell in cells)}:
        raise ValueError("Training cells must determine the held-out answers under the two-rule task.")


def shortcuts(rows, labels):
    fixed = shortcut_metrics(rows, *labels)
    disagreement_rows = [row for row in rows if row.printed != row.scanned]
    ceilings = {}
    for feature in ("depot", "package_class"):
        # Other data in the prompt is matched exactly across cells. Even an oracle
        # that sees every test label gets no advantage from those numeric fields.
        by_group = Counter(
            (getattr(row, feature), row.printed, row.scanned, row.option_a, row.option_b, answer)
            for row, answer in zip(rows, labels[0], strict=True) if row.printed != row.scanned
        )
        keys = {key[:-1] for key in by_group}
        ceiling = sum(max(by_group[(*key, "A")], by_group[(*key, "B")]) for key in keys)
        if 2 * ceiling != len(disagreement_rows):
            raise ValueError(f"The {feature} shortcut is not exactly balanced.")
        ceilings[feature] = {"n": len(disagreement_rows), "both_correct": ceiling}
    return {
        "fixed_policies": fixed,
        "best_fixed": max(score["disagreement"]["both_correct"] for score in fixed.values()),
        "one_feature_oracle_ceilings": ceilings,
    }


def breakdown(rows, labels, scores):
    predictions = [select(side) for side in scores]
    metrics = pair_metrics(rows, *labels, *predictions)
    cells = {}
    for cell in sorted({row.cell for row in rows}):
        indices = [i for i, row in enumerate(rows) if row.cell == cell]
        cell_answers = [[side[i] for i in indices] for side in (*labels, *predictions)]
        cells[" / ".join(cell)] = pair_metrics(
            [rows[i] for i in indices], *cell_answers,
        )
    return {**metrics, "by_cell": cells}


def run(args, rows, labels, baselines):
    exp = Experiment(args)
    encoded = {name: exp.encode([row.prompt() for row in records]) for name, records in rows.items()}
    model = exp.model()
    frozen = {name: exp.score(model, encoded[name]) for name in TEST_SPLITS}
    del model
    gc.collect()
    exp.torch.cuda.empty_cache()
    report = {
        "runtime": exp.config(),
        "shortcuts": baselines,
        "frozen": {name: breakdown(rows[name], labels[name], [scores, scores])
                   for name, scores in frozen.items()},
        "training": {},
    }
    scores = {name: [] for name in TEST_SPLITS}
    for i, copy in enumerate(CHOICES):
        print(f"Training copy {copy} with ordinary LoRA.", flush=True)
        model, info = exp.train(copy, encoded["train"], labels["train"][i], encoded["dev"], labels["dev"][i])
        report["training"][copy] = info
        write_json(args.output / f"metrics-{copy.lower()}.json", info)
        for name in TEST_SPLITS:
            scores[name].append(exp.score(model, encoded[name]))
        del model
        gc.collect()
        exp.torch.cuda.empty_cache()
    if report["training"]["A"]["initial_adapter_sha256"] != report["training"]["B"]["initial_adapter_sha256"]:
        raise RuntimeError("The matched copies did not start with identical adapter weights.")
    report["trained"] = {name: breakdown(rows[name], labels[name], scores[name]) for name in TEST_SPLITS}
    write_json(args.output / "predictions.json", {
        name: {"record_ids": [row.id for row in rows[name]], "labels": labels[name],
               "frozen_scores": frozen[name], "trained_scores": scores[name]}
        for name in TEST_SPLITS
    })
    write_json(args.output / "results.json", report)
    for name, result in report["trained"].items():
        score = result["disagreement"]
        print(f"{name}: {score['both_correct']}/{score['n']} paired", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_arguments(parser)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--design-seed", type=int, default=81726)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    resolve_revision(parser, args)
    if min(args.epochs, args.batch_size) < 1 or not 0 < args.learning_rate < 1:
        parser.error("Training needs positive epochs and batch size, and a learning rate between zero and one.")
    books, reversals, seen, held_out, rows = make_design(args.design_seed)
    validate_design(books, reversals, seen, held_out, rows)
    labels = {name: [answers(records, book, reversals) for book in books] for name, records in rows.items()}
    baselines = {name: shortcuts(rows[name], labels[name]) for name in TEST_SPLITS}
    protocol = {
        "model": args.model_id, "revision": args.revision,
        "design_seed": args.design_seed, "training_seed": args.training_seed,
        "depot_rules": dict(zip(CHOICES, books, strict=True)), "class_reversals": reversals,
        "seen_cells": sorted(seen), "held_out_cells": sorted(held_out),
        "template": TEMPLATE, "choice_suffix": CHOICE_SUFFIX,
        "examples": {name: [asdict(row) for row in records] for name, records in rows.items()},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "lora_rank": 8, "lora_alpha": 16,
                     "target_modules": TARGET_MODULES},
        "primary": "Both copies correct on disagreement records in held-out combinations.",
        "follow_up": "Run the sparse budgets if ordinary LoRA beats both the fixed and one-feature shortcuts.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "shortcuts.json", baselines)
    print({name: len(records) for name, records in rows.items()}, flush=True)
    for name, baseline in baselines.items():
        print(f"{name} shortcut ceilings: {baseline['one_feature_oracle_ceilings']}", flush=True)
    if not args.prepare_only:
        run(args, rows, labels, baselines)


if __name__ == "__main__":
    main()
