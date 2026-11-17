"""Teach each parcel rule separately, then continue on the original seen combinations."""

import argparse
import gc
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from run_composition import (
    TEST_SPLITS, ComposedParcel, answers, breakdown, field, make_records, shortcuts, validate_design,
)
from run_matched_pair import CHOICES, CHOICE_SUFFIX, Experiment, select, write_json


COMPONENTS = ("depot", "class")
COMPOSITION_COUNTS = (16, 8)


@dataclass(frozen=True)
class RuleExample:
    component: str
    group: str
    prompt: str
    labels: tuple[str, str]


def component_examples(books, reversals, split, lanes, depot_count, class_count):
    rng = random.Random(f"composition-component:{split}")
    examples = []
    for component, groups, count in (
        ("depot", [(name, None) for name in books[0]], depot_count),
        ("class", [(name, start) for name in reversals for start in ("printed", "scanned")], class_count),
    ):
        profiles = rng.sample(lanes, 2 * count)
        for name, start in groups:
            if component == "depot":
                question = f"Depot: {name}\nWhich lane does this depot start with?"
                targets = [book[name] for book in books]
                group = name
            else:
                question = f"Package class: {name}\nStarting field: {start}\nWhich lane is used after applying the package-class rule?"
                targets = [field(start, reversals[name])] * 2
                group = f"{name} / {start}"
            for i in range(count):
                printed, scanned = profiles[2 * i:2 * (i + 1)]
                options = ("printed", "scanned") if i % 2 == 0 else ("scanned", "printed")
                values = {"printed": printed, "scanned": scanned}
                prompt = f"{question}\nPrinted lane: {printed}\nScanned lane: {scanned}"
                prompt += CHOICE_SUFFIX.format(option_a=values[options[0]], option_b=values[options[1]])
                labels = tuple(CHOICES[options.index(target)] for target in targets)
                examples.append(RuleExample(component, group, prompt, labels))
    rng.shuffle(examples)
    return examples


def component_metrics(rows, copy_index, scores):
    groups = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "n": 0}))
    for row, prediction in zip(rows, select(scores), strict=True):
        item = groups[row.component][row.group]
        item["n"] += 1
        item["correct"] += prediction == row.labels[copy_index]
    return {
        kind: {"correct": sum(g["correct"] for g in entries.values()),
               "n": sum(g["n"] for g in entries.values()), "groups": dict(entries)}
        for kind, entries in groups.items()
    }


def prepare(reference, epochs, component_epochs):
    raw = (reference / "protocol.json").read_bytes()
    source = json.loads(raw)
    books = [source["depot_rules"][name] for name in CHOICES]
    reversals = source["class_reversals"]
    seen = {tuple(cell) for cell in source["seen_cells"]}
    held_out = {tuple(cell) for cell in source["held_out_cells"]}
    composition = {key: [ComposedParcel(**row) for row in source["examples"][key]]
                   for key in ("train", "dev")}
    fresh_seed = f"{source['design_seed']}:component-composition-test"
    for name, cells in zip(TEST_SPLITS, (seen, held_out), strict=True):
        composition[name] = make_records(fresh_seed, "component-test", cells, range(700, 900),
                                         *COMPOSITION_COUNTS)
        random.Random(f"{fresh_seed}:{name}").shuffle(composition[name])
        earlier_lanes = {r[key] for r in source["examples"][name] for key in ("option_a", "option_b")}
        if any({r.option_a, r.option_b} & earlier_lanes for r in composition[name]):
            raise ValueError("The component study needs fresh test lane values.")
    validate_design(books, reversals, seen, held_out, composition)
    labels = {name: [answers(records, book, reversals) for book in books]
              for name, records in composition.items()}
    baselines = {name: shortcuts(composition[name], labels[name]) for name in TEST_SPLITS}
    components = {
        name: component_examples(books, reversals, name, lanes, depots, classes)
        for name, lanes, depots, classes in (
            ("train", range(100, 300), 24, 12),
            ("dev", range(300, 500), 8, 4),
            ("probe", range(900, 1000), 12, 8),
        )
    }
    protocol = {
        "source_run": str(reference),
        "source_protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "model": source["model"],
        "revision": source["revision"],
        "training_seed": source["training_seed"],
        "depot_rules": source["depot_rules"],
        "class_reversals": reversals,
        "seen_cells": source["seen_cells"],
        "held_out_cells": source["held_out_cells"],
        "composition_examples": {name: [asdict(row) for row in records]
                                 for name, records in composition.items()},
        "component_examples": {name: [asdict(row) for row in records]
                               for name, records in components.items()},
        "composition_template": source["template"],
        "choice_suffix": source["choice_suffix"],
        "training": {**source["training"], "epochs": epochs},
        "component_epochs": component_epochs,
        "primary": "Paired disagreement accuracy on held-out combinations after both training phases.",
        "component_probe": "Same untouched probe before and after combined training. A component passes at "
                           "90% overall for each copy and above 50% in every group.",
        "sparse_follow_up": "Run the sparse budgets if ordinary LoRA beats the best fixed and one-feature shortcuts.",
    }
    return protocol, components, composition, labels, baselines


def summarize(comp_rows, composition, labels, predictions):
    probes = {
        copy: component_metrics(comp_rows, i, predictions[copy]["component_probe"])
        for i, copy in enumerate(CHOICES)
    }
    for details in probes.values():
        for metric in details.values():
            metric["passed"] = (10 * metric["correct"] >= 9 * metric["n"]
                                and all(2 * g["correct"] > g["n"] for g in metric["groups"].values()))
    return {
        "components": probes,
        "composition": {
            name: breakdown(composition[name], labels[name], [predictions[copy][name] for copy in CHOICES])
            for name in TEST_SPLITS
        },
    }


def run(args, protocol, components, composition, labels, baselines):
    root = args.output
    args.model_id, args.revision = protocol["model"], protocol["revision"]
    args.training_seed = protocol["training_seed"]
    args.batch_size = protocol["training"]["batch_size"]
    args.learning_rate = protocol["training"]["learning_rate"]
    exp = Experiment(args)
    comp_tokens = {name: exp.encode([row.prompt for row in rows]) for name, rows in components.items()}
    combo_tokens = {name: exp.encode([row.prompt() for row in rows]) for name, rows in composition.items()}
    phases = ("after_components", "after_composition")
    predictions = {phase: {} for phase in phases}
    histories = {phase: {} for phase in phases}
    for phase in phases:
        (root / phase).mkdir()

    for i, copy in enumerate(CHOICES):
        args.output = root / phases[0]
        args.epochs = protocol["component_epochs"]
        print(f"Teaching individual rules to copy {copy}.", flush=True)
        model, info = exp.train(copy, comp_tokens["train"], [row.labels[i] for row in components["train"]],
                               comp_tokens["dev"], [row.labels[i] for row in components["dev"]])
        histories[phases[0]][copy] = info
        predictions[phases[0]][copy] = {
            "component_probe": exp.score(model, comp_tokens["probe"]),
            **{name: exp.score(model, combo_tokens[name]) for name in TEST_SPLITS},
        }
        probe = component_metrics(components["probe"], i, predictions[phases[0]][copy]["component_probe"])
        print(f"Copy {copy} separate probes:", {k: (v["correct"], v["n"]) for k, v in probe.items()},
              flush=True)

        args.output = root / phases[1]
        args.epochs = protocol["training"]["epochs"]
        print(f"Continuing copy {copy} on the eight seen combinations.", flush=True)
        model, continued = exp.train(copy, combo_tokens["train"], labels["train"][i],
                                    combo_tokens["dev"], labels["dev"][i], model=model)
        if continued["initial_adapter_sha256"] != info["final_adapter_sha256"]:
            raise RuntimeError("Composition training did not continue the learned adapter.")
        histories[phases[1]][copy] = continued
        predictions[phases[1]][copy] = {
            "component_probe": exp.score(model, comp_tokens["probe"]),
            **{name: exp.score(model, combo_tokens[name]) for name in TEST_SPLITS},
        }
        write_json(root / "predictions.json", predictions)
        write_json(root / "training.json", histories)
        del model
        gc.collect()
        exp.torch.cuda.empty_cache()

    if histories[phases[0]]["A"]["initial_adapter_sha256"] != histories[phases[0]]["B"]["initial_adapter_sha256"]:
        raise RuntimeError("The matched copies did not start with identical adapter weights.")
    report = {"runtime": exp.config(), "shortcuts": baselines, "phases": {}}
    for phase, output in predictions.items():
        stage = summarize(components["probe"], composition, labels, output)
        report["phases"][phase] = stage
        print(phase, {name: m["disagreement"]["both_correct"] for name, m in stage["composition"].items()},
              {copy: {k: (v["correct"], v["n"]) for k, v in result.items()}
               for copy, result in stage["components"].items()}, flush=True)
    write_json(root / "results.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--component-epochs", default=6, type=int)
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.component_epochs) < 1:
        parser.error("Both training phases require a positive epoch count.")
    protocol, components, composition, labels, baselines = prepare(
        args.reference, args.epochs, args.component_epochs,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "shortcuts.json", baselines)
    if not args.prepare_only:
        run(args, protocol, components, composition, labels, baselines)


if __name__ == "__main__":
    main()
