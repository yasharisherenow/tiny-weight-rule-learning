"""Compare two versus eight training templates on matched rulebook pairs."""

import argparse
import gc
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from run_matched_pair import (
    CHOICES, CHOICE_SUFFIX, MODEL_ID, REVISION, TARGET_MODULES, TEMPLATES,
    Experiment, build_data, check_data, gold, pair_metrics, select,
    shortcut_metrics, write_json,
)


EXTRA_TRAIN = (
    "At {depot}, the scanner recorded lane {scanned}. The printed sticker reads {printed}.\nPick the routing lane.",
    "Parcel routing ticket\nFacility = {depot}\nScanner = {scanned}\nPaper label = {printed}\nChoose one lane.",
    "The shipment arrived at {depot} with a paper lane of {printed} and a scanned lane of {scanned}. Which candidate is its destination?",
    "Route this parcel. Site: {depot}. Its printed value is {printed}; its scan value is {scanned}.",
    "Sorting site | scan | printed\n{depot} | {scanned} | {printed}\nSelect the correct lane for this row.",
    "I have a parcel from {depot}. I read {printed} off the printed tag and {scanned} from the scanner. Which lane is right?",
)

PRIMARY_TEST = {
    "email": "To the routing clerk: Please send the shipment waiting at {depot} to its assigned lane. "
             "Its label is marked {printed}; the barcode reader reported {scanned}. Which candidate should I use?",
    "answer-before-record": "Which of the two candidate lanes should receive this package?\n"
                            "Scanned destination: {scanned}\nProcessing depot: {depot}\n"
                            "Destination on the printed label: {printed}",
    "inspection-notes": "The scan at inspection was logged as {scanned}. The receiving depot is {depot}. "
                        "Before scanning, staff copied {printed} from the paper label. Select the destination.",
}
ALL_TEST = {"familiar": TEMPLATES[0], "original-unseen": TEMPLATES[2], **PRIMARY_TEST}


def render(row, template):
    return (template + CHOICE_SUFFIX).format(**asdict(row))


def varied_train_forms(rows, seed):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.depot, row.printed != row.scanned, row.answer("printed")].append(i)
    rng = random.Random(f"{seed}:varied-wording")
    choices = [None] * len(rows)
    for indices in groups.values():
        rng.shuffle(indices)
        for j, i in enumerate(indices):
            choices[i] = j % (2 + len(EXTRA_TRAIN))
    return choices


def make_draw(seed):
    rule_a, rule_b, data = build_data(seed)
    # The earlier evaluation used codes 500-699. Keep the new evaluation disjoint.
    data["test"] = [
        replace(row, id=f"fresh-{row.id}", printed=row.printed + 200, scanned=row.scanned + 200,
                option_a=row.option_a + 200, option_b=row.option_b + 200)
        for row in data["test"]
    ]
    check_data(data, rule_a, rule_b)
    return rule_a, rule_b, data


def validate_reference(protocol, reference_dir):
    if protocol["model"] != MODEL_ID or protocol["revision"] != REVISION:
        raise ValueError("The reference pair must use the same pinned Qwen checkpoint.")
    settings = protocol["training"]
    if (settings["lora_rank"], settings["lora_alpha"], tuple(settings["target_modules"])) != (8, 16, TARGET_MODULES):
        raise ValueError("The reference pair must use the same LoRA architecture.")
    if tuple(protocol["templates"]) != TEMPLATES or protocol["choice_suffix"] != CHOICE_SUFFIX:
        raise ValueError("The reference pair must use the original prompt templates.")
    if not all((reference_dir / f"adapter-{s}" / "adapter_model.safetensors").is_file() for s in "ab"):
        raise ValueError("Both reference adapter files are required.")


def evaluate_arm(exp, arm_dir, rule_a, rule_b, data, forms, reuse):
    train_templates = TEMPLATES[:2] + (EXTRA_TRAIN if max(forms) > 1 else ())
    train = exp.encode([render(r, train_templates[f]) for r, f in zip(data["train"], forms, strict=True)])
    dev = exp.encode([r.prompt() for r in data["dev"]])
    test = data["test"]
    evaluation = {
        form: exp.encode([render(r, template) for r in test])
        for form, template in ALL_TEST.items()
    }
    labels = [gold(test, book) for book in (rule_a, rule_b)]
    scores = {form: [] for form in evaluation}
    train_info = {}
    for label, book in zip(CHOICES, (rule_a, rule_b), strict=True):
        print(f"{arm_dir.parent.name} / {arm_dir.name} / copy {label}", flush=True)
        if reuse is None:
            model, info = exp.train(label, train, gold(data["train"], book),
                                    dev, gold(data["dev"], book))
        else:
            model = exp.peft.PeftModel.from_pretrained(exp.model(), reuse / f"adapter-{label.lower()}")
            info = json.loads((reuse / f"metrics-{label.lower()}.json").read_text(encoding="utf-8"))
        train_info[label] = info
        for form, tokens in evaluation.items():
            scores[form].append(exp.score(model, tokens))
        del model
        gc.collect()
        exp.torch.cuda.empty_cache()

    if train_info["A"]["initial_adapter_sha256"] != train_info["B"]["initial_adapter_sha256"]:
        raise RuntimeError("The two copies did not share an adapter initialization.")
    metrics = {form: pair_metrics(test, *labels, *[select(s) for s in pair])
               for form, pair in scores.items()}
    write_json(arm_dir / "predictions.json", {
        "record_ids": [r.id for r in test],
        "expected_a": labels[0],
        "expected_b": labels[1],
        "scores": scores,
    })
    write_json(arm_dir / "results.json", {"training": train_info, "per_template": metrics})
    print("Paired correct on disagreement records:",
          {k: v["disagreement"]["both_correct"] for k, v in metrics.items()}, flush=True)
    return train_info, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate-seed", type=int, default=81719)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((args.reference_run / "protocol.json").read_text(encoding="utf-8"))
    validate_reference(protocol, args.reference_run)
    reference_seed = protocol["rule_seed"]
    seeds = [reference_seed, args.replicate_seed]
    draws = {s: make_draw(s) for s in seeds}
    if len(draws) != 2 or draws[seeds[1]][0] in draws[seeds[0]][:2]:
        parser.error("The replication seed must draw a different rulebook pair, up to A/B swaps.")
    reference_data = draws[reference_seed][2]
    for split in ("train", "dev"):
        if [asdict(r) for r in reference_data[split]] != protocol["examples"][split]:
            raise ValueError(f"The regenerated reference {split} split differs from the saved run.")

    plans = {}
    for seed, (rule_a, rule_b, data) in draws.items():
        forms = {
            "two-templates": [r.template for r in data["train"]],
            "eight-templates": varied_train_forms(data["train"], seed),
        }
        plans[seed] = {
            "rulebooks": [rule_a, rule_b],
            "examples": {name: [asdict(r) for r in rows] for name, rows in data.items()},
            "training_template_indices": forms,
        }
    args.output.mkdir(parents=True, exist_ok=False)
    times = protocol["training"]
    write_json(args.output / "protocol.json", {
        "model": MODEL_ID, "revision": REVISION, "settings": times,
        "training_seed": protocol["training_seed"],
        "reused_reference_pair": args.reference_run.name,
        "training_templates": TEMPLATES[:2] + EXTRA_TRAIN,
        "test_templates": ALL_TEST, "choice_suffix": CHOICE_SUFFIX,
        "primary_templates": list(PRIMARY_TEST), "draws": plans,
        "primary": "Paired correctness on disagreements, reported separately for each fresh template.",
    })
    print("Verified two distinct rulebook assignments and disjoint test lane IDs.", flush=True)
    if args.prepare_only:
        return

    results = {}
    for seed, (rule_a, rule_b, data) in draws.items():
        test = data["test"]
        shortcuts = shortcut_metrics(test, gold(test, rule_a), gold(test, rule_b))
        strongest = max(m["disagreement"]["both_correct"] for m in shortcuts.values())
        summary = {"rulebooks": [rule_a, rule_b], "best_shortcut_disagreement_count": strongest}
        for arm, forms in plans[seed]["training_template_indices"].items():
            directory = args.output / f"rulebook-{seed}" / arm
            directory.mkdir(parents=True)
            trainer_settings = SimpleNamespace(
                checkpoint=args.checkpoint, output=directory,
                training_seed=protocol["training_seed"], epochs=times["epochs"],
                batch_size=times["batch_size"], learning_rate=times["learning_rate"],
            )
            exp = Experiment(trainer_settings)
            reuse = args.reference_run if seed == reference_seed and arm == "two-templates" else None
            train_info, metrics = evaluate_arm(exp, directory, rule_a, rule_b, data, forms, reuse)
            summary[arm] = {
                "train_examples_by_template": dict(Counter(forms)),
                "initial_adapter_sha256": train_info["A"]["initial_adapter_sha256"],
                "per_template": metrics,
            }
        if summary["two-templates"]["initial_adapter_sha256"] != summary["eight-templates"]["initial_adapter_sha256"]:
            raise RuntimeError("The wording arms started with different adapters.")
        results[seed] = summary
        write_json(args.output / "results.json", results)


if __name__ == "__main__":
    main()
