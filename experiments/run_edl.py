"""Measure first-exposure codelength after rulebook or format-only practice."""

import argparse
import gc
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

from run_budget_confirmation import format_only_labels
from run_matched_pair import (
    CHOICES, MODEL_ID, REVISION, TARGET_MODULES, Experiment, Parcel, build_split, gold, pair_metrics,
    select, shortcut_metrics, write_json,
)


SIZES = (64, 192, 384)


def balanced_subsets(rows, seed, sizes):
    groups = defaultdict(list)
    for row in rows:
        groups[row.depot, row.template, row.printed != row.scanned, row.answer("printed")].append(row.id)
    rng = random.Random(f"{seed}:EDL:subsets")
    for group in groups.values():
        rng.shuffle(group)

    subsets = {}
    for size in sizes:
        ids = set()
        for group in groups.values():
            count, remainder = divmod(len(group) * size, len(rows))
            if remainder:
                raise ValueError("EDL sizes must preserve the feature, template, and answer balance.")
            ids.update(group[:count])
        subsets[size] = [row for row in rows if row.id in ids]
        if len(subsets[size]) != size:
            raise ValueError("An EDL subset has the wrong record count.")
    return subsets


def build_edl_inputs(draw, seed, sizes):
    exposure = {k: [Parcel(**row) for row in draw["examples"][k]] for k in ("train", "dev")}
    exposure_labels = {k: format_only_labels(rows, rule_seed=f"{seed}:{k}", separate_templates=True)
                       for k, rows in exposure.items()}
    depots = draw["rulebooks"][0]
    adaptation = {
        "train": build_split(seed, "edl-train", range(500, 700), depots, 72, 24),
        "dev": build_split(seed, "edl-dev", range(700, 800), depots, 16, 8),
        "test": build_split(seed, "edl-test", range(800, 900), depots, 72, 24),
    }
    exposure_lanes = {lane for rows in exposure.values() for r in rows for lane in (r.option_a, r.option_b)}
    for rows in adaptation.values():
        if any({r.option_a, r.option_b} & exposure_lanes for r in rows):
            raise ValueError("The EDL parcels must use fresh lanes relative to both exposure splits.")
    subsets = balanced_subsets(adaptation["train"], seed, sizes)
    inputs = {
        "format_only_labels": exposure_labels,
        "adaptation_examples": {k: [asdict(row) for row in rows] for k, rows in adaptation.items()},
        "nested_train_record_ids": {str(n): [row.id for row in rows] for n, rows in subsets.items()},
    }
    return inputs, exposure, adaptation, subsets


def prepare(args):
    wording_bytes = (args.wording_run / "protocol.json").read_bytes()
    wording = json.loads(wording_bytes)
    if (wording["model"], wording["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("The source data must use the pinned Qwen2.5 checkpoint.")
    if args.prior_run is not None:
        prior_dir = args.prior_run
        prior_bytes = (prior_dir / "protocol.json").read_bytes()
        prior = json.loads(prior_bytes)
        seed = str(prior["rule_seed"])
        draw = wording["draws"][seed]
        settings = prior["training"]
        if ((prior["model"], prior["revision"]) != (MODEL_ID, REVISION)
                or [prior["rulebooks"][letter] for letter in "ab"] != draw["rulebooks"]
                or any(prior["examples"][k] != draw["examples"][k] for k in ("train", "dev"))
                or prior["training_seed"] != wording["training_seed"]):
            raise ValueError("Prior exposure must match the checkpoint, source records, rulebooks, and training seed.")
        prior_metadata = {
            copy: json.loads((prior_dir / f"metrics-{copy.lower()}.json").read_text(encoding="utf-8"))
            for copy in CHOICES
        }
    else:
        seed = str(args.wording_seed)
        draw = wording["draws"][seed]
        prior_dir = args.wording_run / f"rulebook-{seed}" / "two-templates"
        original_forms = [row["template"] for row in draw["examples"]["train"]]
        if draw["training_template_indices"]["two-templates"] != original_forms:
            raise ValueError("The saved wording adapters must have used the original record templates.")
        prior_bytes = wording_bytes
        settings = wording["settings"]
        prior_metadata = json.loads((prior_dir / "results.json").read_text(encoding="utf-8"))["training"]
    if (settings["lora_rank"], settings["lora_alpha"], tuple(settings["target_modules"])) != (8, 16, TARGET_MODULES):
        raise ValueError("Prior exposure must use the matched Qwen2.5 checkpoint, data, rules, and LoRA architecture.")
    books = draw["rulebooks"]
    inputs, exposure, adaptation, subsets = build_edl_inputs(draw, seed, args.sizes)
    frozen_inputs = getattr(args, "frozen_inputs", None)
    if frozen_inputs is not None:
        frozen = json.loads(frozen_inputs.read_text(encoding="utf-8"))
        if inputs != frozen[seed]:
            raise ValueError("EDL records or format labels differ from the frozen inputs.")
    protocol = {
        "method": "Donoway et al., Bits That Count, sections 2.2–2.4",
        "source": "https://proceedings.mlr.press/v306/donoway26a.html",
        "model": MODEL_ID, "revision": REVISION, "rule_seed": seed, "rulebooks": books,
        "training_seed": wording["training_seed"], "training": settings,
        "wording_protocol_sha256": hashlib.sha256(wording_bytes).hexdigest(),
        "prior_protocol_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        "prior_run": str(prior_dir),
        "prior_adapters": {copy: m["final_adapter_sha256"] for copy, m in prior_metadata.items()},
        **inputs,
        "sizes": args.sizes,
        "loss": "Single next-token label negative log-probability over the full vocabulary.",
        "edl": "Sum of first-epoch pre-update label losses, minus the number of train labels times "
               "mean final held-out label loss, converted to bits.",
    }
    if args.opposite_only or args.include_opposite:
        protocol["opposite_prior_adapter_by_target"] = dict(zip(CHOICES, reversed(CHOICES), strict=True))
    if args.adaptation_learning_rate is not None:
        protocol["adaptation_learning_rate"] = args.adaptation_learning_rate
    if args.reuse_format_run is not None:
        reference_bytes = (args.reuse_format_run / "protocol.json").read_bytes()
        reference = json.loads(reference_bytes)
        match = ("model", "revision", "rule_seed", "rulebooks", "training_seed", "training",
                 "prior_adapters", "format_only_labels", "adaptation_examples")
        if any(reference[k] != protocol[k] for k in match):
            raise ValueError("The saved format exposure must match the checkpoint, rulebooks, and parcels.")
        for n, ids in protocol["nested_train_record_ids"].items():
            if reference["nested_train_record_ids"].get(n) != ids:
                raise ValueError("The learning-rate comparison must use the same adaptation subsets.")
        format_dir = args.reuse_format_run / "format-exposure"
        hashes = {}
        for copy in CHOICES:
            training = json.loads((format_dir / f"metrics-{copy.lower()}.json").read_text(encoding="utf-8"))
            if training["initial_adapter_sha256"] != prior_metadata[copy]["initial_adapter_sha256"]:
                raise ValueError("The saved exposure conditions must share the original adapter initialization.")
            hashes[copy] = training["final_adapter_sha256"]
        protocol["format_exposure_reference"] = {
            "path": str(format_dir),
            "protocol_sha256": hashlib.sha256(reference_bytes).hexdigest(),
            "adapter_hashes": hashes,
        }
    return protocol, exposure, adaptation, subsets, prior_metadata


def score_with_loss(exp, model, encoded, targets):
    torch = exp.torch
    model.eval()
    all_scores, all_losses = [], []
    with torch.inference_mode():
        for i in range(0, len(encoded), exp.args.batch_size):
            batch = encoded[i:i + exp.args.batch_size]
            choices = targets[i:i + len(batch)]
            logits = exp.logits(model, batch)
            expected = torch.tensor([exp.answer_token_ids[CHOICES.index(y)] for y in choices], device=exp.device)
            losses = torch.nn.functional.cross_entropy(logits, expected, reduction="none")
            all_losses.extend(losses.cpu().tolist())
            all_scores.extend(logits[:, exp.answer_token_ids].cpu().tolist())
    if len(all_scores) != len(targets) or any(not math.isfinite(loss) for loss in all_losses):
        raise RuntimeError("Expected a finite held-out codelength for each target label.")
    return {"choice_logits": all_scores, "label_nll_nats": all_losses}


def code_metrics(first_nll, final_nll):
    mdl = math.fsum(first_nll) / math.log(2)
    residual = fmean(final_nll) / math.log(2)
    edl = mdl - len(first_nll) * residual
    return {
        "labels": len(first_nll), "prequential_bits": mdl,
        "first_pass_bits_per_label": mdl / len(first_nll),
        "final_bits_per_label": residual, "edl_bits": edl, "edl_bits_per_label": edl / len(first_nll),
    }


def run(args, protocol, exposure, adaptation, subsets, prior_metadata):
    root = args.output
    prior_dir = Path(protocol["prior_run"])
    settings = protocol["training"]
    config = SimpleNamespace(checkpoint=args.checkpoint, model_id=MODEL_ID, revision=REVISION,
                             training_seed=protocol["training_seed"], epochs=settings["epochs"],
                             batch_size=settings["batch_size"], learning_rate=settings["learning_rate"])
    exp = Experiment(config)
    enc = lambda rows: exp.encode([row.prompt() for row in rows])
    held_out = adaptation["test"]
    test_x = enc(held_out)
    dev_x = enc(adaptation["dev"])
    books = protocol["rulebooks"]
    held_targets = [gold(held_out, book) for book in books]
    shortcuts = shortcut_metrics(held_out, *held_targets)
    fixed = max(x["disagreement"]["both_correct"] for x in shortcuts.values())
    report = {"runtime": exp.config(), "best_fixed_shortcut": fixed, "exposure_fit_count": 0,
              "adaptation_fit_count": 0, "starting_models": {}, "sizes": {}}
    identity = {copy: copy for copy in CHOICES}
    sources = {}
    if not args.opposite_only:
        sources["prior-rulebook"] = (prior_dir, identity, protocol["prior_adapters"])
        saved_format = protocol.get("format_exposure_reference")
        if saved_format:
            format_dir = Path(saved_format["path"])
            format_hashes = saved_format["adapter_hashes"]
        else:
            exposure_enc = {split: enc(records) for split, records in exposure.items()}
            format_dir = root / "format-exposure"
            format_dir.mkdir()
            config.output = format_dir
            format_hashes = {}
            for copy in CHOICES:
                print(f"Preparing format-only copy {copy}.", flush=True)
                targets = protocol["format_only_labels"]
                model, training = exp.train(copy, exposure_enc["train"], targets["train"][copy]["labels"],
                                            exposure_enc["dev"], targets["dev"][copy]["labels"])
                if training["initial_adapter_sha256"] != prior_metadata[copy]["initial_adapter_sha256"]:
                    raise RuntimeError("Prior-rulebook and format-only controls must share an adapter initialization.")
                format_hashes[copy] = training["final_adapter_sha256"]
                write_json(format_dir / f"metrics-{copy.lower()}.json", training)
                report["exposure_fit_count"] += 1
                del model
                gc.collect()
                exp.torch.cuda.empty_cache()
        sources["format-only"] = (format_dir, identity, format_hashes)
    if args.opposite_only or args.include_opposite:
        sources["opposite-rulebook"] = (prior_dir, protocol["opposite_prior_adapter_by_target"],
                                        protocol["prior_adapters"])

    scored_start = {}
    for condition, (parent, source_copy, _) in sources.items():
        reads = []
        for i, copy in enumerate(CHOICES):
            model = exp.peft.PeftModel.from_pretrained(exp.model(), parent / f"adapter-{source_copy[copy].lower()}")
            reads.append(score_with_loss(exp, model, test_x, held_targets[i]))
            del model
            gc.collect()
            exp.torch.cuda.empty_cache()
        scored_start[condition] = reads
        report["starting_models"][condition] = pair_metrics(
            held_out, *held_targets, *[select(item["choice_logits"]) for item in reads],
        )
    write_json(root / "starting_predictions.json", scored_start)
    write_json(root / "results.json", report)
    print("Before adaptation, paired on disagreement:",
          {k: v["disagreement"]["both_correct"] for k, v in report["starting_models"].items()}, flush=True)

    config.learning_rate = protocol.get("adaptation_learning_rate", settings["learning_rate"])
    for n, records in subsets.items():
        train_x = enc(records)
        report["sizes"][n] = {}
        for condition, (parent, source_copy, hashes) in sources.items():
            config.output = root / f"n-{n}" / condition
            config.output.mkdir(parents=True)
            predictions, per_copy = [], {}
            for i, copy in enumerate(CHOICES):
                print(f"EDL size {n}, {condition}, copy {copy}.", flush=True)
                filename = f"adapter-{source_copy[copy].lower()}"
                model = exp.peft.PeftModel.from_pretrained(exp.model(), parent / filename,
                                                          is_trainable=True)
                model, training = exp.train(copy, train_x, gold(records, books[i]),
                                            dev_x, gold(adaptation["dev"], books[i]),
                                            model=model, record_first_exposure=True)
                if training["initial_adapter_sha256"] != hashes[source_copy[copy]]:
                    raise RuntimeError("An EDL fit started from the wrong exposure adapter.")
                first = training["first_exposure_nll_nats"]
                if sorted(index for index, _ in first) != list(range(n)):
                    raise RuntimeError("Each target label must be scored once on first exposure.")
                scored = score_with_loss(exp, model, test_x, held_targets[i])
                predictions.append(scored)
                readout = {"all": code_metrics([loss for _, loss in first], scored["label_nll_nats"])}
                for stratum, differs in (("disagreement", True), ("agreement", False)):
                    first_nll = [loss for index, loss in first
                                 if (records[index].printed != records[index].scanned) == differs]
                    final_nll = [loss for row, loss in zip(held_out, scored["label_nll_nats"], strict=True)
                                 if (row.printed != row.scanned) == differs]
                    readout[stratum] = code_metrics(first_nll, final_nll)
                per_copy[copy] = readout
                write_json(config.output / f"metrics-{copy.lower()}.json", {
                    "training": training, "held_out": scored, "code": readout,
                })
                report["adaptation_fit_count"] += 1
                del model
                gc.collect()
                exp.torch.cuda.empty_cache()
            paired = pair_metrics(held_out, *held_targets, *[select(p["choice_logits"]) for p in predictions])
            report["sizes"][n][condition] = {
                "copy_metrics": per_copy, "paired_accuracy": paired,
                "mean_edl_bits_per_label": {k: fmean(item[k]["edl_bits_per_label"] for item in per_copy.values())
                                            for k in ("all", "disagreement", "agreement")},
                "mean_final_bits_per_label": fmean(item["all"]["final_bits_per_label"]
                                                   for item in per_copy.values()),
            }
            write_json(root / "results.json", report)
            print(f"EDL size {n}, {condition}:",
                  report["sizes"][n][condition]["mean_edl_bits_per_label"],
                  "paired", paired["disagreement"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wording-run", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prior-run", type=Path)
    source.add_argument("--wording-seed", type=int, help="Reuse the saved two-template adapters for this rulebook pair.")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    exposures = parser.add_mutually_exclusive_group()
    exposures.add_argument("--opposite-only", action="store_true",
                           help="Start from the adapter trained on the opposite rulebook.")
    exposures.add_argument("--include-opposite", action="store_true",
                           help="Include the opposite-rulebook adapter as a third starting condition.")
    parser.add_argument("--reuse-format-run", type=Path,
                        help="Reuse verified format-only adapters from an existing matched EDL run.")
    parser.add_argument("--adaptation-learning-rate", type=float,
                        help="Override the optimizer rate for the adaptation fits only.")
    parser.add_argument("--frozen-inputs", type=Path,
                        help="Check generated adaptation records and format labels against a frozen manifest.")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if list(args.sizes) != sorted(set(args.sizes)) or any(not 1 <= n <= max(SIZES) for n in args.sizes):
        parser.error("Training sizes must be distinct, increasing, and between 1 and 384.")
    if args.reuse_format_run and args.opposite_only:
        parser.error("Reusing format-only adapters requires a format-only condition.")
    if args.adaptation_learning_rate is not None and not 0 < args.adaptation_learning_rate < 1:
        parser.error("The adaptation learning rate must be between zero and one.")
    prepared = prepare(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", prepared[0])
    if not args.prepare_only:
        run(args, *prepared)


if __name__ == "__main__":
    main()
