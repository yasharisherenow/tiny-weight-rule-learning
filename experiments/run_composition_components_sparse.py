"""Train the component curriculum with 1,000 and 2,000 trainable LoRA scalars."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

from safetensors.torch import save_file

from run_composition import TEST_SPLITS, ComposedParcel, answers
from run_composition_components import RuleExample, summarize
from run_matched_pair import CHOICES, Experiment, write_json
from run_scalar_sweep import ALPHA, RANK, apply_mask, eligible_modules, evaluate_epoch, sample_masks


BUDGETS = (1000, 2000)
EPOCHS = (6, 12, 18)
MASK_SEED = 122
BASIS_SEED = 119


def capture_scores(exp, model, parameters, state, comp, combo):
    with exp.torch.no_grad():
        for name, values in state.items():
            parameters[name].copy_(values)
    return {"component_probe": exp.score(model, comp["probe"]),
            **{name: exp.score(model, combo[name]) for name in TEST_SPLITS}}


def train_budget(exp, model, parameters, component_rows, composition, labels, comp, combo,
                 component_epochs):
    predictions = {f"epoch_{epoch}": {} for epoch in EPOCHS}
    predictions["after_components"] = {}
    histories = {}
    final_values = {}
    changed = []
    for i, copy in enumerate(CHOICES):
        exp.args.epochs = component_epochs
        print(f"Teaching individual rules to copy {copy}.", flush=True)
        primitive = evaluate_epoch(exp, model, parameters, comp["train"],
                                   [row.labels[i] for row in component_rows["train"]],
                                   comp["dev"], exp.args.training_seed, exp.args.learning_rate)
        learned = primitive["states"][-1]
        predictions["after_components"][copy] = capture_scores(
            exp, model, parameters, learned, comp, combo,
        )
        exp.args.epochs = EPOCHS[-1]
        print(f"Continuing copy {copy} on seen combinations.", flush=True)
        combined = evaluate_epoch(exp, model, parameters, combo["train"], labels["train"][i],
                                  combo["dev"], exp.args.training_seed, exp.args.learning_rate,
                                  initial_state=learned)
        for epoch in EPOCHS:
            predictions[f"epoch_{epoch}"][copy] = capture_scores(
                exp, model, parameters, combined["states"][epoch - 1], comp, combo,
            )
        final = combined["states"][-1]
        changed.append(sum(int(value.count_nonzero()) for value in final.values()))
        final_values.update({f"{copy}.{name}": value for name, value in final.items()})
        histories[copy] = {
            phase: {k: fit[k] for k in ("train_loss", "seconds_train_and_dev", "peak_allocated_bytes")}
            for phase, fit in (("components", primitive), ("composition", combined))
        }
    summaries = {phase: summarize(component_rows["probe"], composition, labels, output)
                 for phase, output in predictions.items()}
    return {"phases": summaries, "predictions": predictions, "training": histories,
            "changed_parameters": changed}, final_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary-run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_bytes = (args.ordinary_run / "protocol.json").read_bytes()
    protocol = json.loads(source_bytes)
    reference = json.loads((args.ordinary_run / "results.json").read_text(encoding="utf-8"))
    held = reference["phases"]["after_composition"]["composition"]["test_held_out"]["disagreement"]
    shortcut = reference["shortcuts"]["test_held_out"]
    ceiling = max(shortcut["best_fixed"],
                  *(m["both_correct"] for m in shortcut["one_feature_oracle_ceilings"].values()))
    if held["both_correct"] <= ceiling:
        parser.error("The ordinary pair must beat the held-out shortcut before running sparse budgets.")
    component_rows = {name: [RuleExample(**row) for row in rows]
                      for name, rows in protocol["component_examples"].items()}
    composition = {name: [ComposedParcel(**row) for row in rows]
                   for name, rows in protocol["composition_examples"].items()}
    books = [protocol["depot_rules"][copy] for copy in CHOICES]
    labels = {name: [answers(rows, book, protocol["class_reversals"]) for book in books]
              for name, rows in composition.items()}
    args.model_id, args.revision = protocol["model"], protocol["revision"]
    args.training_seed = protocol["training_seed"]
    args.batch_size, args.learning_rate = 16, 0.01
    exp = Experiment(args)
    comp = {name: exp.encode([row.prompt for row in rows]) for name, rows in component_rows.items()}
    combo = {name: exp.encode([row.prompt() for row in rows]) for name, rows in composition.items()}
    model = exp.model().requires_grad_(False)
    eligible, masks = sample_masks(eligible_modules(model), BUDGETS, MASK_SEED)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {
        "source_run": str(args.ordinary_run),
        "source_protocol_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model": args.model_id, "revision": args.revision, "runtime": exp.config(),
        "training_seed": args.training_seed, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "component_epochs": protocol["component_epochs"], "fixed_composition_epochs": EPOCHS,
        "rank": RANK, "alpha": ALPHA, "mask_seed": MASK_SEED, "basis_seed": BASIS_SEED,
        "eligible_B_coordinates": eligible, "masks": masks,
    })
    summaries = {"ordinary": reference["phases"], "shortcuts": reference["shortcuts"], "budgets": {}}
    for budget, mask in masks.items():
        print(f"Training the curriculum with {budget} trainable scalars.", flush=True)
        with apply_mask(model, mask, BASIS_SEED, budget) as parameters:
            results, values = train_budget(
                exp, model, parameters, component_rows, composition, labels, comp, combo,
                protocol["component_epochs"],
            )
        del parameters
        gc.collect()
        exp.torch.cuda.empty_cache()
        file = f"budget-{budget}-values.safetensors"
        save_file(values, args.output / file)
        results["learned_values_file"] = file
        write_json(args.output / f"budget-{budget}.json", results)
        summaries["budgets"][budget] = {k: results[k] for k in ("phases", "changed_parameters")}
        write_json(args.output / "results.json", summaries)
        print(f"{budget} paired disagreement scores:",
              {phase: {name: stage["composition"][name]["disagreement"]["both_correct"]
                       for name in TEST_SPLITS} for phase, stage in results["phases"].items()}, flush=True)


if __name__ == "__main__":
    main()
