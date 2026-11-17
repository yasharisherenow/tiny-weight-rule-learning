"""Train exact scalar budgets in frozen-A, sparse-B LoRA adapters."""

import argparse
import gc
import hashlib
import json
import math
import random
import time
from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from run_matched_pair import (
    CHOICES, CHOICE_SUFFIX, MODEL_ID, REVISION, TARGET_MODULES, TEMPLATES, Experiment, Parcel, check_data,
    checkpoint_arguments, gold, pair_metrics, resolve_revision, select,
    shortcut_metrics, write_json,
)


RANK = 8
ALPHA = 16
DEFAULT_BUDGETS = (10, 100, 1000)


class SparseLoRAOutput(nn.Module):
    def __init__(self, base, coordinates, basis_seed):
        super().__init__()
        self.base = base
        self.register_buffer("coordinates", None if coordinates is None else
                             torch.tensor(coordinates, device=base.weight.device))
        generator = torch.Generator(device="cpu").manual_seed(basis_seed)
        basis = torch.empty(RANK, base.in_features, dtype=torch.float32)
        bound = 1 / math.sqrt(base.in_features)
        basis.uniform_(-bound, bound, generator=generator)
        self.register_buffer("basis", basis.to(base.weight.device))
        count = base.out_features * RANK if coordinates is None else len(coordinates)
        self.values = nn.Parameter(torch.zeros(count, device=base.weight.device))

    def forward(self, x):
        output = self.base(x)
        matrix = self.values
        if self.coordinates is not None:
            matrix = self.values.new_zeros(self.base.out_features * RANK).scatter(0, self.coordinates, self.values)
        matrix = matrix.view(self.base.out_features, RANK)
        update = F.linear(F.linear(x.to(self.basis.dtype), self.basis), matrix)
        return output + (update * (ALPHA / RANK)).to(output.dtype)


def eligible_modules(model):
    return sorted((name, layer.out_features * RANK) for name, layer in model.named_modules()
                  if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_MODULES)


def sample_masks(modules, budgets, mask_seed):
    if not modules:
        raise RuntimeError("No eligible linear projections were found.")
    boundaries = [0, *accumulate(n for _, n in modules)]
    picks = random.Random(mask_seed).sample(range(boundaries[-1]), max(budgets))
    masks = {}
    for budget in budgets:
        by_module = defaultdict(list)
        for pick in picks[:budget]:
            index = bisect_right(boundaries, pick) - 1
            by_module[modules[index][0]].append(pick - boundaries[index])
        masks[budget] = {name: sorted(indices) for name, indices in sorted(by_module.items())}
    return boundaries[-1], masks


def fixed_basis_seed(basis_seed, module):
    digest = hashlib.sha256(f"{basis_seed}:{module}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


@contextmanager
def apply_mask(model, mask, basis_seed, budget):
    mounted = {}
    try:
        for name, coordinates in mask.items():
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            base = getattr(parent, leaf)
            layer = SparseLoRAOutput(base, coordinates, fixed_basis_seed(basis_seed, name))
            setattr(parent, leaf, layer)
            mounted[name] = (parent, leaf, layer)
        parameters = {name: layer.values for name, (_, _, layer) in mounted.items()}
        expected = {id(p) for p in parameters.values()}
        found = {id(p) for p in model.parameters() if p.requires_grad}
        if expected != found or sum(p.numel() for p in parameters.values()) != budget:
            raise RuntimeError("Parameters outside the sampled mask are trainable.")
        yield parameters
    finally:
        for parent, leaf, layer in mounted.values():
            setattr(parent, leaf, layer.base)


def evaluate_epoch(exp, model, parameters, x, y, validation, training_seed, learning_rate,
                   *, initial_state=None):
    with torch.no_grad():
        if initial_state is None:
            for p in parameters.values():
                p.zero_()
        else:
            for name, p in parameters.items():
                p.copy_(initial_state[name])
    torch.manual_seed(training_seed)
    rng = random.Random(training_seed)
    optimizer = torch.optim.AdamW(parameters.values(), lr=learning_rate, weight_decay=0.0)
    candidates, dev_scores, history, optimizer_steps = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(exp.args.epochs):
        model.train()
        order = list(range(len(x)))
        rng.shuffle(order)
        losses = []
        for offset in range(0, len(order), exp.args.batch_size):
            batch = order[offset:offset + exp.args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = exp.logits(model, [x[i] for i in batch])
            targets = torch.tensor([exp.answer_token_ids[CHOICES.index(y[i])] for i in batch],
                                   device=exp.device)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("The masked update produced non-finite loss.")
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        history.append(float(torch.stack(losses).mean()))
        optimizer_steps.append(len(losses))
        dev_scores.append(exp.score(model, validation))
        candidates.append({name: p.detach().to(device="cpu", copy=True) for name, p in parameters.items()})
        print(f"  epoch {epoch + 1}: loss={history[-1]:.5f}", flush=True)
    torch.cuda.synchronize()
    return {
        "states": candidates, "dev_scores": dev_scores, "train_loss": history,
        "optimizer_steps_by_epoch": optimizer_steps,
        "seconds_train_and_dev": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def run_pair(exp, model, parameters, x, validation, test, rows, books, training_seed, learning_rate,
             *, training_labels=None, novel_test=None):
    labels = {k: [gold(v, rule) for rule in books] for k, v in rows.items()}
    targets = labels["train"] if training_labels is None else training_labels
    fitting = []
    for label, y in zip(CHOICES, targets, strict=True):
        print(f" copy {label}", flush=True)
        fitting.append(evaluate_epoch(exp, model, parameters, x, y, validation, training_seed, learning_rate))
    dev = []
    for i in range(exp.args.epochs):
        a, b = [select(run["dev_scores"][i]) for run in fitting]
        dev.append(pair_metrics(rows["dev"], *labels["dev"], a, b))
    if training_labels is None:
        best = max(range(len(dev)), key=lambda i: dev[i]["disagreement"]["both_correct"])
        print(f"  matched epoch {best + 1} selected from validation", flush=True)
    else:
        best = exp.args.epochs - 1
        print(f"  using fixed epoch {best + 1} for the format-only control", flush=True)
    evaluations = {}
    for epoch in sorted({best + 1, *exp.args.test_epochs}):
        scores, novel_scores = [], []
        for run in fitting:
            with torch.no_grad():
                for name, values in run["states"][epoch - 1].items():
                    parameters[name].copy_(values)
            scores.append(exp.score(model, test))
            if novel_test is not None:
                novel_scores.append(exp.score(model, novel_test))
        evaluations[epoch] = {
            "test": pair_metrics(rows["test"], *labels["test"], *[select(s) for s in scores]),
            "test_scores": scores,
        }
        if novel_test is not None:
            evaluations[epoch].update(
                novel_wording_test=pair_metrics(rows["test"], *labels["test"], *map(select, novel_scores)),
                novel_wording_test_scores=novel_scores,
                optimizer_steps=[sum(run["optimizer_steps_by_epoch"][:epoch]) for run in fitting],
            )
    states = [run["states"][best] for run in fitting]
    metrics = {
        "selected_epoch": best + 1, "dev_by_epoch": dev, **evaluations[best + 1],
        "changed_parameters": [sum(int(v.count_nonzero()) for v in side.values()) for side in states],
        "training_loss": [run["train_loss"] for run in fitting],
        "seconds_train_and_dev": [run["seconds_train_and_dev"] for run in fitting],
        "peak_allocated_bytes": [run["peak_allocated_bytes"] for run in fitting],
        "peak_reserved_bytes": [run["peak_reserved_bytes"] for run in fitting],
    }
    if exp.args.test_epochs:
        metrics["fixed_epoch_evaluations"] = {epoch: evaluations[epoch] for epoch in exp.args.test_epochs}
    return metrics, states


def run_draw(exp, model, seed, draw, conventional, masks, args, training_seed, *, training_labels=None):
    books = draw["rulebooks"]
    rows = {k: [Parcel(**r) for r in v] for k, v in draw["examples"].items()}
    check_data(rows, *books)
    train = exp.encode([r.prompt() for r in rows["train"]])
    validation = exp.encode([r.prompt() for r in rows["dev"]])
    test = exp.encode([r.prompt(template=0) for r in rows["test"]])
    novel_test = None
    if getattr(args, "compare_test_wording", False):
        novel_test = exp.encode([r.prompt(template=2) for r in rows["test"]])
    expected = [gold(rows["test"], rule) for rule in books]
    shortcuts = shortcut_metrics(rows["test"], *expected)
    best = max(shortcuts.values(), key=lambda m: (m["disagreement"]["both_correct"],
                                                  m["agreement"]["both_correct"]))
    frozen = exp.score(model, test)
    readout = {
        "frozen": pair_metrics(rows["test"], *expected, select(frozen), select(frozen)),
        "best_fixed_shortcut": best,
        "conventional_lora": conventional,
        "budgets": {},
    }
    folder = args.output / f"rulebook-{seed}"
    folder.mkdir(exist_ok=True)
    for budget, mask in masks.items():
        print(f"Rulebook {seed} / {budget} trainable scalars", flush=True)
        dense = next(iter(mask.values())) is None
        lr = args.dense_learning_rate if dense else args.learning_rate
        with apply_mask(model, mask, args.basis_seed, budget) as parameters:
            if exp.score(model, test[:exp.args.batch_size]) != frozen[:exp.args.batch_size]:
                raise RuntimeError("The zero-valued sparse adapter changed baseline logits.")
            result, states = run_pair(exp, model, parameters, train, validation, test,
                                      rows, books, training_seed, lr, training_labels=training_labels,
                                      novel_test=novel_test)
        del parameters
        gc.collect()
        torch.cuda.empty_cache()
        result.update(learning_rate=lr, dense_b=dense)
        if novel_test is not None:
            result.update(trainable_parameters=budget, test_record_ids=[row.id for row in rows["test"]])
        if budget <= 1000:
            result["learned_values"] = [{name: v.tolist() for name, v in side.items()} for side in states]
        else:
            from safetensors.torch import save_file
            target = folder / f"budget-{budget}-values.safetensors"
            save_file({f"{copy}.{name}": v for copy, side in zip(CHOICES, states, strict=True)
                       for name, v in side.items()}, target)
            result["learned_values_file"] = target.name
        write_json(folder / f"budget-{budget}.json", result)
        keys = ("selected_epoch", "changed_parameters", "test", "seconds_train_and_dev",
                "peak_allocated_bytes", "peak_reserved_bytes", "learning_rate", "dense_b")
        readout["budgets"][budget] = {k: result[k] for k in keys}
        if args.test_epochs:
            fixed = {epoch: evaluation["test"]
                     for epoch, evaluation in result["fixed_epoch_evaluations"].items()}
            readout["budgets"][budget]["test_by_epoch"] = fixed
            counts = {epoch: value["disagreement"]["both_correct"] for epoch, value in fixed.items()}
            print(f"  fixed epoch tests: {counts}", flush=True)
        write_json(folder / "summary.json", readout)
        print("  test:", test_result_text(result), flush=True)
    return readout


def test_result_text(result):
    score = result["test"]["disagreement"]
    return f"{score['both_correct']}/{score['n']} paired, changed {result['changed_parameters']}"


def matched_conventional(path, model, revision, wording, source_sha):
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    seed = str(protocol["rule_seed"])
    draw = wording["draws"].get(seed)
    if (
        (protocol["model"], protocol["revision"]) != (model, revision)
        or protocol["data_protocol_sha256"] != source_sha
        or draw is None
        or [protocol["rulebooks"]["a"], protocol["rulebooks"]["b"]] != draw["rulebooks"]
        or protocol["examples"] != draw["examples"]
    ):
        raise ValueError("The matched LoRA run must share the checkpoint, parcel split, and rulebooks.")
    report = json.loads((path / "results.json").read_text(encoding="utf-8"))
    return {seed: report["trained_hidden_seen_wording"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_arguments(parser)
    parser.add_argument("--wording-run", type=Path, required=True)
    parser.add_argument("--matched-run", type=Path,
                        help="Conventional LoRA results on the shared split for a second checkpoint.")
    parser.add_argument("--frozen-masks", type=Path,
                        help="Use a prepared protocol and verify coordinates against the frozen masks JSON.")
    parser.add_argument("--rule-seed", type=str, help="Run just this draw from the source protocol.")
    parser.add_argument("--compare-test-wording", action="store_true",
                        help="Also save test metrics and per-parcel choice logits for template 2.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-seed", type=int, default=119)
    parser.add_argument("--basis-seed", type=int, help="Seed for the frozen A matrices. Defaults to the mask seed.")
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                        help="Train these scalar budgets in the supplied order.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--test-epochs", type=int, nargs="+", default=(),
                        help="Also evaluate these fixed checkpoints on the held-out test set.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dense-b", action="store_true", help="Also train every B coefficient with A still frozen.")
    parser.add_argument("--dense-learning-rate", type=float, default=0.001)
    parser.add_argument("--reference-sweep", type=Path, help="Verify nesting against an earlier sweep's protocol.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Freeze an NF4-quantized backbone.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Recompute transformer activations during backpropagation to save memory.")
    parser.add_argument("--checkpoint-interval", type=int, default=1,
                        help="With gradient checkpointing, checkpoint every Nth decoder layer.")
    args = parser.parse_args()
    resolve_revision(parser, args)
    if args.basis_seed is None:
        args.basis_seed = args.mask_seed
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < 1:
        parser.error("Training requires positive epochs and batch size and a learning rate between 0 and 1.")
    if len(args.test_epochs) != len(set(args.test_epochs)) or any(not 1 <= n <= args.epochs for n in args.test_epochs):
        parser.error("Test checkpoints must be distinct epochs within the training schedule.")
    if min(args.budgets) < 1 or len(args.budgets) != len(set(args.budgets)):
        parser.error("Supply distinct positive budgets.")
    if not 0 < args.dense_learning_rate < 1:
        parser.error("The dense control learning rate must be between 0 and 1.")
    if args.checkpoint_interval < 1 or (args.checkpoint_interval != 1 and not args.gradient_checkpointing):
        parser.error("A positive checkpoint interval above one requires gradient checkpointing.")
    quantization = "NF4 with double quantization" if args.load_in_4bit else None
    protocol_path = args.wording_run / "protocol.json"
    wording_bytes = protocol_path.read_bytes()
    source_sha = hashlib.sha256(wording_bytes).hexdigest()
    wording = json.loads(wording_bytes)
    if (wording["model"], wording["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("Expected a source protocol for the pinned Qwen checkpoint.")
    if args.frozen_masks and (wording["choice_suffix"] != CHOICE_SUFFIX
                              or tuple(wording["training_templates"][:2]) != TEMPLATES[:2]):
        raise ValueError("Frozen data must use the same training prompts and answer choices.")
    if not args.frozen_masks:
        reference = json.loads((args.wording_run / "results.json").read_text(encoding="utf-8"))
        if set(reference) != set(wording["draws"]):
            raise ValueError("Expected the completed Qwen wording study as the data source.")
    if args.matched_run:
        conventional = matched_conventional(args.matched_run, args.model_id, args.revision, wording, source_sha)
    else:
        if (args.model_id, args.revision) != (MODEL_ID, REVISION):
            parser.error("A new checkpoint requires a completed --matched-run on the shared split.")
        if args.frozen_masks:
            conventional = dict.fromkeys(wording["draws"])
        else:
            conventional = {seed: report["two-templates"]["per_template"]["familiar"]
                            for seed, report in reference.items()}
    if args.rule_seed is not None:
        if args.rule_seed not in conventional:
            parser.error("The source does not contain the requested rulebook seed.")
        conventional = {args.rule_seed: conventional[args.rule_seed]}
    args.output.mkdir(parents=True, exist_ok=False)
    exp = Experiment(SimpleNamespace(checkpoint=args.checkpoint, model_id=args.model_id,
                                     revision=args.revision, epochs=args.epochs,
                                     batch_size=args.batch_size, learning_rate=args.learning_rate,
                                     test_epochs=args.test_epochs, load_in_4bit=args.load_in_4bit))
    model = exp.model().requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.checkpoint_interval > 1:
            from transformers.modeling_layers import GradientCheckpointingLayer

            layers = [layer for layer in model.modules()
                      if isinstance(layer, GradientCheckpointingLayer) and layer.gradient_checkpointing]
            if not layers:
                raise ValueError("This checkpoint does not expose activation checkpointing by layer.")
            for i, layer in enumerate(layers):
                layer.gradient_checkpointing = i % args.checkpoint_interval == 0
    modules = eligible_modules(model)
    prior = json.loads((args.reference_sweep / "protocol.json").read_text(encoding="utf-8")) if args.reference_sweep else None
    prior_budgets = {
        int(n): mask for n, mask in prior["budgets"].items()
        if all(coordinates is not None for coordinates in mask.values())
    } if prior else {}
    frozen = {}
    if args.frozen_masks:
        frozen = json.loads(args.frozen_masks.read_text(encoding="utf-8"))[str(args.mask_seed)]
    budgets_to_sample = set(args.budgets) | prior_budgets.keys() | set(map(int, frozen))
    eligible, nested = sample_masks(modules, sorted(budgets_to_sample), args.mask_seed)
    if prior and (
        prior["mask_seed"] != args.mask_seed
        or (prior["model"], prior["revision"]) != (args.model_id, args.revision)
        or prior.get("basis_seed", prior["mask_seed"]) != args.basis_seed
        or prior["wording_protocol_sha256"] != source_sha
        or prior.get("base_quantization") != quantization
        or prior.get("gradient_checkpointing", False) != args.gradient_checkpointing
        or prior.get("checkpoint_interval", 1) != args.checkpoint_interval
        or any(nested[n] != mask for n, mask in prior_budgets.items())
    ):
        raise ValueError("The larger masks must preserve the original sweep's coordinates and dataset.")
    masks = {n: nested[n] for n in args.budgets}
    if args.frozen_masks and any(frozen.get(str(n)) != mask for n, mask in masks.items()):
        raise ValueError("Sampled coordinates differ from the frozen masks.")
    if args.dense_b:
        if eligible in masks:
            parser.error("The dense B budget is already included.")
        masks[eligible] = {name: None for name, _ in modules}
    write_json(args.output / "protocol.json", {
        "model": args.model_id, "revision": args.revision, "runtime": exp.config(),
        "wording_run": args.wording_run.name, "wording_protocol_sha256": source_sha,
        "matched_run": args.matched_run.name if args.matched_run else None,
        "rank": RANK, "alpha": ALPHA, "mask_seed": args.mask_seed, "basis_seed": args.basis_seed,
        "eligible_B_coordinates": eligible,
        "budgets": masks, "training_seed": wording["training_seed"],
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "fixed_test_epochs": args.test_epochs,
        "optimizer_steps_by_epoch": {
            seed: math.ceil(len(wording["draws"][seed]["examples"]["train"]) / args.batch_size)
            for seed in conventional
        },
        "base_quantization": quantization,
        "gradient_checkpointing": args.gradient_checkpointing,
        "checkpoint_interval": args.checkpoint_interval,
        "dense_b_learning_rate": args.dense_learning_rate if args.dense_b else None,
        "reference_sweep": args.reference_sweep.name if args.reference_sweep else None,
        "selection": "Earliest shared epoch with maximum paired disagreement accuracy on validation.",
        "primary": "Paired correctness on disagreement records under familiar wording, with the rulebooks withheld.",
        **({"test_wording_templates": {"test": 0, "novel_wording_test": 2}}
           if args.compare_test_wording else {}),
    })
    combined = {}
    for seed, comparison in conventional.items():
        combined[seed] = run_draw(exp, model, seed, wording["draws"][seed], comparison,
                                  masks, args, wording["training_seed"])
        write_json(args.output / "results.json", combined)


if __name__ == "__main__":
    main()
