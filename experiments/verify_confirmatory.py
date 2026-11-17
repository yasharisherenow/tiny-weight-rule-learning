"""Recompute the confirmatory metrics from frozen rows and raw output files."""

import hashlib
import math
import random
from statistics import fmean

from run_confirmatory import (
    ANALYSIS, BASIS_SEED, EDL_CONDITIONS, EDL_RATES, EDL_SIZE, LORA_BATCH_SIZE, LORA_EPOCHS, LORA_RATE,
    MODEL_ID, REVISION, RULEBOOK_SEEDS, SPARSE_BATCH_SIZE, SPARSE_EPOCHS, SPARSE_RATE, TEST_EPOCHS, TRAINING_SEED,
    build_study_data, build_study_masks, jobs_for, read_json, sha256, tree_hashes, verify_inputs, write_json,
)


def require(test, message):
    if not test:
        raise ValueError(message)


def expected(row, book):
    return "A" if row[book[row["depot"]]] == row["option_a"] else "B"


def answer(scores):
    require(len(scores) == 2 and all(math.isfinite(x) for x in scores), "Invalid pair of choice logits.")
    return "A" if scores[0] >= scores[1] else "B"


def paired(rows, books, logits):
    require(len(logits) == 2, "Expected logits for both model copies.")
    strata = {name: dict.fromkeys(("n", "a_correct", "b_correct", "both_correct"), 0)
              for name in ("disagreement", "agreement")}
    for row, a, b in zip(rows, *logits, strict=True):
        name = "disagreement" if row["printed"] != row["scanned"] else "agreement"
        counts = strata[name]
        correct_a, correct_b = answer(a) == expected(row, books[0]), answer(b) == expected(row, books[1])
        counts["n"] += 1
        counts["a_correct"] += correct_a
        counts["b_correct"] += correct_b
        counts["both_correct"] += correct_a and correct_b
    return strata


def check_job(root, job, inputs_hash, settings, dependency_hashes):
    receipt_path = root / job.name / "completed.json"
    receipt = read_json(receipt_path)
    request = receipt["request"]
    attempt = root / job.name / receipt["attempt"]
    result = attempt / "results"
    require(request["fits"] == job.fits and request["inputs_sha256"] == inputs_hash
            and request["training_script_sha256"] == settings["training_script_sha256"]
            and request["dependency_receipt_sha256"] == dependency_hashes,
            f"{job.name} receipt does not match the frozen job or its dependencies.")
    require(set(job.outputs) <= receipt["artifacts"].keys() and tree_hashes(result) == receipt["artifacts"]
            and sha256(attempt / "run.log") == receipt["log_sha256"],
            f"{job.name} has an incomplete or changed artifact.")
    return result, sha256(receipt_path)


def check_runtime(runtime, settings):
    require(runtime["dtype"] == settings["dtype"] and all(
        runtime[key] == settings["packages"][key] for key in ("torch", "transformers", "peft")
    ), "The saved runtime does not match the frozen bfloat16 stack.")


def check_training(metrics, steps=48):
    history = metrics["dev_history"]
    require([(item["epoch"], item["optimizer_steps"]) for item in history]
            == [(epoch, steps) for epoch in range(1, LORA_EPOCHS + 1)], "Unexpected LoRA optimizer step count.")
    require(metrics["trainable_parameters"] == 4_399_104, "Unexpected ordinary LoRA parameter count.")
    require(all(math.isfinite(item["mean_training_loss"]) for item in history), "Invalid LoRA training loss.")
    require(metrics["seconds_train_and_dev"] > 0 and metrics["peak_allocated_bytes"] > 0
            and metrics["peak_reserved_bytes"] >= metrics["peak_allocated_bytes"],
            "Missing or invalid LoRA runtime measurements.")


def compare_numbers(reconstructed, saved):
    return reconstructed.keys() == saved.keys() and all(
        math.isclose(value, saved[key], rel_tol=1e-9, abs_tol=1e-10)
        for key, value in reconstructed.items()
    )


def saved_adapter_hash(path, copy, modules):
    from safetensors.torch import load_file

    weights = load_file(path / f"adapter-{copy.lower()}" / "adapter_model.safetensors", device="cpu")

    def training_order(key):
        _, _, tail = key.partition(".layers.")
        layer, *_, module, lora, weight = tail.split(".")
        require(lora in ("lora_A", "lora_B") and weight == "weight", f"Unexpected adapter tensor name {key}.")
        return int(layer), modules.index(module), lora

    require(sum(value.numel() for value in weights.values()) == 4_399_104, "Saved adapter has the wrong size.")
    digest = hashlib.sha256()
    for key in sorted(weights, key=training_order):
        digest.update(f"{key.removesuffix('.weight')}.default.weight".encode("utf-8"))
        digest.update(weights[key].float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_ordinary(path, draw, seed, settings, protocol_hash):
    protocol = read_json(path / "protocol.json")
    require((protocol["model"], protocol["revision"], protocol["training_seed"], protocol["rule_seed"])
            == (MODEL_ID, REVISION, TRAINING_SEED, seed) and protocol["data_protocol_sha256"] == protocol_hash
            and list(protocol["rulebooks"].values()) == draw["rulebooks"] and protocol["examples"] == draw["examples"],
            "The ordinary LoRA protocol does not match the frozen inputs.")
    lora = protocol["training"]
    require((lora["epochs"], lora["batch_size"], lora["learning_rate"], lora["lora_rank"], lora["lora_alpha"],
             lora["target_modules"]) == (LORA_EPOCHS, LORA_BATCH_SIZE, LORA_RATE, settings["rank"],
                                        settings["alpha"], settings["target_modules"]), "Unexpected LoRA protocol.")
    report = read_json(path / "results.json")
    check_runtime(report["runtime"], settings)
    rows = draw["examples"]["test"]
    books = draw["rulebooks"]
    predictions = read_json(path / "predictions.json")
    conditions = ("frozen_hidden", "frozen_visible", "trained_hidden_seen_wording", "trained_hidden_novel_wording")
    scores = {condition: [[], []] for condition in conditions}
    for row, prediction in zip(rows, predictions, strict=True):
        require((prediction["id"], prediction["expected_a"], prediction["expected_b"])
                == (row["id"], expected(row, books[0]), expected(row, books[1])), "Ordinary prediction row mismatch.")
        for condition in conditions:
            for values, copy in zip(scores[condition], prediction["conditions"][condition], strict=True):
                logits = copy["scores_a_b"]
                require(copy["prediction"] == answer(logits), "Ordinary prediction and logits disagree.")
                values.append(logits)
    metrics = {name: paired(rows, books, scores[name]) for name in conditions}
    require(all(report[name] == values for name, values in metrics.items()), "Ordinary LoRA metrics disagree with logits.")
    for copy in "AB":
        raw = read_json(path / f"metrics-{copy.lower()}.json")
        require(raw == report["training"][copy], "Ordinary copy fit metadata differs from the report.")
        check_training(raw)
        require(saved_adapter_hash(path, copy, settings["target_modules"]) == raw["final_adapter_sha256"],
                "Ordinary source adapter hash does not match the saved tensors.")
    require(report["training"]["A"]["initial_adapter_sha256"] == report["training"]["B"]["initial_adapter_sha256"],
            "The ordinary copies used different adapter initializations.")
    return {**metrics, "training": report["training"], "runtime": report["runtime"]}


def check_sparse(path, draw, seed, mask_seed, budget, masks, settings, protocol_hash):
    protocol = read_json(path / "protocol.json")
    check_runtime(protocol["runtime"], settings)
    require((protocol["model"], protocol["revision"], protocol["training_seed"], protocol["basis_seed"],
             protocol["rank"], protocol["alpha"], protocol["mask_seed"], protocol["epochs"], protocol["batch_size"],
             protocol["learning_rate"], protocol["wording_protocol_sha256"])
            == (MODEL_ID, REVISION, TRAINING_SEED, BASIS_SEED, settings["rank"], settings["alpha"], mask_seed,
                SPARSE_EPOCHS, SPARSE_BATCH_SIZE, SPARSE_RATE, protocol_hash), "Unexpected sparse training protocol.")
    mask = masks[str(mask_seed)][str(budget)]
    require(protocol["budgets"] == {str(budget): mask}
            and protocol["fixed_test_epochs"] == list(TEST_EPOCHS)
            and protocol["optimizer_steps_by_epoch"] == {str(seed): 24}
            and protocol["test_wording_templates"] == {"test": 0, "novel_wording_test": 2},
            "Unexpected sparse mask, evaluation wording, or optimizer steps.")
    summary = read_json(path / "results.json")[str(seed)]
    require(summary["best_fixed_shortcut"]["disagreement"]["both_correct"] == 64, "Unexpected sparse shortcut.")
    budget_path = path / f"rulebook-{seed}"
    budget_result = read_json(budget_path / f"budget-{budget}.json")
    rows, books = draw["examples"]["test"], draw["rulebooks"]
    require(budget_result["test_record_ids"] == [row["id"] for row in rows]
            and budget_result["trainable_parameters"] == budget and not budget_result["dense_b"],
            "Sparse prediction IDs or trainable parameter count mismatch.")
    fixed = budget_result["fixed_epoch_evaluations"]
    require(set(fixed) == set(map(str, TEST_EPOCHS)), "Missing or extra sparse checkpoint evaluations.")
    points = {}
    for epoch in TEST_EPOCHS:
        step = 24 * epoch
        at = fixed[str(epoch)]
        require(at["optimizer_steps"] == [step, step], "The fixed checkpoint was scored at the wrong step.")
        familiar = paired(rows, books, at["test_scores"])
        novel = paired(rows, books, at["novel_wording_test_scores"])
        require(familiar == at["test"] == summary["budgets"][str(budget)]["test_by_epoch"][str(epoch)]
                and novel == at["novel_wording_test"], "Sparse pair counts disagree with saved logits.")
        points[str(step)] = {"familiar": familiar, "novel": novel}
    measurements = {key: budget_result[key] for key in (
        "changed_parameters", "seconds_train_and_dev", "peak_allocated_bytes", "peak_reserved_bytes",
    )}
    require(all(len(values) == 2 for values in measurements.values()) and all(
        0 <= count <= budget for count in measurements["changed_parameters"]
    ) and all(x > 0 for x in measurements["seconds_train_and_dev"] + measurements["peak_allocated_bytes"])
            and all(used <= reserved for used, reserved in zip(measurements["peak_allocated_bytes"],
                                                               measurements["peak_reserved_bytes"], strict=True)),
            "Missing or invalid sparse runtime measurements.")
    if "learned_values" in budget_result:
        values = budget_result["learned_values"]
    else:
        from safetensors.torch import load_file

        filename = f"budget-{budget}-values.safetensors"
        require(budget_result["learned_values_file"] == filename, "Unexpected sparse checkpoint file.")
        tensors = load_file(budget_path / filename, device="cpu")
        require(set(tensors) == {f"{copy}.{name}" for copy in "AB" for name in mask},
                "Unexpected tensors in the sparse checkpoint.")
        values = [{name: tensors[f"{copy}.{name}"].tolist() for name in mask} for copy in "AB"]
    sizes = {name: len(indices) for name, indices in mask.items()}
    for state, changed in zip(values, measurements["changed_parameters"], strict=True):
        require({name: len(weights) for name, weights in state.items()} == sizes
                and all(math.isfinite(w) for weights in state.values() for w in weights)
                and sum(w != 0 for weights in state.values() for w in weights) == changed,
                "The saved sparse model values disagree with frozen mask sizes or recorded changed counts.")
    return {"mask_seed": mask_seed, "budget": budget, "points": points, **measurements}


def loss_summary(first_nats, final_nats):
    require(all(math.isfinite(v) and v >= 0 for v in first_nats + final_nats), "Invalid full-vocabulary NLL.")
    first = fmean(first_nats) / math.log(2)
    final = fmean(final_nats) / math.log(2)
    edl = first - final
    return {"labels": len(first_nats), "prequential_bits": first * len(first_nats),
            "first_pass_bits_per_label": first, "final_bits_per_label": final, "edl_bits": edl * len(first_nats),
            "edl_bits_per_label": edl}


def loss_strata(first, final, train, test):
    groups = {"all": loss_summary([loss for _, loss in first], final)}
    for name, disagrees in (("agreement", False), ("disagreement", True)):
        first_group = [loss for i, loss in first if (train[i]["printed"] != train[i]["scanned"]) == disagrees]
        final_group = [loss for row, loss in zip(test, final, strict=True)
                       if (row["printed"] != row["scanned"]) == disagrees]
        groups[name] = loss_summary(first_group, final_group)
    return groups


def check_edl(path, prior_path, format_path, draw, seed, rate, inputs, settings, protocol_hash, ordinary):
    protocol = read_json(path / "protocol.json")
    books = draw["rulebooks"]
    require((protocol["model"], protocol["revision"], protocol["rule_seed"], protocol["training_seed"],
             protocol["adaptation_learning_rate"], protocol["sizes"])
            == (MODEL_ID, REVISION, str(seed), TRAINING_SEED, rate, [EDL_SIZE])
            and protocol["rulebooks"] == books and protocol["wording_protocol_sha256"] == protocol_hash
            and protocol["prior_protocol_sha256"] == sha256(prior_path / "protocol.json")
            and protocol["training"] == read_json(prior_path / "protocol.json")["training"]
            and all(protocol[k] == value for k, value in inputs.items())
            and protocol["opposite_prior_adapter_by_target"] == {"A": "B", "B": "A"},
            "EDL protocol does not match the frozen adaptation inputs.")
    module_order = settings["target_modules"]
    prior_hashes = {copy: fit["final_adapter_sha256"] for copy, fit in ordinary["training"].items()}
    require(protocol["prior_adapters"] == prior_hashes, "Wrong ordinary source adapter hashes in EDL protocol.")
    format_hashes, format_metrics = {}, {}
    exposure = format_path / "format-exposure"
    for copy in "AB":
        meta = read_json(exposure / f"metrics-{copy.lower()}.json")
        check_training(meta)
        require(meta["initial_adapter_sha256"] == ordinary["training"][copy]["initial_adapter_sha256"],
                "Format exposure did not match the ordinary initial adapter.")
        format_hashes[copy] = saved_adapter_hash(exposure, copy, module_order)
        format_metrics[copy] = meta
        require(format_hashes[copy] == meta["final_adapter_sha256"], "Format adapter tensors differ from their metadata.")
    if rate == EDL_RATES[0]:
        require(path == format_path and "format_exposure_reference" not in protocol,
                "The first EDL job did not create the format control.")
    else:
        reference = protocol["format_exposure_reference"]
        require(reference["adapter_hashes"] == format_hashes
                and reference["protocol_sha256"] == sha256(format_path / "protocol.json"),
                "The second rate did not reuse the first rate's format-control adapters.")
    starts = {"prior-rulebook": prior_hashes, "format-only": format_hashes,
              "opposite-rulebook": {"A": prior_hashes["B"], "B": prior_hashes["A"]}}
    report = read_json(path / "results.json")
    check_runtime(report["runtime"], settings)
    require(report["exposure_fit_count"] == (2 if rate == EDL_RATES[0] else 0)
            and report["adaptation_fit_count"] == 6 and report["best_fixed_shortcut"] == 144,
            "Unexpected EDL fit count or fixed shortcut.")
    saved = report["sizes"][str(EDL_SIZE)]
    require(set(saved) == set(EDL_CONDITIONS), "Unexpected EDL condition set.")
    train = inputs["adaptation_examples"]["train"]
    test = inputs["adaptation_examples"]["test"]
    require([row["id"] for row in train] == inputs["nested_train_record_ids"][str(EDL_SIZE)]
            and len(train) == len(test) == EDL_SIZE, "EDL label loss inputs have the wrong order or size.")
    order = list(range(EDL_SIZE))
    random.Random(TRAINING_SEED).shuffle(order)
    initial_predictions = read_json(path / "starting_predictions.json")
    conditions = {}
    for condition in EDL_CONDITIONS:
        initial = initial_predictions[condition]
        require(all(len(m["label_nll_nats"]) == EDL_SIZE
                    and all(math.isfinite(v) and v >= 0 for v in m["label_nll_nats"]) for m in initial)
                and paired(test, books, [m["choice_logits"] for m in initial]) == report["starting_models"][condition],
                f"{condition} has invalid starting predictions or loss.")
        logits, copies, training = [], {}, {}
        for copy in "AB":
            folder = path / f"n-{EDL_SIZE}" / condition
            result = read_json(folder / f"metrics-{copy.lower()}.json")
            fit, heldout = result["training"], result["held_out"]
            check_training(fit)
            first = fit["first_exposure_nll_nats"]
            require(fit["initial_adapter_sha256"] == starts[condition][copy]
                    and [i for i, _ in first] == order and len(heldout["label_nll_nats"]) == EDL_SIZE
                    and fit["final_adapter_sha256"] == saved_adapter_hash(folder, copy, module_order),
                    f"{condition} copy {copy} has unexpected adapter provenance or exposure order.")
            copies[copy] = loss_strata(first, heldout["label_nll_nats"], train, test)
            require(all(compare_numbers(values, result["code"][name])
                        and compare_numbers(values, saved[condition]["copy_metrics"][copy][name])
                        for name, values in copies[copy].items()), f"EDL arithmetic mismatch for {condition} copy {copy}.")
            logits.append(heldout["choice_logits"])
            training[copy] = {key: fit[key] for key in (
                "seconds_train_and_dev", "peak_allocated_bytes", "peak_reserved_bytes", "initial_adapter_sha256",
            )}
        paired_accuracy = paired(test, books, logits)
        means = {name: fmean(copy[name]["edl_bits_per_label"] for copy in copies.values())
                 for name in ANALYSIS["edl_strata"]}
        final = fmean(copy["all"]["final_bits_per_label"] for copy in copies.values())
        require(paired_accuracy == saved[condition]["paired_accuracy"]
                and compare_numbers(means, saved[condition]["mean_edl_bits_per_label"])
                and math.isclose(final, saved[condition]["mean_final_bits_per_label"], abs_tol=1e-10),
                f"EDL condition summary does not follow from raw predictions for {condition}.")
        conditions[condition] = {"paired_accuracy": paired_accuracy, "mean_edl_bits_per_label": means,
                                 "mean_final_bits_per_label": final, "training": training}

    final_losses = [c["mean_final_bits_per_label"] for c in conditions.values()]
    span = max(final_losses) - min(final_losses)
    comparable = span <= ANALYSIS["edl_comparable_max_final_loss_span_bits_per_label"] and all(
        c["paired_accuracy"]["disagreement"]["n"] == ANALYSIS["edl_disagreement_total"]
        and c["paired_accuracy"]["disagreement"]["both_correct"] >= ANALYSIS["edl_comparable_min_paired_correct"]
        for c in conditions.values()
    )
    return {"rate": rate, "conditions": conditions, "final_loss_span_bits_per_label": span,
            "endpoints_comparable": comparable, "initial_adapter_sha256": starts, "first_exposure_index_order": order,
            **({"format_exposure_training": format_metrics} if rate == EDL_RATES[0] else {})}


def verify_draw(root, seed, checkpoint, inputs_hash):
    manifests = root / "manifests"
    protocol_path = manifests / "protocol.json"
    protocol_hash = sha256(protocol_path)
    draw = read_json(protocol_path)["draws"][str(seed)]
    masks, settings, edl_inputs = (read_json(manifests / name) for name in ("masks.json", "settings.json", "edl-inputs.json"))
    run = root / "runs" / f"rulebook-{seed}"
    receipts, paths, sparse, edl = {}, {}, [], []
    fits = 0
    for job in jobs_for(seed, manifests, checkpoint):
        path, digest = check_job(run, job, inputs_hash, settings, {name: receipts[name] for _, name in job.dependencies})
        receipts[job.name], paths[job.name] = digest, path
        fits += job.fits
        if job.name == "ordinary-lora":
            ordinary = check_ordinary(path, draw, seed, settings, protocol_hash)
        elif job.script == "run_scalar_sweep.py":
            mask_seed = int(job.args[job.args.index("--mask-seed") + 1])
            budget = int(job.args[job.args.index("--budgets") + 1])
            sparse.append(check_sparse(path, draw, seed, mask_seed, budget, masks, settings, protocol_hash))
        else:
            rate = float(job.args[job.args.index("--adaptation-learning-rate") + 1])
            format_path = paths[f"edl-rate-{EDL_RATES[0]:.5f}"]
            edl.append(check_edl(path, paths["ordinary-lora"], format_path, draw, seed, rate, edl_inputs[str(seed)],
                                 settings, protocol_hash, ordinary))
    if edl:
        first, second = edl
        require(first["initial_adapter_sha256"] == second["initial_adapter_sha256"]
                and first["first_exposure_index_order"] == second["first_exposure_index_order"],
                "The EDL rates did not use the same starting adapters and parcel order.")
    return {"rulebook": seed, "fit_count": fits, "inputs_sha256": inputs_hash, "job_receipt_sha256": receipts,
            "ordinary": ordinary, "sparse": sparse, "edl": edl}


def verify_design(manifests, checkpoint):
    inputs_hash = verify_inputs(manifests, checkpoint)
    protocol, edl_inputs = build_study_data()
    require(protocol == read_json(manifests / "protocol.json") and edl_inputs == read_json(manifests / "edl-inputs.json"),
            "The frozen routing or adaptation inputs do not reproduce from the study design.")
    masks, geometry = build_study_masks(checkpoint)
    expected_masks = {str(seed): {str(budget): mask for budget, mask in budgets.items()}
                      for seed, budgets in masks.items()}
    require(expected_masks == read_json(manifests / "masks.json")
            and geometry == read_json(manifests / "mask-geometry.json"),
            "The frozen mask geometry or coordinates do not reproduce from the checkpoint.")
    return inputs_hash


def verify_study(root, draws, checkpoint):
    inputs_hash = verify_design(root / "manifests", checkpoint)
    output = root / "verification"
    output.mkdir(exist_ok=True)
    for seed in draws:
        verification = verify_draw(root, seed, checkpoint, inputs_hash)
        target = output / f"rulebook-{seed}.json"
        write_json(target, verification)
        write_json(target.with_suffix(".sha256.json"), {"sha256": sha256(target)})
        print(f"Verified rulebook {seed} ({verification['fit_count']} training fits). Results in {target}", flush=True)


def read_verified_summary(path):
    checksum = read_json(path.with_suffix(".sha256.json"))
    require(sha256(path) == checksum["sha256"], f"Verification summary {path.name} changed after verification.")
    return read_json(path)


def load_verified_draws(root):
    manifests = root / "manifests"
    inputs_hash = sha256(manifests / "sha256.json")
    settings = read_json(manifests / "settings.json")
    for name, digest in read_json(manifests / "sha256.json").items():
        require(sha256(manifests / name) == digest, f"Frozen manifest {name} changed after verification.")
    draws = {}
    for seed in RULEBOOK_SEEDS:
        run = root / "runs" / f"rulebook-{seed}"
        verified = read_verified_summary(root / "verification" / f"rulebook-{seed}.json")
        receipts = {}
        fits = 0
        for job in jobs_for(seed, manifests, None):
            _, receipts[job.name] = check_job(run, job, inputs_hash, settings,
                                             {name: receipts[name] for _, name in job.dependencies})
            fits += job.fits
        require(verified["rulebook"] == seed and verified["fit_count"] == fits
                and verified["inputs_sha256"] == inputs_hash and verified["job_receipt_sha256"] == receipts,
                f"The verification report for {seed} does not cover the complete current run.")
        draws[seed] = verified
    require(sum(draw["fit_count"] for draw in draws.values()) == 58, "Expected exactly 58 verified study fits.")
    return inputs_hash, draws
