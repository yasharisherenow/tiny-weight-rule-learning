"""Run the confirmatory routing study against previously frozen inputs."""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from run_matched_pair import (
    CHOICES, CHOICE_SUFFIX, FIELDS, MODEL_ID, REVISION, TARGET_MODULES, TEMPLATES,
    build_split, check_data, gold, shortcut_metrics, write_json,
)


SCRIPTS = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPTS.parent / "output" / "experiments" / "qwen-05b-confirmatory"
RULEBOOK_DEPOTS = {
    90101: ("Alder", "Beech", "Cypress", "Dogwood"),
    90102: ("Hickory", "Juniper", "Larch", "Maple"),
    90106: ("Oak", "Pine", "Spruce", "Willow"),
}
RULEBOOK_SEEDS = tuple(RULEBOOK_DEPOTS)
EDL_SEEDS = (90101, 90102)
MASK_SEEDS = (123, 124)
BUDGETS = (1000, 2000)
TRAINING_SEED = 42
BASIS_SEED = 119
SPARSE_EPOCHS = 18
TEST_EPOCHS = (6, 12, 18)
SPARSE_BATCH_SIZE = 16
SPARSE_RATE = 0.01
LORA_EPOCHS = 6
LORA_BATCH_SIZE = 8
LORA_RATE = 0.0001
EDL_SIZE = 384
EDL_RATES = (LORA_RATE, 0.00005)
EDL_CONDITIONS = ("prior-rulebook", "format-only", "opposite-rulebook")
MANIFEST_FILES = ("protocol.json", "masks.json", "edl-inputs.json", "checkpoint.json", "settings.json",
                  "analysis.json", "mask-geometry.json")
CHECKPOINT_FILES = (
    "config.json", "generation_config.json", "merges.txt", "model.safetensors",
    "tokenizer_config.json", "tokenizer.json", "vocab.json",
)
PINNED_WEIGHTS_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
TRAINING_SCRIPTS = (
    "run_matched_pair.py", "run_scalar_sweep.py", "run_edl.py", "run_budget_confirmation.py",
)
TRAINING_PACKAGES = ("torch", "transformers", "peft", "safetensors", "accelerate", "tokenizers", "numpy")
SPLIT_SPECS = {
    "train": (range(100, 300), 72, 24, {0, 1}),
    "dev": (range(300, 500), 16, 8, {0, 1}),
    "test": (range(900, 1000), 32, 16, {2}),
}
EDL_SPLIT_SPECS = {
    "train": (range(500, 700), 72, 24, {0, 1}),
    "dev": (range(700, 800), 16, 8, {0, 1}),
    "test": (range(800, 900), 72, 24, {0, 1}),
}
ANALYSIS = {
    "paired_correct": "Count a parcel only if both copies predict their own assigned answer.",
    "choice_tie": "A",
    "sparse_primary_template": 0,
    "sparse_transfer_template": 2,
    "sparse_steps": [144, 288, 432],
    "sparse_disagreement_total": 128,
    "sparse_best_fixed_shortcut": 64,
    "sparse_summary": "Report each draw and mask, the 144 to 432 step change, and the range of draw means.",
    "edl_bits_per_label": "(mean first epoch pre-update full-vocabulary label NLL in nats minus mean final "
                          "held-out full-vocabulary label NLL in nats) divided by ln(2); average across copies.",
    "edl_strata": ["all", "disagreement", "agreement"],
    "edl_disagreement_total": 288,
    "edl_comparable_min_paired_correct": 274,
    "edl_comparable_max_final_loss_span_bits_per_label": 0.05,
    "edl_contrasts": ["format-only minus prior-rulebook", "opposite-rulebook minus format-only",
                      "opposite-rulebook minus prior-rulebook"],
}


@dataclass(frozen=True)
class Job:
    name: str
    script: str
    fits: int
    args: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[tuple[str, str], ...] = ()


def adapter_outputs(parent):
    return tuple(
        f"{parent}{name}"
        for copy in CHOICES
        for name in (f"metrics-{copy.lower()}.json", f"adapter-{copy.lower()}/adapter_model.safetensors")
    )


def jobs_for(draw, manifests, checkpoint):
    source = ("--checkpoint", str(checkpoint))
    yield Job("ordinary-lora", "run_matched_pair.py", 2, source + (
        "--data-protocol", str(manifests / "protocol.json"), "--rule-seed", str(draw),
        "--training-seed", str(TRAINING_SEED), "--epochs", str(LORA_EPOCHS),
        "--batch-size", str(LORA_BATCH_SIZE), "--learning-rate", str(LORA_RATE),
    ), ("protocol.json", "results.json", "predictions.json", *adapter_outputs("")))

    for mask_seed in MASK_SEEDS:
        for budget in BUDGETS:
            prefix = f"rulebook-{draw}/budget-{budget}"
            outputs = ("protocol.json", "results.json", f"{prefix}.json")
            if budget > 1000:
                outputs += (f"{prefix}-values.safetensors",)
            yield Job(f"sparse-mask-{mask_seed}-budget-{budget}", "run_scalar_sweep.py", 2, source + (
                "--wording-run", str(manifests), "--frozen-masks", str(manifests / "masks.json"),
                "--rule-seed", str(draw), "--mask-seed", str(mask_seed), "--basis-seed", str(BASIS_SEED),
                "--budgets", str(budget), "--epochs", str(SPARSE_EPOCHS), "--test-epochs", *map(str, TEST_EPOCHS),
                "--batch-size", str(SPARSE_BATCH_SIZE), "--learning-rate", str(SPARSE_RATE),
                "--compare-test-wording",
            ), outputs)

    if draw in EDL_SEEDS:
        for rate in EDL_RATES:
            name = f"edl-rate-{rate:.5f}"
            dependencies = (("--prior-run", "ordinary-lora"),)
            outputs = ("protocol.json", "results.json", "starting_predictions.json")
            fits = len(EDL_CONDITIONS) * len(CHOICES)
            if rate == EDL_RATES[0]:
                outputs += adapter_outputs("format-exposure/")
                fits += len(CHOICES)
            else:
                dependencies += (("--reuse-format-run", f"edl-rate-{EDL_RATES[0]:.5f}"),)
            for condition in EDL_CONDITIONS:
                outputs += adapter_outputs(f"n-{EDL_SIZE}/{condition}/")
            yield Job(name, "run_edl.py", fits, source + (
                "--wording-run", str(manifests), "--frozen-inputs", str(manifests / "edl-inputs.json"),
                "--include-opposite", "--sizes", str(EDL_SIZE), "--adaptation-learning-rate", str(rate),
            ), outputs, dependencies)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hashes(root):
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def study_settings():
    jobs = {}
    for draw in RULEBOOK_SEEDS:
        jobs[str(draw)] = [asdict(job) for job in jobs_for(draw, Path("manifests"), Path("checkpoint"))]
    return json.loads(json.dumps({
        "training_seed": TRAINING_SEED, "dtype": "torch.bfloat16", "rank": 8, "alpha": 16,
        "target_modules": TARGET_MODULES, "optimizer": "AdamW", "weight_decay": 0.0,
        "python": platform.python_version(), "packages": {name: version(name) for name in TRAINING_PACKAGES},
        "training_script_sha256": {name: sha256(SCRIPTS / name) for name in TRAINING_SCRIPTS},
        "jobs": jobs,
    }))


def check_strata(rows, books, spec):
    lanes, disagree, agree, templates = spec
    ids = {row.id for row in rows}
    if len(ids) != len(rows) or {row.depot for row in rows} != books[0].keys():
        raise ValueError("Parcel IDs or depot coverage do not match the study design.")
    if {row.template for row in rows} != templates or any(
        lane not in lanes for row in rows for lane in (row.option_a, row.option_b, row.printed, row.scanned)
    ):
        raise ValueError("Parcel lane values or prompt templates do not match the split specification.")
    for depot in books[0]:
        for differs, expected in ((True, disagree), (False, agree)):
            group = [row for row in rows if row.depot == depot and (row.printed != row.scanned) == differs]
            if len(group) != expected or any(gold(group, book).count("A") * 2 != expected for book in books):
                raise ValueError(f"Unexpected parcel count or answer balance for {depot}.")
    return {lane for row in rows for lane in (row.option_a, row.option_b)}


def build_study_data():
    from run_edl import build_edl_inputs

    draws, edl = {}, {}
    for seed, depots in RULEBOOK_DEPOTS.items():
        fields = [*FIELDS, *FIELDS]
        random.Random(seed).shuffle(fields)
        rule_a = dict(zip(depots, fields, strict=True))
        rule_b = {depot: FIELDS[1 - FIELDS.index(field)] for depot, field in rule_a.items()}
        books = [rule_a, rule_b]
        rows = {name: build_split(seed, name, lanes, depots, disagree, agree)
                for name, (lanes, disagree, agree, _) in SPLIT_SPECS.items()}
        check_data(rows, *books)
        observed = set()
        for name, spec in SPLIT_SPECS.items():
            seen = check_strata(rows[name], books, spec)
            if seen & observed:
                raise ValueError("The rulebook splits share lane values.")
            observed |= seen
        test = rows["test"]
        shortcuts = shortcut_metrics(test, *(gold(test, book) for book in books))
        if max(s["disagreement"]["both_correct"] for s in shortcuts.values()) != 64:
            raise ValueError(f"Unexpected fixed shortcut for rulebook {seed}.")
        draw = {"rulebooks": books, "examples": {k: [asdict(row) for row in v] for k, v in rows.items()}}
        draws[str(seed)] = draw
        if seed in EDL_SEEDS:
            inputs, _, adaptation, _ = build_edl_inputs(draw, str(seed), (EDL_SIZE,))
            for name, spec in EDL_SPLIT_SPECS.items():
                seen = check_strata(adaptation[name], books, spec)
                if seen & observed:
                    raise ValueError("An EDL split shares lane values with another split.")
                observed |= seen
            edl[str(seed)] = inputs
    return {
        "model": MODEL_ID, "revision": REVISION, "training_seed": TRAINING_SEED,
        "training_templates": list(TEMPLATES[:2]), "test_templates": {"familiar": 0, "novel": 2},
        "choice_suffix": CHOICE_SUFFIX, "draws": draws,
    }, edl


def build_study_masks(checkpoint):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from run_scalar_sweep import eligible_modules, sample_masks

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    modules = eligible_modules(model)
    masks = {}
    for seed in MASK_SEEDS:
        total, by_budget = sample_masks(modules, BUDGETS, seed)
        coordinates = {budget: {(name, index) for name, indices in mask.items() for index in indices}
                       for budget, mask in by_budget.items()}
        if any(len(coordinates[budget]) != budget for budget in BUDGETS) or not coordinates[1000] < coordinates[2000]:
            raise ValueError(f"Mask seed {seed} does not produce nested exact budgets.")
        masks[seed] = by_budget
    return masks, {"eligible_coordinates": total, "modules": dict(modules)}


def prepare_study(root, checkpoint):
    with study_lock(root):
        manifests = root / "manifests"
        if manifests.exists():
            print(f"Verified existing frozen manifests ({verify_inputs(manifests, checkpoint)}).", flush=True)
            return
        protocol, edl = build_study_data()
        masks, geometry = build_study_masks(checkpoint)
        with TemporaryDirectory(prefix=".prepare-", dir=root) as scratch:
            staged = Path(scratch) / "manifests"
            staged.mkdir()
            files = {
                "protocol.json": protocol, "masks.json": masks, "edl-inputs.json": edl,
                "checkpoint.json": {name: sha256(checkpoint / name) for name in CHECKPOINT_FILES},
                "settings.json": study_settings(), "analysis.json": ANALYSIS, "mask-geometry.json": geometry,
            }
            for name, content in files.items():
                write_json(staged / name, content)
            write_json(staged / "sha256.json", tree_hashes(staged))
            inputs_hash = verify_inputs(staged, checkpoint)
            staged.rename(manifests)
        print(f"Frozen {len(files)} manifests at {manifests} ({inputs_hash}).", flush=True)


def verify_inputs(manifests, checkpoint):
    index = manifests / "sha256.json"
    if not index.is_file():
        raise ValueError(f"Frozen study inputs are required at {index}.")
    hashes = read_json(index)
    if not set(MANIFEST_FILES) <= hashes.keys() or any(Path(name).name != name for name in hashes):
        raise ValueError("The study checksum index must name all required manifest files in its directory.")
    for name, expected in hashes.items():
        if sha256(manifests / name) != expected:
            raise ValueError(f"Frozen manifest checksum mismatch for {name}.")
    protocol = read_json(manifests / "protocol.json")
    if ((protocol["model"], protocol["revision"], protocol["training_seed"]) != (MODEL_ID, REVISION, TRAINING_SEED)
            or set(map(int, protocol["draws"])) != set(RULEBOOK_SEEDS)):
        raise ValueError("The frozen protocol does not match the study checkpoint, training seed, or rulebook seeds.")
    checkpoint_hashes = read_json(manifests / "checkpoint.json")
    if set(checkpoint_hashes) != set(CHECKPOINT_FILES):
        raise ValueError("The frozen checkpoint inventory does not name the expected model and tokenizer files.")
    if checkpoint_hashes["model.safetensors"] != PINNED_WEIGHTS_SHA256:
        raise ValueError(f"The weights do not match the published checksum for {MODEL_ID}@{REVISION}.")
    for name, expected in checkpoint_hashes.items():
        if sha256(checkpoint / name) != expected:
            raise ValueError(f"Pinned checkpoint checksum mismatch for {name}.")
    if read_json(manifests / "settings.json") != study_settings():
        raise ValueError("Runtime, training scripts, or job settings differ from the frozen study.")
    if read_json(manifests / "analysis.json") != ANALYSIS:
        raise ValueError("Analysis definitions differ from the frozen study.")
    return sha256(index)


@contextmanager
def study_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("ab+") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another runner is using {root}.") from error
        yield


def run_job(folder, job, command, inputs_hash, script_hashes, dependency_hashes, *, cwd=SCRIPTS.parent):
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "completed.json"
    request = {
        "command_without_output": command, "fits": job.fits, "inputs_sha256": inputs_hash,
        "training_script_sha256": script_hashes, "dependency_receipt_sha256": dependency_hashes,
    }
    if cwd != SCRIPTS.parent:
        request["working_directory"] = str(cwd)
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        attempt = folder / receipt["attempt"]
        payload = attempt / "results"
        if (receipt["request"] != request or not set(job.outputs) <= receipt["artifacts"].keys()
                or tree_hashes(payload) != receipt["artifacts"]
                or sha256(attempt / "run.log") != receipt["log_sha256"]):
            raise ValueError(f"The completed job at {folder} has different inputs or damaged artifacts.")
        print(f"Skipping {job.name}, verified complete.", flush=True)
        return payload

    attempts = sorted(folder.glob("attempt-*"))
    failures = 0
    for attempt in attempts:
        previous = attempt / "request.json"
        if previous.is_file() and read_json(previous) != request:
            raise ValueError(f"The earlier attempt at {attempt} used different inputs or settings.")
        failures += (attempt / "failure.json").is_file()

    next_number = max((int(attempt.name.removeprefix("attempt-")) for attempt in attempts), default=0) + 1
    while failures < 2:
        attempt = folder / f"attempt-{next_number:03}"
        attempt.mkdir()
        write_json(attempt / "request.json", request)
        payload = attempt / "results"
        log_path = attempt / "run.log"
        print(f"Starting {job.name} ({job.fits} fits). Log at {log_path}", flush=True)
        try:
            with log_path.open("wb") as log:
                subprocess.run([*command, "--output", str(payload)], cwd=cwd,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            artifacts = tree_hashes(payload)
            if missing := sorted(set(job.outputs) - artifacts.keys()):
                raise ValueError(f"Missing expected output files: {', '.join(missing)}")
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            write_json(attempt / "failure.json", {"error": str(error)})
            print(f"Attempt {next_number} failed for {job.name}. Log at {log_path}", flush=True)
            failures += 1
            next_number += 1
            continue
        write_json(receipt_path, {
            "attempt": attempt.name, "request": request, "artifacts": artifacts, "log_sha256": sha256(log_path),
        })
        print(f"Completed {job.name}.", flush=True)
        return payload

    raise RuntimeError(f"Both attempts for {job.name} failed. Logs are in {folder}.")


def run_study(root, draws, checkpoint):
    manifests = root / "manifests"
    inputs_hash = verify_inputs(manifests, checkpoint)

    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The confirmatory runs require CUDA with bfloat16 support.")
    script_hashes = {name: sha256(SCRIPTS / name) for name in TRAINING_SCRIPTS}
    with study_lock(root):
        for draw in draws:
            if draw != RULEBOOK_SEEDS[0]:
                from verify_confirmatory import read_verified_summary

                pilot = RULEBOOK_SEEDS[0]
                verified_path = root / "verification" / f"rulebook-{pilot}.json"
                if not verified_path.is_file():
                    raise RuntimeError(f"Verify rulebook {pilot} before starting another draw.")
                verified = read_verified_summary(verified_path)
                pilot_jobs = list(jobs_for(pilot, manifests, checkpoint))
                pilot_runs = root / "runs" / f"rulebook-{pilot}"
                if (verified["inputs_sha256"] != inputs_hash or verified["fit_count"] != sum(j.fits for j in pilot_jobs)
                        or verified["job_receipt_sha256"] != {j.name: sha256(pilot_runs / j.name / "completed.json")
                                                               for j in pilot_jobs}):
                    raise RuntimeError(f"Rulebook {pilot} verification does not cover the current frozen study.")
            run_root = root / "runs" / f"rulebook-{draw}"
            completed = {}
            for job in jobs_for(draw, manifests, checkpoint):
                command = [sys.executable, "-B", "-u", str(SCRIPTS / job.script), *job.args]
                dependency_hashes = {}
                for flag, name in job.dependencies:
                    command.extend((flag, str(completed[name])))
                    dependency_hashes[name] = sha256(run_root / name / "completed.json")
                folder = run_root / job.name
                completed[job.name] = run_job(folder, job, command, inputs_hash, script_hashes, dependency_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    prepare = stages.add_parser("prepare", help="Freeze data, masks, runtime settings, and analysis definitions.")
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--checkpoint", type=Path, required=True, help="Local copy of the pinned Qwen checkpoint.")
    run = stages.add_parser("run", help="Train selected rulebooks or print the job plan.")
    run.add_argument("--draw", type=int, nargs="+", choices=RULEBOOK_SEEDS, required=True)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--checkpoint", type=Path, help="Local copy of the pinned Qwen checkpoint.")
    run.add_argument("--dry-run", action="store_true", help="List jobs without training or creating files.")
    verify = stages.add_parser("verify", help="Independently recompute metrics and verify saved job artifacts.")
    verify.add_argument("--draw", type=int, nargs="+", choices=RULEBOOK_SEEDS, required=True)
    verify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--checkpoint", type=Path, required=True, help="Local copy of the pinned Qwen checkpoint.")
    replay = stages.add_parser("replay", help="Repeat the prespecified sparse pair from a clean local checkout.")
    replay.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    replay.add_argument("--checkpoint", type=Path, required=True, help="Local copy of the pinned Qwen checkpoint.")
    report = stages.add_parser("report", help="Generate tables and figures after study and replay verification.")
    report.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    package = stages.add_parser("package", help="Archive the verified runs, analysis, code, and environment pins.")
    package.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    package.add_argument("--archive", type=Path, help="Destination ZIP path. Defaults to the study directory.")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage == "prepare":
        prepare_study(root, args.checkpoint.resolve())
        return
    if args.stage == "replay":
        from replay_confirmatory import replay_study

        replay_study(root, args.checkpoint.resolve())
        return
    if args.stage == "report":
        from report_confirmatory import report_study

        report_study(root)
        return
    if args.stage == "package":
        from package_confirmatory import ARCHIVE_NAME, package_study

        archive = args.archive.resolve() if args.archive else root / ARCHIVE_NAME
        package_study(root, archive)
        return
    if len(set(args.draw)) != len(args.draw):
        parser.error("List each rulebook seed at most once.")
    if args.stage == "verify":
        from verify_confirmatory import verify_study

        verify_study(root, args.draw, args.checkpoint.resolve())
        return
    if args.dry_run:
        count = 0
        for draw in args.draw:
            for job in jobs_for(draw, root / "manifests", args.checkpoint):
                dependencies = ", ".join(name for _, name in job.dependencies)
                after = f" after {dependencies}" if dependencies else ""
                print(f"Rulebook {draw} / {job.name}: {job.fits} fits{after}")
                count += job.fits
        print(f"Total: {count} training fits")
        return
    if args.checkpoint is None:
        parser.error("A local --checkpoint is required for training.")
    run_study(root, args.draw, args.checkpoint.resolve())


if __name__ == "__main__":
    main()
