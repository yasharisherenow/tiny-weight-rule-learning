"""Replay the prespecified sparse pair with an auditable clean Git checkout."""

import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from run_confirmatory import (
    SCRIPTS, TRAINING_SCRIPTS, jobs_for, read_json, run_job, sha256, study_lock, tree_hashes, write_json,
)
from verify_confirmatory import check_job, check_sparse, load_verified_draws, paired, require, verify_design


DRAW = 90101
MASK_SEED = 123
BUDGET = 2000
FINAL_STEP = "432"
JOB = f"sparse-mask-{MASK_SEED}-budget-{BUDGET}"


def git(cwd, *args):
    return subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.hooksPath=NUL", *map(str, args)],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def prepare_checkout(snapshot, manifests):
    if not snapshot.exists():
        with TemporaryDirectory(prefix=".snapshot-", dir=snapshot.parent) as scratch:
            work = Path(scratch)
            source, staged = work / "source", work / "snapshot"
            source.mkdir()
            staged.mkdir()
            (source / "experiments").mkdir()
            for filename in (*TRAINING_SCRIPTS, "run_confirmatory.py", "verify_confirmatory.py", "replay_confirmatory.py"):
                shutil.copyfile(SCRIPTS / filename, source / "experiments" / filename)
            shutil.copytree(manifests, source / "manifests")
            (source / ".gitattributes").write_bytes(b"* -text\n")
            git(source, "init", "-q", "-b", "main")
            git(source, "add", "--", ".gitattributes", "experiments", "manifests")
            git(source, "-c", "user.name=Tiny Weight Rule Learning", "-c", "user.email=local-replay@localhost",
                "-c", "commit.gpgsign=false", "commit", "-qm", "Freeze confirmatory replay code and inputs")
            bundle = staged / "source.bundle"
            git(source, "bundle", "create", bundle, "refs/heads/main", "HEAD")
            git(staged, "clone", "--quiet", bundle, staged / "checkout")
            git(staged / "checkout", "remote", "set-url", "origin", "../source.bundle")
            staged.rename(snapshot)
    return snapshot / "checkout"


def check_checkout(snapshot, checkout, manifests, settings):
    require(not git(checkout, "status", "--porcelain", "--untracked-files=all"), "The replay checkout is not clean.")
    commit = git(checkout, "rev-parse", "HEAD")
    bundle_refs = dict(line.split(maxsplit=1)[::-1] for line in git(snapshot, "bundle", "list-heads", "source.bundle").splitlines())
    require(bundle_refs.get("HEAD") == commit and bundle_refs.get("refs/heads/main") == commit,
            "The replay checkout and archived Git bundle refer to different commits.")
    require(tree_hashes(checkout / "manifests") == tree_hashes(manifests), "The checkout changed the frozen manifests.")
    actual_scripts = {name: sha256(checkout / "experiments" / name) for name in TRAINING_SCRIPTS}
    require(actual_scripts == settings["training_script_sha256"], "The replay uses different training code.")
    return commit


def replay_study(root, checkpoint):
    import torch

    manifests = root / "manifests"
    replay = root / "replay"
    with study_lock(root):
        inputs_hash, draws = load_verified_draws(root)
        require(verify_design(manifests, checkpoint) == inputs_hash, "The verified study inputs changed.")
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "The replay requires CUDA bfloat16.")
        replay.mkdir(exist_ok=True)
        snapshot = replay / "snapshot"
        checkout = prepare_checkout(snapshot, manifests)
        settings = read_json(manifests / "settings.json")
        commit = check_checkout(snapshot, checkout, manifests, settings)
        checkout_manifests = checkout / "manifests"
        require(verify_design(checkout_manifests, checkpoint) == inputs_hash, "The replay design differs from the study.")
        job = next(job for job in jobs_for(DRAW, checkout_manifests, checkpoint) if job.name == JOB)
        command = [sys.executable, "-B", "-u", str(checkout / "experiments" / job.script), *job.args]
        run_folder = replay / "run"
        run_job(run_folder / job.name, job, command, inputs_hash, settings["training_script_sha256"], {}, cwd=checkout)
        require(check_checkout(snapshot, checkout, manifests, settings) == commit,
                "Replay checkout changed while training.")
        result, receipt = check_job(run_folder, job, inputs_hash, settings, {})
        protocol_path = manifests / "protocol.json"
        raw_draw = read_json(protocol_path)["draws"][str(DRAW)]
        masks = read_json(manifests / "masks.json")
        verified = check_sparse(result, raw_draw, DRAW, MASK_SEED, BUDGET, masks, settings, sha256(protocol_path))
        original = next(item for item in draws[DRAW]["sparse"] if (item["mask_seed"], item["budget"]) == (MASK_SEED, BUDGET))
        original_job = root / "runs" / f"rulebook-{DRAW}" / job.name
        original_receipt = read_json(original_job / "completed.json")
        budget_file = f"rulebook-{DRAW}/budget-{BUDGET}.json"
        original_fixed = read_json(original_job / original_receipt["attempt"] / "results" / budget_file)["fixed_epoch_evaluations"]
        replay_fixed = read_json(result / budget_file)["fixed_epoch_evaluations"]
        scores = {name: {
            "original": original["points"][FINAL_STEP][name]["disagreement"]["both_correct"],
            "replay": verified["points"][FINAL_STEP][name]["disagreement"]["both_correct"],
        } for name in ("familiar", "novel")}
        match = all(item["original"] == item["replay"] for item in scores.values())
        proof = {
            "rulebook": DRAW, "mask_seed": MASK_SEED, "budget": BUDGET, "fit_count": 2,
            "optimizer_steps": int(FINAL_STEP), "inputs_sha256": inputs_hash, "git_commit": commit,
            "source_bundle_sha256": sha256(snapshot / "source.bundle"),
            "replay_job_receipt_sha256": receipt, "original_job_receipt_sha256": sha256(original_job / "completed.json"),
            "final_paired_disagreement": scores, "final_paired_counts_match": match,
            "fixed_evaluation_logits_identical": all(replay_fixed[epoch][key] == original_fixed[epoch][key]
                                                     for epoch in replay_fixed
                                                     for key in ("test_scores", "novel_wording_test_scores")),
        }
        write_json(replay / "verification.json", proof)
        require(match, f"The clean-checkout replay produced different final paired counts: {scores}.")
        print(f"Verified 2 replay fits at clean commit {commit}. Final paired counts: {scores}.", flush=True)


def load_verified_replay(root, inputs_hash, draws):
    replay = root / "replay"
    proof = read_json(replay / "verification.json")
    require((proof["rulebook"], proof["mask_seed"], proof["budget"], proof["optimizer_steps"], proof["fit_count"])
            == (DRAW, MASK_SEED, BUDGET, int(FINAL_STEP), 2) and proof["inputs_sha256"] == inputs_hash
            and proof["final_paired_counts_match"] and proof["original_job_receipt_sha256"] == draws[DRAW]["job_receipt_sha256"][JOB],
            "The clean-checkout replay does not cover the verified study target.")
    require(sha256(replay / "snapshot" / "source.bundle") == proof["source_bundle_sha256"],
            "The replay source bundle has changed since verification.")
    manifests = root / "manifests"
    job = next(job for job in jobs_for(DRAW, manifests, None) if job.name == JOB)
    result, receipt = check_job(replay / "run", job, inputs_hash, read_json(manifests / "settings.json"), {})
    require(receipt == proof["replay_job_receipt_sha256"], "The replay artifacts have changed since verification.")
    protocol = read_json(manifests / "protocol.json")["draws"][str(DRAW)]
    original_job = root / "runs" / f"rulebook-{DRAW}" / JOB
    attempt = read_json(original_job / "completed.json")["attempt"]
    budget_file = f"rulebook-{DRAW}/budget-{BUDGET}.json"
    original = read_json(original_job / attempt / "results" / budget_file)["fixed_epoch_evaluations"]
    repeated = read_json(result / budget_file)["fixed_epoch_evaluations"]
    same_logits = all(repeated[epoch][key] == original[epoch][key] for epoch in original
                      for key in ("test_scores", "novel_wording_test_scores"))
    require(same_logits == proof["fixed_evaluation_logits_identical"], "Replay logit equality differs from its proof.")
    for name, key in (("familiar", "test_scores"), ("novel", "novel_wording_test_scores")):
        counts = {}
        for label, evaluations in (("original", original), ("replay", repeated)):
            at = evaluations["18"]
            require(at["optimizer_steps"] == [432, 432], "The replay comparison used a different optimizer step.")
            score = paired(protocol["examples"]["test"], protocol["rulebooks"], at[key])
            counts[label] = score["disagreement"]["both_correct"]
        require(counts == proof["final_paired_disagreement"][name] and counts["original"] == counts["replay"],
                "The replay paired counts do not match raw logits and the original study.")
    return proof
