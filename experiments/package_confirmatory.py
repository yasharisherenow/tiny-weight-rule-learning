"""Archive the confirmatory artifacts and verify every archived file by SHA-256."""

import argparse
import csv
import hashlib
import json
from importlib.metadata import distributions
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from replay_confirmatory import load_verified_replay
from report_confirmatory import build_tables, markdown_tables
from run_confirmatory import MODEL_ID, REVISION, SCRIPTS, read_json, sha256, study_settings
from verify_confirmatory import load_verified_draws, require


ARCHIVE_NAME = "tiny-weight-rule-learning-confirmatory.zip"
CHECKSUMS = "archive-checksums.json"


def verify_archive(path):
    with ZipFile(path) as archive:
        checksums = json.loads(archive.read(CHECKSUMS))
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == {CHECKSUMS, *checksums},
                "The archive inventory differs from its checksum index.")
        for member, expected in checksums.items():
            with archive.open(member) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            require(actual == expected, f"Archive checksum mismatch for {member}.")
    return len(checksums)


def write_archive(path, sources, records):
    require(not set(sources) & records.keys() and CHECKSUMS not in sources and CHECKSUMS not in records,
            "Archive member names overlap.")
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    checksums = {name: sha256(file) for name, file in sources.items()}
    checksums.update({name: hashlib.sha256(content).hexdigest() for name, content in records.items()})
    with ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for name, file in sorted(sources.items()):
            archive.write(file, name)
        for name, content in sorted(records.items()):
            archive.writestr(name, content)
        archive.writestr(CHECKSUMS, json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    count = verify_archive(staged)
    staged.replace(path)
    return count


def check_report(root, verified, replay, inputs_hash):
    report = root / "report"
    summary = read_json(report / "summary.json")
    tables = build_tables(verified)
    require((summary["inputs_sha256"], summary["study_fits"], summary["replay_fits"], summary["replay"])
            == (inputs_hash, 58, 2, replay)
            and summary["sparse_draw_ranges"] == tables["sparse-draw-ranges"]
            and summary["edl_conditions"] == tables["edl-conditions"] and summary["edl_contrasts"] == tables["edl-contrasts"],
            "The report summary differs from the verified run and replay.")
    for name, rows in tables.items():
        with (report / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            require(list(csv.DictReader(stream)) == [{k: str(v) for k, v in row.items()} for row in rows],
                    f"Table {name} differs from verified records.")
    require((report / "tables.md").read_text(encoding="utf-8") == markdown_tables(tables),
            "The Markdown tables differ from verified records.")
    for name in ("sparse-duration.png", "edl-conditions.png"):
        with (report / name).open("rb") as stream:
            require(stream.read(8) == b"\x89PNG\r\n\x1a\n", f"Expected a rendered PNG for {name}.")
    return summary["report_runtime"]


def reproduction_notes(settings, report_runtime):
    return f"""# Tiny Weight Rule Learning confirmatory study

The archive contains the 58 study fits and 2 clean-checkout replay fits. The study results in `study/report` were generated from verified raw artifacts. `study/manifests/sha256.json` identifies the frozen inputs and `archive-checksums.json` covers every archived member. Per-job attempt logs and receipts are alongside the predictions. `study/replay/snapshot/source.bundle` preserves the clean replay commit and its exact training sources and inputs.

## Environment and checkpoint

The training runtime used Python {settings['python']} on Windows, CUDA 12.8, bfloat16, and a single RTX 4060. Install the pinned CUDA build first, then the other recorded dependencies. Run the following from the extracted directory in a Python {settings['python']} virtual environment.

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch=={settings['packages']['torch']}
python -m pip install -r requirements-training.txt
```

Set `$checkpoint` to a local copy of {MODEL_ID} at revision `{REVISION}`. The manifest lists the exact seven files and their hashes. The following downloads that inventory into an adjacent `checkpoint` directory.

```powershell
python -c "import json; from huggingface_hub import snapshot_download; snapshot_download('{MODEL_ID}', revision='{REVISION}', allow_patterns=list(json.load(open('study/manifests/checkpoint.json', encoding='utf-8'))), local_dir='checkpoint')"
$checkpoint = (Resolve-Path checkpoint).Path
```

## Check archived outputs

The archive can be hashed without extracting it. Run the training verifier after extracting, with the pinned environment and checkpoint. The three verification lines account for 24, 24, and 10 study fits.

```powershell
python -B experiments/package_confirmatory.py --verify ./{ARCHIVE_NAME}
python -B experiments/run_confirmatory.py verify --output ./study --draw 90101 90102 90106 --checkpoint $checkpoint
```

## Repeat the fits

Use a fresh output directory and keep the steps in order. The verifier after the first rulebook is the rollout gate for the other two. Existing valid job receipts can be resumed in the same output directory with the same absolute paths.

```powershell
python -B experiments/run_confirmatory.py prepare --output ./fresh --checkpoint $checkpoint
Get-FileHash ./study/manifests/sha256.json, ./fresh/manifests/sha256.json
python -B experiments/run_confirmatory.py run --output ./fresh --draw 90101 --checkpoint $checkpoint
python -B experiments/run_confirmatory.py verify --output ./fresh --draw 90101 --checkpoint $checkpoint
python -B experiments/run_confirmatory.py run --output ./fresh --draw 90102 90106 --checkpoint $checkpoint
python -B experiments/run_confirmatory.py verify --output ./fresh --draw 90102 90106 --checkpoint $checkpoint
python -B experiments/run_confirmatory.py replay --output ./fresh --checkpoint $checkpoint
```

The two `Get-FileHash` digests should match. Replay preserves the Git commit, raw files, and fixed 432-step paired count comparison in `fresh/replay/verification.json`. Inspect any endpoint comparability flags as scientific results. The study integrity gates still apply when an endpoint flag is false.

## Rebuild the analysis

The plots were created with Python {report_runtime['python']} and the versions in `requirements-figures.txt`. Use this separate plotting environment for either `./study` or `./fresh` after replay verification.

```powershell
python -m pip install -r requirements-figures.txt
python -B experiments/run_confirmatory.py report --output ./study
```
"""


def package_study(root, archive):
    manifests = root / "manifests"
    settings = read_json(manifests / "settings.json")
    require(study_settings() == settings, "Build the archive using the recorded training environment.")
    inputs_hash, draws = load_verified_draws(root)
    replay = load_verified_replay(root, inputs_hash, draws)
    report_runtime = check_report(root, draws, replay, inputs_hash)
    sources = {f"experiments/{script.name}": script for script in SCRIPTS.glob("*.py")}
    for folder in ("manifests", "runs", "verification", "report", "replay/run"):
        for file in (root / folder).rglob("*"):
            if file.is_file():
                sources[f"study/{file.relative_to(root).as_posix()}"] = file
    for file in (root / "replay" / "verification.json", root / "replay" / "snapshot" / "source.bundle",
                 *root.glob("progress.md"), *root.glob("queue-*.log")):
        sources[f"study/{file.relative_to(root).as_posix()}"] = file
    installed = {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]}
    training = "\n".join(f"{name}=={value}" for name, value in sorted(installed.items(), key=lambda p: p[0].lower())
                         if name.lower() not in {"torch", "pip", "bitsandbytes"}) + "\n"
    figures = "\n".join(f"{key}=={report_runtime[key]}" for key in ("matplotlib", "numpy")) + "\n"
    records = {"README.md": reproduction_notes(settings, report_runtime).encode("utf-8"),
               "requirements-training.txt": training.encode("utf-8"), "requirements-figures.txt": figures.encode("utf-8"),
               "training-environment.json": (json.dumps(installed, indent=2, sort_keys=True) + "\n").encode("utf-8")}
    count = write_archive(archive, sources, records)
    print(f"Verified {count} archived files with SHA-256. Archive at {archive}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", required=True, type=Path, help="Verify the contents of an existing archive.")
    args = parser.parse_args()
    print(f"Archive verified: {verify_archive(args.verify)} files.")
