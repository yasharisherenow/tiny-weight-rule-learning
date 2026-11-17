# Reproducing the confirmatory study

The repository includes [the reference tables and figures](results/confirmatory/report/), [frozen inputs](results/confirmatory/manifests/), and [verification summaries](results/confirmatory/verification/). You can inspect those directly without a checkpoint or GPU.

## Training environment

The verified study used Windows, Python 3.13.15, Git, and one NVIDIA RTX 4060 with CUDA 12.8 support. The base model runs in bfloat16. Run these PowerShell commands from the repository root with Python 3.13.15 installed.

```powershell
py -3.13 -m venv .venv
$train = (Resolve-Path .venv/Scripts/python.exe).Path
& $train -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.9.1+cu128'
& $train -m pip install -r requirements/training.txt
```

The checkpoint is [Qwen2.5 0.5B at revision `060db6499f32faf8b98477b0a26969ef7d8b9987`](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987). Download the exact files listed in the manifest into an ignored local directory.

```powershell
& $train -B -c "import json; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', revision='060db6499f32faf8b98477b0a26969ef7d8b9987', allow_patterns=list(json.load(open('results/confirmatory/manifests/checkpoint.json', encoding='utf-8'))), local_dir='checkpoint')"
$checkpoint = (Resolve-Path checkpoint).Path
```

## Run the study

Commands below use the default output directory, `output/experiments/qwen-05b-confirmatory/`. To use a separate directory, pass the same `--output` path to every stage.

```powershell
& $train -B experiments/run_confirmatory.py run --dry-run --draw 90101 90102 90106
& $train -B experiments/run_confirmatory.py prepare --checkpoint $checkpoint

$original = (Get-FileHash results/confirmatory/manifests/sha256.json).Hash
$generated = (Get-FileHash output/experiments/qwen-05b-confirmatory/manifests/sha256.json).Hash
if ($original -ne $generated) { throw 'Frozen inputs differ from the reference study.' }

& $train -B experiments/run_confirmatory.py run --draw 90101 --checkpoint $checkpoint
& $train -B experiments/run_confirmatory.py verify --draw 90101 --checkpoint $checkpoint
& $train -B experiments/run_confirmatory.py run --draw 90102 90106 --checkpoint $checkpoint
& $train -B experiments/run_confirmatory.py verify --draw 90101 90102 90106 --checkpoint $checkpoint
& $train -B experiments/run_confirmatory.py replay --checkpoint $checkpoint
```

The first rulebook's verification gates the remaining queue. The verifier accounts for 24, 24, and 10 fits across the three draws. Replay repeats the two 2,000-weight fits for draw 90101 and mask 123 from a clean Git checkout. The reference replay scored 114/128 with familiar wording and 23/128 with unseen wording at 432 steps.

## Regenerate tables and figures

The figures used a separate Python environment with the versions in [`requirements/figures.txt`](requirements/figures.txt).

```powershell
py -3.13 -m venv .venv-figures
$plot = (Resolve-Path .venv-figures/Scripts/python.exe).Path
& $plot -m pip install -r requirements/figures.txt
& $plot -B experiments/run_confirmatory.py report
```

The new report appears under `output/experiments/qwen-05b-confirmatory/report/`. Compare the tables with the tracked reference report. The run and replay receipts contain local paths and run-specific metadata.

## Verify or rebuild the full archive

Git tracks the compact research artifacts. Full logs, raw predictions and losses, and adapter tensors live in the generated run directory. With the complete run and report in place, build and verify the archive.

```powershell
& $train -B experiments/run_confirmatory.py package
& $train -B experiments/package_confirmatory.py --verify output/experiments/qwen-05b-confirmatory/tiny-weight-rule-learning-confirmatory.zip
```

An existing archive contains its own README, checksums, raw artifacts, and commands to verify and regenerate the report from those artifacts.

## Rebuild the paper

Every figure used by [the manuscript](paper/draft.md) is tracked. The PDF renderer uses the Times New Roman files from Windows, with an optional `--font-dir` override.

```powershell
& $plot -m pip install -r requirements/paper.txt
& $plot -B paper/render_manuscript.py paper/draft.md output/pdf/tiny-weight-rule-learning.pdf
```
