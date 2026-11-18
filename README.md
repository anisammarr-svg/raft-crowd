# RAFT-CrowdFlow

A reproducible starting implementation for motion-based video anomaly analysis using frozen RAFT optical flow, motion segmentation and Crowd Flow Instability (CFI) cues.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anisammarr-svg/raft-crowd/blob/main/output/jupyter-notebook/RAFTCrowdFlow_from_scratch.ipynb)

**Status:** 115 synthetic CPU tests passed locally. Pretrained RAFT/GPU inference and real-dataset performance have not been validated. This implementation does not reproduce historical paper scores or establish crowd-crush prediction, physical pressure, early warning or real-time deployment.

## Start here

1. Open the notebook using the Colab button above and select a GPU runtime.
2. Run the setup and synthetic-check cells.
3. Supply your own benchmark frames, true binary annotations and train/validation/test manifest.
4. Extract flow, calibrate on training/validation data and evaluate the frozen methods on test data.

For VS Code, open the notebook and select **Select Kernel > Colab > Auto Connect** using the official Google Colab extension. Sign in when prompted. The file-upload cell targets the browser Colab UI; in VS Code, use files already accessible to the remote runtime and set `MANIFEST` explicitly.

Read [the full setup and usage guide](RAFTCROWD_README.md) before a benchmark run.

## Repository layout

```text
src/raftcrowd/                 RAFT adapter, cues, metrics, pipeline, CLI, visualization
configs/                      Explicit defaults and example data manifest
scripts/                      CLI entry point and notebook build/check tools
scripts/templates/            Notebook source template
tests/                        Synthetic mathematics, split, metric and pipeline tests
docs/                         Method specification and limitations
plan/                         Prospective experiment protocol
tables/                       Result-table schema
figures/                      Figure data/provenance schema
output/jupyter-notebook/       Standalone notebook, with embedded source checksums
requirements-raftcrowd.txt     Model/inference dependency ranges
requirements-dev.txt          Lightweight CPU test/notebook-check dependencies
```

Datasets, labels, videos, checkpoints, caches and experiment outputs are not included. `data/` and `results/` are local runtime directories excluded from Git.

## Command-line workflow

Install a compatible PyTorch/Torchvision pair for your GPU, then:

```bash
python -m pip install -r requirements-raftcrowd.txt
python scripts/raftcrowd.py --help
python scripts/raftcrowd.py extract --manifest configs/raftcrowd_manifest.example.csv --config configs/raftcrowd.json --out results/raftcrowd/cache
python scripts/raftcrowd.py calibrate --cache results/raftcrowd/cache --out results/raftcrowd/model.json
python scripts/raftcrowd.py evaluate --cache results/raftcrowd/cache --model results/raftcrowd/model.json --out results/raftcrowd/evaluation
python scripts/raftcrowd.py visualize --cache results/raftcrowd/cache --model results/raftcrowd/model.json --video-id test001 --frame-index 1 --out results/raftcrowd/panel.png
```

Edit the manifest paths first; it contains no downloaded data. Defaults use motion-only CFI. Density coupling requires externally supplied, aligned pedestrian-density maps. Blurred optical flow is not used as pedestrian density. Scientific outputs are exclusive-create: use a new path for each run.

The flow-magnitude baseline, individual cues, leave-one-cue-out variants and no-segmentation control share the same input pairs. Normalization uses training data; operational thresholds use validation labels; test evaluation keeps both fixed. No anomaly network is trained.

## Lightweight verification

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=src python -m pytest -q tests
python scripts/check_raftcrowd_notebook.py
```

On PowerShell set `$env:PYTHONPATH = 'src'` before running pytest. The tests use synthetic flows and an injected estimator; they do not require PyTorch or prove scientific performance. Dependency files declare ranges, not a frozen benchmark environment.

To update the standalone notebook after editing its canonical sources:

```bash
python scripts/build_raftcrowd_notebook.py
python scripts/check_raftcrowd_notebook.py
```

The builder also produces a local ZIP archive; that duplicate archive is excluded from Git. Keep LF line endings so embedded source checksums remain valid across operating systems.

## Method and protocol

- [Architecture and explicit implementation choices](docs/raftcrowd-method.md)
- [Prospective experiment protocol](plan/raftcrowd-experiment-protocol.md)
- [Result-table schema](tables/raftcrowd-table-schema.md)
- [Figure provenance](figures/raftcrowd-data-manifest.md)
- [Official Torchvision RAFT API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html)

No dataset or third-party pretrained weights are redistributed in this repository. Check the respective dataset/model access and licensing terms before use.
