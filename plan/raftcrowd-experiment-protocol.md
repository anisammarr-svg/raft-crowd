# RAFT-CrowdFlow prospective experiment protocol

Date: 2026-11-18. Phase: method implementation and experiment planning. This protocol concerns the new RAFT video-anomaly implementation, not the SDD trajectory project documented in `experiment-protocol.md`. No benchmark performance is recorded here.

## Scope and preregistered choices

Question: does a specified combination of compression and regional motion variability improve frame-level anomaly discrimination over a matched flow-magnitude score? Pedestrian-density coupling is an optional, separately declared study requiring external density maps. Crush prediction, prevention, physical pressure and spatial risk localization are not validated tasks in this protocol.

The initial method specification is in `../docs/raftcrowd-method.md`. Freeze configuration, cue list, equal fusion weights, moving threshold, clustering seed, training normalization, aggregation rule, datasets and split manifest before test evaluation. A change after inspecting test results creates an exploratory variant; it does not recover an untouched holdout. The paper's historical test exposure must be disclosed if the same videos are reused.

## Data manifest and exclusions

Use a CSV with columns `id,split,group,frames,labels,density`. Each row identifies a video represented as an ordered directory of frames. Valid split values are `train`, `val` and `test`.

| Field | Meaning |
| --- | --- |
| `id` | Unique stable video identifier, safe for output naming |
| `split` | Role fixed before scoring; all three roles required for the full workflow |
| `group` | Leakage-prevention grouping such as recording, camera or scene, chosen according to the study |
| `frames` | Directory containing sorted PNG/JPG frames with zero-padded names and no duplicate frame stems |
| `labels` | NumPy vector of N binary frame labels in that exact order; required for validation/test; training may omit labels only when all frames are declared normal |
| `density` | Optional directory of aligned source-frame pedestrian-density arrays; required when pressure is enabled |

Declare how groups were constructed and verify that train/validation/test groups are disjoint. Do not silently mix normal and abnormal excerpts from the same original recording across splits. Duplicate frame directories must be rejected. Group checks cannot prove independence if authors assign misleading or incomplete groups; inspect original recording provenance as well.

Record dataset release/version and access terms, official split correspondence or any departures, frame extraction cadence, image ordering, label source, exclusions and input hashes. Frame decoding/extraction must preserve every intended frame in order; video codec metadata alone is not proof of alignment. Exclude frame zero from pair-score metrics in every method. Stop on malformed labels, fewer than two frames, absent required inputs or density-coordinate mismatches rather than dropping rows.

## Fixed preprocessing and method

- Default resized shape: H=360, W=640; RGB uint8 input; consecutive frame pairs; numeric flow cropped after padding.
- Frozen RAFT Large weights: `Raft_Large_Weights.C_T_V1`; 12 updates; batch 1; fp32 default. Any mixed-precision variant has its own configuration and result identity.
- Proposed defaults: Gaussian flow smoothing sigma=1; moving magnitude threshold=0.5 resized pixels/pair; at most four motion groups; deterministic seed 7.
- Cues: nonnegative compression, circular directional variance and regional magnitude variance. Pressure is absent unless valid external pedestrian density is explicitly enabled.
- Fit per-cue scaling from deterministic bounded samples of training maps only, using a zero lower bound and 99th-percentile upper scale. Freeze scales before validation/test scoring; preserve the sampling procedure and counts.
- Full active CFI uses fixed equal cue weights. The frame score is the mean of the largest 1% of pixels over the whole map.
- RAFT weights do not train; optimizer, learning rate, epochs, early stopping and model-selection checkpoints are not applicable. Calibration outputs and threshold locks replace trainable anomaly-model artifacts.

The starting JSON configuration additionally records `kmeans_iterations=20`, `kmeans_samples=4096`, `normalization_samples_per_frame=128` and `normalization_max_samples=50000`. The density mode is explicitly `none` or `external`; a missing density input must not be silently replaced by motion magnitude.

## Executable workflow and artifact layout

From the coordination repository root, after providing actual frames and editing the example manifest:

```bash
python scripts/raftcrowd.py extract --manifest configs/raftcrowd_manifest.example.csv --config configs/raftcrowd.json --out results/raftcrowd/cache
python scripts/raftcrowd.py calibrate --cache results/raftcrowd/cache --out results/raftcrowd/model.json
python scripts/raftcrowd.py evaluate --cache results/raftcrowd/cache --model results/raftcrowd/model.json --out results/raftcrowd/evaluation
python scripts/raftcrowd.py visualize --cache results/raftcrowd/cache --model results/raftcrowd/model.json --video-id test001 --frame-index 1 --out results/raftcrowd/panel.png
```

The example manifest contains example paths, not a downloaded dataset. GPU/Colab dependencies and pretrained weights are required for real extraction; the coordinator's native Windows environment has not validated GPU execution. Change device/resolution settings deliberately and retain the resulting configuration identity.

Extraction writes per-pair numeric NPZ shards under `cache/<video_id>/pair_000001.npz`, with the index naming the second/availability frame. This keeps extraction streaming rather than loading the entire benchmark into RAM. It does not make dense maps small: at 360 x 640, one float32 map is about 0.92 MB before compression, and storing many cues/controls over many frames can consume substantial disk space. Estimate storage with a small pilot before full extraction; compression ratios depend on the actual data. Keep source data, caches and result outputs local and out of Git.

The calibration JSON freezes cue scales and validation-selected thresholds. Evaluation and visualization must load those frozen values. Existing outputs must not be overwritten; use new paths for a new run. Preserve the cache receipt and input identity alongside the model/evaluation artifacts. An interrupted or partial cache is not complete scientific evidence.

## Comparator fairness

Run all configurations on the exact same admitted video/frame-pair cohort:

1. Flow-magnitude-only score.
2. Each active CFI cue alone.
3. Full active CFI.
4. Full CFI with each active cue removed in turn.
5. Full CFI with all moving pixels assigned to one region, retaining the other rules.

Pressure comparisons require the same external density coverage in every method. Changing the cohort to accommodate density is a separate declared analysis, not a replacement for the original comparison. Train the normalization for the no-segmentation control using its own training-derived cue distribution; keep its data and scaling policy matched. Give each method its own validation-selected operational threshold without allowing test selection.

The method is not an RGB-only trainable anomaly baseline, and this implementation does not fabricate that comparator. Published comparator values require verified primary sources and compatible protocols. In particular, the reviewer-requested HF2-VAD comparison needs a separate primary-source/protocol audit before reporting a numerical comparison.

## Calibration and threshold freeze

Training labels, when supplied, must establish normal-only calibration data. Validation must contain both classes for F1-based threshold selection. Select the threshold maximizing validation F1, using the highest threshold to break ties. Save threshold values, validation input identity, calibration identity and selection rule in a new output location. A test run may load this frozen artifact; it may not alter it using test scores or labels.

Maximum F1 obtained on test labels must never be described as deployment performance. No per-test-video min-max normalization, cue-weight optimization, cluster tuning or aggregation selection is permitted. If the model is revised after test inspection, label the following results exploratory and obtain genuinely unseen data for confirmatory claims.

## Metrics and uncertainty

Report micro-pooled frame ROC-AUC and EER using all retained pair scores and binary labels; also report fixed-threshold precision, recall, F1 and confusion counts. Preserve per-video metrics and sample counts. ROC-AUC and EER are undefined for a single-class video; show null with an explanation rather than a manufactured zero. A pooled metric is not the mean of per-video metrics; label the aggregation explicitly.

No repeated-run variability, confidence interval or significance claim is supported until the appropriate runs or resampling analysis have actually been performed. If confidence intervals are later added, account for temporal correlation by resampling videos or independent groups rather than treating adjacent frames as independent samples. Report failed cases and negative ablations without suppressing them.

## Runtime and reproducibility record

For each run archive the exact command, config JSON, code revision and dirty state or source digest, manifest and split identities, input/label/density hashes, RAFT weight identity, random seed, software versions, Python interpreter, OS, device and output location. Preserve numeric prediction arrays and generated table provenance. Never overwrite historical experiment output directories.

If benchmarking efficiency, record device name, power mode, input shape, batch, precision, RAFT update count, warm-up duration/count, timed pair count, CUDA synchronization, peak memory and mean/median/P95 latency. State whether decoding, resizing, RAFT, segmentation, density inference, CFI, visualization and file writing are included. Report FPS separately from encoded video playback speed. No Jetson speed number from the rejected paper is imported as a verified result.

## Completion gates

| Gate | Required evidence | Current evidence status |
| --- | --- | --- |
| Method specification | Source agrees with formulas, defaults and deliberate departures | New specification; coordinator must cross-check final code |
| Unit/integration correctness | Executed transformations, metric, split and pipeline tests | Actual test outcome belongs in progress/handoff, not assumed here |
| Real RAFT execution | Supported environment, real frames, weights and numeric output | NOT_DONE in this protocol |
| Reproducible benchmark results | Real manifests, labels, configs, predictions and regenerated tables | NOT_DONE |
| Safety/early-warning claim | Appropriate event annotations, lead-time and false-alarm study | NOT_DONE |

Synthetic fixtures verify implementation behavior. They are not anomaly-benchmark evidence, and any generated demonstration data must be marked `mock_` / `PLANNING DATA - NOT EXPERIMENTAL EVIDENCE` when placed in a result-facing artifact.
