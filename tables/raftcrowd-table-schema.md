# RAFT-CrowdFlow result table schema

Date: 2026-11-18. This file defines future outputs; it contains no measured results. Never copy the rejected manuscript's 98.9/91.4/73.4 values into new result tables as defaults.

## Comparison and ablation rows

One row per dataset, manifest identity, method/configuration and evaluation aggregation:

| Column | Meaning |
| --- | --- |
| `dataset` / `dataset_version` | Explicit release or author-defined dataset identity |
| `manifest_hash` | Frozen video/split/label manifest identity |
| `code_revision` / `source_digest` | Actual executable source identity; record dirty state |
| `config_hash` | Flow, cues, normalization, fusion and aggregation identity |
| `method` | Magnitude, cue-alone, full CFI, cue removed or no segmentation |
| `density_source` | Disabled, or estimator/checkpoint/coordinate provenance |
| `seed` | Actual deterministic seed; no invented repeated-run variability |
| `split` / `aggregation` | Test/validation and micro-pooled/per-video explicitly |
| `video_count` / `frame_pair_count` | Retained evaluation cohort, excluding every first frame |
| `roc_auc` | ROC-AUC as fraction [0,1]; multiply by 100 only in a clearly labeled display |
| `eer` | EER fraction [0,1]; display percent only when labeled |
| `threshold` | Actual frozen numeric operational threshold |
| `threshold_source` | Validation F1, selection rule and artifact identity |
| `precision`, `recall`, `f1` | Fixed-threshold metrics, fractions [0,1] |
| `tp`, `fp`, `tn`, `fn` | Integer confusion counts |
| `prediction_artifact` | Exact per-frame score file and hash |
| `evidence_status` | Executed/verified or explicitly pending; never imply a planned run occurred |

For per-video single-class rows, ROC-AUC/EER must be null and the reason stated. Keep those videos in pooled analysis if their inputs are valid. Missing density or malformed labels must not silently change one method's cohort.

The full-CFI row in the comparison table and full-CFI row in the ablation table must reference the same predictions when claiming the same dataset/protocol. Generate both displays from one reviewed result source; reconcile any differences before manuscript editing.

## Operating-threshold table

If reporting several alarm thresholds, name the dataset, split, unit of evaluation, score definition and numeric threshold in each row. Distinguish a validation threshold sweep from fixed-threshold held-out performance. Repeated values across displays are acceptable only when they intentionally describe the same predictions and operating point.

Frame-label precision/recall/F1 must not be labeled risk-localization performance. Spatial localization requires a separate table with spatial ground truth, region/pixel matching rules and a validated evaluation implementation.

## Efficiency table

Include hardware, OS/software, power mode, input dimensions, batch, precision, RAFT updates, timed/warm-up counts, synchronized timing boundary, included components, mean/median/P95 milliseconds, throughput pairs/s, peak memory and timing-log artifact. Do not infer processing speed from a video writer's FPS setting. Results on Colab cannot be renamed Jetson measurements.

## Consistency checks before publication

Verify precision/recall/F1 against confusion counts, units across every display, cohort counts across methods, threshold source independence, and table-to-prediction hashes. Report rounding only in presentation; retain full numeric precision in artifacts. No AUC difference alone establishes statistical significance.
