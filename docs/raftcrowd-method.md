# RAFT-CrowdFlow starting implementation

Date: 2026-11-18. This is a new, explicitly specified implementation inspired by the rejected manuscript. It does not recover the missing original evaluation code, reproduce its tables, establish novelty, or validate crowd-crush prediction. The SDD LiteCrowdRelNet project is separate and unchanged.

## Pipeline

```text
ordered RGB frames I_t, I_(t+1)
    -> one frozen RAFT flow field F_t
    -> magnitude and moving-pixel mask
    -> deterministic motion segmentation
    -> compression + regional directional/magnitude variability
       + optional external pedestrian-density x magnitude
    -> training-fitted cue scaling
    -> fixed-weight CFI map
    -> top-fraction spatial aggregation
    -> anomaly score available at frame t+1
```

There is no learned anomaly network or optimizer in this version. RAFT uses pretrained weights without updating them. Training data fit cue normalization; validation data select a fixed operating threshold for each declared method. Test data only evaluate those frozen choices.

## Explicit implementation choices

| Component | Starting specification | Limit or departure from the manuscript |
| --- | --- | --- |
| Flow estimator | Torchvision `raft_large`, explicit `Raft_Large_Weights.C_T_V1` | FlyingChairs/FlyingThings training lineage; not a byte-identical copy of the uploaded notebook's `raft-things.pth` file |
| Inputs | Consecutive RGB frames, resized to 360 x 640 by default, batch size 1 | Displacements are pixels in this resized image per pair, not original-image pixels, metres or velocity per second |
| Geometry | Replicate-pad images to a multiple of eight, run RAFT, crop the output flow back to the resized image | Crop, rather than resize, removes padding; input height and width must satisfy RAFT's supported minimum |
| RAFT updates | 12 by default; floating point 32 by default | These settings do not reproduce the paper's unverified timing; optional CUDA mixed precision is an explicit configuration choice |
| Flow smoothing | Optional Gaussian smoothing of numeric u/v fields, default sigma 1 | Smoothing acts on flow, not on a color visualization |
| Moving mask | Magnitude greater than 0.5 pixels per pair by default | This threshold is a proposed setting, not a recovered experimental value |
| Segmentation | At most four deterministic k-means groups using moving-pixel u/v vectors; bounded fitting sample and fixed seed | No semantic person segmentation, no temporal mask propagation/warping in version 1 |
| Compression | `max(0, -(du/dx + dv/dy))` | Finite differences use the resized image grid; this is a visual motion cue, not a calibrated physical pressure measurement |
| Directional variability | Per-region circular variance `1 - hypot(mean(cos(theta)), mean(sin(theta)))` | Explicitly replaces naive linear angle variance so directions near -pi and pi remain close |
| Magnitude variability | Per-region population variance of flow magnitude | Broadcast to that region's moving pixels; zero outside the moving mask |
| Density coupling | Optional `D_t * magnitude(F_t)` from externally supplied, nonnegative pedestrian-density maps | No density estimator is invented or inferred from flow. Default pressure cue is disabled. Blurred flow magnitude is not pedestrian density |
| Scaling | Training-only sampled 99th-percentile upper scale per nonnegative cue; zero lower bound; clamp to [0,1] | No per-test-video min-max normalization and no test-fitted scaling |
| Fusion | Equal weights for active CFI cues, renormalized for each declared ablation | These weights are proposed and fixed; not the manuscript's unknown weights |
| Frame score | Mean of the largest 1% of map pixels, over the whole image | Uses all spatial pixels for a consistent denominator; no best-test-score aggregation search |

Map names used by the implementation include `magnitude`, `compression`, `direction_variance`, `magnitude_variance` and optional `pressure`. Segmentation labels and the moving mask support interpretation but are not extra CFI terms. Empty moving regions produce zero variability rather than missing scores.

The initial configuration records seed 7, at most 20 k-means updates on at most 4,096 fitting vectors, and normalization sampling of 128 values per frame capped at 50,000 per cue. These are implementation defaults, not values recovered from the rejected paper. The cache also stores the single-region control so ablation scoring does not rerun RAFT under different inputs.

Normalization samples only moving pixels from normal training pairs. A bounded random-priority reservoir is deterministic for a fixed manifest traversal order. An empty or effectively zero-range training cue receives an explicitly recorded scale of 1 to avoid division by zero; inspect `zero_scale_fallback_cues` before interpreting its contribution. Single-region variability maps have their own training-derived scales. Scaling is fixed before validation chooses thresholds.

## Coordinates, timing and density

The flow from frames t and t+1 lives on source-frame t coordinates. Any external density map must refer to those same source-frame coordinates and the configured resized resolution. Name its `.npy` file using the source frame's stem. A density array must be a finite, nonnegative H x W NumPy array; the pipeline must reject a missing or mismatched density map when pressure is enabled. Density provenance must name its estimator, checkpoint, training data, calibration, coordinate transform and units. It must not be a flow-magnitude image relabeled as a count map.

The resulting pair score becomes available only when frame t+1 has arrived. Evaluation therefore aligns it with `labels[t+1]`; frame zero has no pair score and is excluded. This convention measures anomaly discrimination, not advance prediction. No warning lead time is established without separate annotated event-onset analysis.

The resized coordinate system and camera perspective limit comparisons of motion magnitude across scenes. Neither a CFI heatmap nor negative image divergence independently proves crowd compression, crowd density, a dangerous interaction or causal risk. Heatmaps require separate spatial annotations before localization accuracy can be claimed.

## Controls and claim-to-experiment traceability

All methods must use the same frame pairs, RAFT flow and input cohort. The magnitude baseline must not receive different normalization, labels or threshold selection privileges.

| Question | Required comparison/evidence | Permitted conclusion after successful execution |
| --- | --- | --- |
| Does CFI add information beyond speed? | Full active CFI versus flow-magnitude-only baseline | Dataset-specific anomaly-score comparison, not safety effectiveness |
| Which cue contributes? | Each cue alone and each leave-one-cue-out variant | Conditional contribution under the declared fixed scoring protocol |
| Does segmentation contribute? | Full CFI versus a single-moving-region control | Effect of this specific segmentation construction |
| Does pedestrian density contribute? | Same cohort with external density and pressure removed; inspect density provenance | Effect of that supplied estimator's density coupling |
| Are operating-point results valid? | Validation-chosen threshold frozen before test, saved per-frame predictions | Held-out precision, recall and F1 for that fixed threshold |
| Are reported AUC values reproducible? | Archived configuration, manifest, input identities, code and predictions | Regenerated values from the new implementation only |
| Is early crush warning supported? | Separate crowd-crush/onset annotations and prospective lead-time/false-alarm evaluation | No such conclusion from the current anomaly benchmarks alone |
| Is embedded real time supported? | Actual target-device benchmark with synchronized timing, warm-up and latency distribution | No such conclusion from video playback FPS or Colab metadata |

## Primary implementation references

- [Torchvision RAFT Large API and weight definitions](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html).
- [Official Torchvision optical-flow example](https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html), including RGB preprocessing, pretrained transforms and output convention.

These sources document the RAFT API. They do not establish novelty, experimental results, crowd-safety validity or equivalence to the rejected paper.
