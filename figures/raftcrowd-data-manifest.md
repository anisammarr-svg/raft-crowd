# RAFT-CrowdFlow figure provenance requirements

Date: 2026-11-18. No new scientific figure is claimed in this manifest. These requirements apply to figures produced by the new implementation, not retrospectively verified figures from the rejected PDF.

For each figure retain source video/frame names and hashes, manifest/config/source identities, original and resized dimensions, pair indices, RAFT weights, numeric flow and cue arrays, calibration/threshold artifacts, plotting settings, output path and caption. Use one flow tensor for all panels belonging to a pair. Record the time at which the displayed score becomes available.

| Panel/output | Numerical source | Required labeling |
| --- | --- | --- |
| RGB source | Source frame I_t | Video, frame index, resolution and any resize/crop |
| Flow color | Same numeric F_t used by every derived panel | Color direction convention; magnitude display scaling |
| CFI map | Normalized active cues with declared weights | Actual CFI; cue set, fixed scale and available-at-frame t+1 convention |
| Magnitude | `sqrt(u^2+v^2)` | Resized pixels per pair, not metres/s or density |
| Divergence/compression | Numeric signed divergence or nonnegative compression | Name the chosen quantity; fixed zero/scale and finite-difference convention |
| Density | External aligned pedestrian-density map only | Estimator and units; no density panel when density is absent |
| Segmentation | Deterministic motion-region labels | Motion clusters, not person segmentation or annotated risk zones |

Do not substitute flow arrows for CFI or blurred flow magnitude for pedestrian density. A colorized map is a visualization, not the numeric scoring input. Use consistent fixed scales when making between-frame intensity comparisons; if a visualization uses independent display normalization, disclose it and do not infer changes in calibrated score from color alone.

Quantitative plots and tables must read reviewed numeric artifacts, not manually entered manuscript values. Synthetic examples must be clearly labeled `mock_` / `PLANNING DATA - NOT EXPERIMENTAL EVIDENCE`. Qualitative frames may illustrate behavior but do not validate early warning, physical pressure or spatial localization without task-matched annotations and quantitative evaluation.
