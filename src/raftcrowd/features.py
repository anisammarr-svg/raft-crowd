"""Numeric motion cues for a documented RAFT-CrowdFlow starting implementation.

These choices are prospective implementation choices, not a reconstruction of
the rejected paper's experiments. Flow is horizontal/vertical displacement in
pixels per observed frame pair. Derivatives use a unit image-pixel grid. The
optional density input must be a separately obtained, aligned pedestrian-density
map; flow magnitude is never substituted for pedestrian density.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class CueConfig:
    """Motion segmentation settings, fixed before held-out evaluation.

    K-means clusters moving pixels by (u, v), without temporal label tracking.
    A seeded bounded subsample fits the centers; every moving pixel is assigned.
    Pixels with magnitude <= min_motion are background, labeled -1. This gate
    deliberately excludes near-static regions and does not detect static crushes.
    """

    clusters: int = 4
    min_motion: float = 0.5
    kmeans_iterations: int = 20
    kmeans_samples: int = 4096
    seed: int = 7

    def __post_init__(self) -> None:
        for name in ("clusters", "kmeans_iterations", "kmeans_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.min_motion) or self.min_motion < 0:
            raise ValueError("min_motion must be finite and nonnegative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _nearest(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Bound temporary memory when assigning full-resolution frames."""
    labels = np.empty(len(points), dtype=np.int32)
    for start in range(0, len(points), 65536):
        stop = min(start + 65536, len(points))
        delta = points[start:stop, None, :] - centers[None, :, :]
        labels[start:stop] = np.argmin(np.sum(delta * delta, axis=2), axis=1)
    return labels


def _cluster(points: np.ndarray, config: CueConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    if len(points) > config.kmeans_samples:
        indices = np.sort(rng.choice(len(points), config.kmeans_samples, replace=False))
        sample = points[indices]
    else:
        sample = points
    unique = np.unique(sample, axis=0)
    count = min(config.clusters, len(unique))
    # A shared scale preserves Euclidean geometry in (u, v). Do not separately
    # whiten the axes, which would change the relative directional distances.
    scale = max(float(np.median(np.hypot(sample[:, 0], sample[:, 1]))), 1e-12)
    sample = sample / scale
    unique = unique / scale
    centers = [unique[int(rng.integers(len(unique)))]]
    nearest_distance = np.sum((unique - centers[0]) ** 2, axis=1)
    for _ in range(1, count):
        centers.append(unique[int(np.argmax(nearest_distance))])
        nearest_distance = np.minimum(
            nearest_distance, np.sum((unique - centers[-1]) ** 2, axis=1)
        )
    centers = np.asarray(centers)
    for _ in range(config.kmeans_iterations):
        assignments = _nearest(sample, centers)
        updated = centers.copy()
        for label in range(count):
            selected = assignments == label
            if np.any(selected):
                updated[label] = np.mean(sample[selected], axis=0)
        if np.allclose(updated, centers, rtol=0, atol=1e-8):
            centers = updated
            break
        centers = updated
    return _nearest(points / scale, centers)


def compute_cues(
    flow: np.ndarray,
    config: CueConfig,
    density: np.ndarray | None = None,
    segment: bool = True,
) -> dict[str, np.ndarray]:
    """Return numeric cue maps plus ``mask`` and integer ``segments``.

    Input shape is (height, width, 2), with each spatial dimension >= 2.
    Returned numeric maps are float64, finite, and zero outside the motion mask:
    ``magnitude``, ``compression``, ``direction_variance``,
    ``magnitude_variance``, and optionally ``pressure`` = density * magnitude.
    Direction variance is circular variance 1 - |mean(exp(i theta))|, rather
    than the paper's ambiguous ordinary variance of wrapped angles. Regional
    magnitude variance is population variance (ddof=0). ``segment=False``
    assigns all moving pixels to one region, as a no-segmentation control.

    No input is modified. No image/color normalization is performed here.
    A separate training-only fitted normalizer must precede score fusion.
    """
    if not isinstance(config, CueConfig):
        raise TypeError("config must be a CueConfig")
    field = np.asarray(flow, dtype=np.float64)
    if field.ndim != 3 or field.shape[-1] != 2 or min(field.shape[:2]) < 2:
        raise ValueError("flow must have shape (H, W, 2), with H and W >= 2")
    if not np.all(np.isfinite(field)):
        raise ValueError("flow must contain only finite values")
    u, v = field[..., 0], field[..., 1]
    magnitude = np.hypot(u, v)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("flow magnitude exceeds numeric range")
    mask = magnitude > config.min_motion
    labels = np.full(mask.shape, -1, dtype=np.int32)
    if np.any(mask):
        labels[mask] = _cluster(field[mask], config) if segment else 0
    compression = np.maximum(
        0.0, -(np.gradient(u, axis=1) + np.gradient(v, axis=0))
    )
    direction_variance = np.zeros(mask.shape, dtype=np.float64)
    magnitude_variance = np.zeros(mask.shape, dtype=np.float64)
    for label in np.unique(labels[mask]):
        selected = labels == label
        resultant = np.hypot(
            np.mean(u[selected] / magnitude[selected]),
            np.mean(v[selected] / magnitude[selected]),
        )
        direction_variance[selected] = np.clip(1.0 - resultant, 0.0, 1.0)
        magnitude_variance[selected] = np.var(magnitude[selected], ddof=0)
    result = {
        "magnitude": np.where(mask, magnitude, 0.0),
        "compression": np.where(mask, compression, 0.0),
        "direction_variance": direction_variance,
        "magnitude_variance": magnitude_variance,
        "mask": mask,
        "segments": labels,
    }
    if density is not None:
        density_map = np.asarray(density, dtype=np.float64)
        if density_map.shape != mask.shape:
            raise ValueError("density must have exactly the flow spatial shape")
        if not np.all(np.isfinite(density_map)) or np.any(density_map < 0):
            raise ValueError("density must be finite and nonnegative")
        result["pressure"] = np.where(mask, density_map * magnitude, 0.0)
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("cue computation exceeded numeric range")
    return result


def aggregate_map(
    values: np.ndarray,
    top_fraction: float = 0.01,
    mask: np.ndarray | None = None,
) -> float:
    """Mean of the largest ceil(top_fraction * N) selected pixel scores.

    Without a mask N is the full image area, including zero-valued background;
    this fixes the scored area across frames. Passing a mask changes N and must
    be declared as a separate protocol. An empty selected mask returns 0.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a nonempty finite two-dimensional map")
    if np.any(array < 0):
        raise ValueError("score map must be nonnegative")
    if not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if mask is None:
        selected = array.ravel()
    else:
        selected_mask = np.asarray(mask)
        if selected_mask.shape != array.shape or selected_mask.dtype != np.bool_:
            raise ValueError("mask must be Boolean and match the score map")
        selected = array[selected_mask]
    if selected.size == 0:
        return 0.0
    count = max(1, int(np.ceil(top_fraction * selected.size)))
    return float(np.mean(np.partition(selected, selected.size - count)[-count:]))
