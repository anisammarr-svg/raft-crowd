import numpy as np
import pytest

from raftcrowd.features import CueConfig, aggregate_map, compute_cues


def test_constant_translation_has_no_instability():
    flow = np.zeros((8, 9, 2))
    flow[..., 0] = 3.0
    original = flow.copy()
    cues = compute_cues(flow, CueConfig())
    np.testing.assert_array_equal(flow, original)
    np.testing.assert_allclose(cues["magnitude"], 3.0)
    for name in ("compression", "direction_variance", "magnitude_variance"):
        np.testing.assert_allclose(cues[name], 0.0, atol=1e-12)
    assert "pressure" not in cues
    assert np.all(cues["mask"])


def test_linear_convergence_has_correct_compression_and_expansion_does_not():
    y, x = np.mgrid[-3:4, -4:5]
    flow = np.stack((-0.5 * x, -0.25 * y), axis=-1)
    cues = compute_cues(flow, CueConfig(min_motion=0.0))
    np.testing.assert_allclose(cues["compression"][cues["mask"]], 0.75)
    assert cues["compression"][3, 4] == 0.0  # Explicit moving-only policy.
    expansion = compute_cues(-flow, CueConfig(min_motion=0.0))
    np.testing.assert_allclose(expansion["compression"], 0.0)


def test_circular_variance_respects_angle_wraparound():
    angles = np.array([[np.pi - 0.01, -np.pi + 0.01]] * 2)
    flow = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    cues = compute_cues(flow, CueConfig(clusters=1))
    np.testing.assert_allclose(cues["direction_variance"], 1 - np.cos(0.01), atol=1e-12)
    np.testing.assert_allclose(cues["magnitude_variance"], 0.0, atol=1e-12)


def test_population_magnitude_variance_and_no_segmentation_control():
    flow = np.zeros((2, 2, 2))
    flow[..., 0] = [[1.0, 3.0], [1.0, 3.0]]
    config = CueConfig(clusters=2)
    segmented = compute_cues(flow, config)
    unsegmented = compute_cues(flow, config, segment=False)
    np.testing.assert_allclose(segmented["magnitude_variance"], 0.0)
    np.testing.assert_allclose(unsegmented["magnitude_variance"], 1.0)
    np.testing.assert_array_equal(unsegmented["segments"], 0)


def test_external_density_is_required_for_pressure_and_not_modified():
    flow = np.full((3, 4, 2), 0.0)
    flow[..., 0] = 2.0
    density = np.arange(12, dtype=float).reshape(3, 4)
    original = density.copy()
    cues = compute_cues(flow, CueConfig(), density=density)
    np.testing.assert_allclose(cues["pressure"], density * 2.0)
    np.testing.assert_array_equal(density, original)


def test_segmentation_is_repeatable_with_bounded_training_sample():
    flow = np.random.default_rng(4).normal(size=(20, 21, 2))
    config = CueConfig(kmeans_samples=16, clusters=4, seed=7)
    first = compute_cues(flow, config)
    second = compute_cues(flow, config)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert set(np.unique(first["segments"])) <= {-1, 0, 1, 2, 3}


def test_no_motion_has_zero_maps_and_background_labels():
    result = compute_cues(np.zeros((2, 2, 2)), CueConfig())
    assert not result["mask"].any()
    np.testing.assert_array_equal(result["segments"], -1)
    for name in ("magnitude", "compression", "direction_variance", "magnitude_variance"):
        np.testing.assert_array_equal(result[name], 0)


@pytest.mark.parametrize("flow", [np.zeros((2, 2)), np.zeros((1, 2, 2)), np.full((2, 2, 2), np.nan)])
def test_invalid_flow_is_rejected(flow):
    with pytest.raises(ValueError):
        compute_cues(flow, CueConfig())


@pytest.mark.parametrize("density", [np.ones((3, 3)), -np.ones((2, 2)), np.full((2, 2), np.inf)])
def test_invalid_density_is_rejected(density):
    with pytest.raises(ValueError):
        compute_cues(np.ones((2, 2, 2)), CueConfig(), density=density)


@pytest.mark.parametrize("kwargs", [{"clusters": 0}, {"clusters": True}, {"kmeans_samples": 0}, {"seed": -1}, {"min_motion": -1}, {"min_motion": np.nan}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CueConfig(**kwargs)


def test_top_fraction_aggregation_uses_full_area_unless_mask_supplied():
    values = np.array([[0.0, 0.0], [0.0, 8.0]])
    original = values.copy()
    assert aggregate_map(values, top_fraction=0.5) == 4.0
    assert aggregate_map(values, top_fraction=1.0) == 2.0
    assert aggregate_map(values, top_fraction=0.5, mask=values > 0) == 8.0
    assert aggregate_map(values, mask=np.zeros((2, 2), dtype=bool)) == 0.0
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, np.nan])
def test_invalid_aggregation_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        aggregate_map(np.ones((2, 2)), fraction)


def test_bad_aggregation_map_and_mask_are_rejected():
    with pytest.raises(ValueError):
        aggregate_map(np.array([[np.inf]]))
    with pytest.raises(ValueError):
        aggregate_map(-np.ones((2, 2)))
    with pytest.raises(ValueError):
        aggregate_map(np.ones((2, 2)), mask=np.ones((2, 2)))
