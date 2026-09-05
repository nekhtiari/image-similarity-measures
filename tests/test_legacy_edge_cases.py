import numpy as np
import pytest

from image_similarity_measures import quality_metrics


@pytest.fixture
def sample_images():
    original = (np.arange(16 * 16 * 3).reshape(16, 16, 3) % 256).astype(np.uint8)
    predicted = original.copy()
    predicted[2:9, 4:12, :] = np.clip(
        predicted[2:9, 4:12, :].astype(np.uint16) + 17, 0, 255
    ).astype(np.uint8)
    return original, predicted


def test_shape_mismatch_raises_legacy_assertion_error(sample_images):
    original, predicted = sample_images
    with pytest.raises(AssertionError, match="Input shapes not identical"):
        quality_metrics.rmse(original, predicted[:, :, :2])


def test_2d_grayscale_has_known_legacy_failures(sample_images):
    original, predicted = sample_images
    gray_original = original[:, :, 0]
    gray_predicted = predicted[:, :, 0]

    expected_errors = {
        "fsim": IndexError,
        "issm": IndexError,
        "sre": IndexError,
        "ssim": IndexError,
        "uiq": ValueError,
    }
    with np.errstate(all="ignore"):
        for name, error in expected_errors.items():
            with pytest.raises(error):
                quality_metrics.metric_functions[name](gray_original, gray_predicted)


def test_2d_grayscale_rmse_preserves_legacy_broadcasting(sample_images):
    original, predicted = sample_images
    gray_original = original[:, :, 0]
    gray_predicted = predicted[:, :, 0]

    two_dimensional = quality_metrics.rmse(gray_original, gray_predicted)
    channel_last = quality_metrics.rmse(
        gray_original[:, :, np.newaxis], gray_predicted[:, :, np.newaxis]
    )

    assert two_dimensional == pytest.approx(0.0249851793)
    assert channel_last == pytest.approx(0.0019072039)


def test_integer_sam_preserves_known_overflow_behavior(sample_images):
    original, _ = sample_images

    integer_result = quality_metrics.sam(original, original)
    floating_result = quality_metrics.sam(
        original.astype(np.float64), original.astype(np.float64)
    )

    assert integer_result == pytest.approx(83.44096258)
    assert floating_result == pytest.approx(0.0, abs=1e-6)


def test_identical_zero_images_preserve_legacy_non_finite_results():
    zeros = np.zeros((16, 16, 3), dtype=np.uint8)

    with np.errstate(all="ignore"):
        assert np.isnan(quality_metrics.fsim(zeros, zeros))
        assert np.isnan(quality_metrics.sre(zeros, zeros))
        assert quality_metrics.uiq(zeros, zeros) == 0.0


def test_issm_preserves_legacy_identity_result(sample_images):
    original, _ = sample_images

    with np.errstate(all="ignore"):
        assert quality_metrics.issm(original, original) == 0.0
