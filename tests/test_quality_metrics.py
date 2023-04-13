import numpy as np
from image_similarity_measures import quality_metrics


def test_rmse(test_array1, test_array2):
    rmse = quality_metrics.rmse(test_array1, test_array2)
    assert float(rmse) == 0.0636659786105156


def test_fsim(test_array1, test_array2):
    fsim = quality_metrics.fsim(test_array1, test_array2)
    assert float(fsim) == 0.6065146307254211


def test_issm(test_array1, test_array2):
    issm = quality_metrics.issm(test_array1, test_array2)
    assert float(issm) == 0.133647886351807


def test_psnr(test_array1, test_array2):
    psnr = quality_metrics.psnr(test_array1, test_array2)
    assert float(psnr) == 23.879422595287842


def test_sam(test_array1, test_array2):
    sam = quality_metrics.sam(test_array1, test_array2)
    assert float(sam) == 89.24550823067034


def test_sre(test_array1, test_array2):
    sre = quality_metrics.sre(test_array1, test_array2)
    assert float(sre) == 65.2174177232156


def test_ssim(test_array1, test_array2):
    ssim = quality_metrics.ssim(test_array1, test_array2)
    assert float(ssim) == 0.6543283753736631


def test_uiq(test_array1, test_array2):
    uiq = quality_metrics.uiq(test_array1, test_array2)
    assert float(uiq) == 0.4606513484076305


def test_similarity_measure(test_array1, test_array2):
    sm = quality_metrics._similarity_measure(test_array1, test_array2, constant=1)
    assert sm.shape == (200, 200, 4)
    assert sm.mean() == 8.02524196795888


def test_gradient_magnitude(test_array1):
    gm = quality_metrics._gradient_magnitude(test_array1, img_depth=3)
    assert gm.shape == (200, 200, 4)
    np.testing.assert_array_equal(
        gm[0:3, 0, 0], np.array([0, 8.246211, 48.166378], dtype=np.float32)
    )
