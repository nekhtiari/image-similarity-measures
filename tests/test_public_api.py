import inspect

from image_similarity_measures import evaluate, quality_metrics

EXPECTED_METRIC_SIGNATURES = {
    "fsim": "(org_img: numpy.ndarray, pred_img: numpy.ndarray, T1: float = 0.85, T2: float = 160) -> float",
    "issm": "(org_img: numpy.ndarray, pred_img: numpy.ndarray) -> float",
    "psnr": "(org_img: numpy.ndarray, pred_img: numpy.ndarray, max_p: int = 4095) -> float",
    "rmse": "(org_img: numpy.ndarray, pred_img: numpy.ndarray, max_p: int = 4095) -> float",
    "sam": "(org_img: numpy.ndarray, pred_img: numpy.ndarray, convert_to_degree: bool = True) -> float",
    "sre": "(org_img: numpy.ndarray, pred_img: numpy.ndarray)",
    "ssim": "(org_img: numpy.ndarray, pred_img: numpy.ndarray, max_p: int = 4095) -> float",
    "uiq": "(org_img: numpy.ndarray, pred_img: numpy.ndarray, step_size: int = 1, window_size: int = 8) -> float",
}


def test_metric_registry_is_stable():
    assert list(quality_metrics.metric_functions) == [
        "fsim",
        "issm",
        "psnr",
        "rmse",
        "sam",
        "sre",
        "ssim",
        "uiq",
    ]


def test_metric_signatures_are_stable():
    actual = {
        name: str(inspect.signature(function))
        for name, function in quality_metrics.metric_functions.items()
    }
    assert actual == EXPECTED_METRIC_SIGNATURES


def test_evaluation_signatures_are_stable():
    assert str(inspect.signature(evaluate.evaluation)) == (
        "(org_img_path: str, pred_img_path: str, metrics: List[str])"
    )
    assert str(inspect.signature(evaluate.read_image)) == "(path: str)"


def test_explicit_loader_api_is_additive():
    assert str(inspect.signature(evaluate.read_image_with_loader)) == (
        "(path: str, loader: Literal['auto', 'opencv', 'rasterio'] = 'auto')"
    )
    assert str(inspect.signature(evaluate.evaluation_with_loader)) == (
        "(org_img_path: str, pred_img_path: str, metrics: List[str], "
        "loader: Literal['auto', 'opencv', 'rasterio'] = 'auto')"
    )
