import json
import sys
from types import SimpleNamespace

import numpy as np

from image_similarity_measures import evaluate


def test_evaluation_reads_both_images_and_runs_requested_metrics(monkeypatch):
    original = np.array([[[1]]], dtype=np.uint8)
    predicted = np.array([[[2]]], dtype=np.uint8)
    paths = {"original.tif": original, "predicted.tif": predicted}
    calls = []

    monkeypatch.setattr(evaluate, "read_image", paths.__getitem__)

    def metric(org_img, pred_img):
        calls.append((org_img, pred_img))
        return np.float64(2.5)

    monkeypatch.setitem(evaluate.metric_functions, "probe", metric)

    assert evaluate.evaluation("original.tif", "predicted.tif", ["probe"]) == {
        "probe": 2.5
    }
    assert calls == [(original, predicted)]


def test_read_image_uses_rasterio_for_lowercase_tiff(monkeypatch):
    band_first = np.arange(12).reshape(2, 2, 3)
    opened = []

    class Dataset:
        def read(self):
            return band_first

    fake_rasterio = SimpleNamespace(open=lambda path: opened.append(path) or Dataset())
    monkeypatch.setattr(evaluate, "rasterio", fake_rasterio)
    monkeypatch.setattr(
        evaluate.cv2,
        "imread",
        lambda path: (_ for _ in ()).throw(AssertionError(path)),
    )

    actual = evaluate.read_image("image.tiff")

    assert opened == ["image.tiff"]
    np.testing.assert_array_equal(actual, np.rollaxis(band_first, 0, 3))


def test_read_image_uses_opencv_without_rasterio(monkeypatch):
    expected = np.ones((2, 3, 3), dtype=np.uint8)
    opened = []
    monkeypatch.setattr(evaluate, "rasterio", None)
    monkeypatch.setattr(
        evaluate.cv2,
        "imread",
        lambda path: opened.append(path) or expected,
    )

    assert evaluate.read_image("image.tif") is expected
    assert opened == ["image.tif"]


def test_read_image_preserves_case_sensitive_legacy_selection(monkeypatch):
    expected = np.ones((2, 3, 3), dtype=np.uint8)
    opened = []
    monkeypatch.setattr(
        evaluate,
        "rasterio",
        SimpleNamespace(open=lambda path: (_ for _ in ()).throw(AssertionError(path))),
    )
    monkeypatch.setattr(
        evaluate.cv2,
        "imread",
        lambda path: opened.append(path) or expected,
    )

    assert evaluate.read_image("IMAGE.TIF") is expected
    assert opened == ["IMAGE.TIF"]


def test_cli_defaults_to_psnr(monkeypatch, capsys):
    observed = {}

    def fake_evaluation(org_img_path, pred_img_path, metrics):
        observed.update(
            org_img_path=org_img_path,
            pred_img_path=pred_img_path,
            metrics=metrics,
        )
        return {"psnr": 12.5}

    monkeypatch.setattr(evaluate, "evaluation", fake_evaluation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image-similarity-measures",
            "--org_img_path=original.tif",
            "--pred_img_path=predicted.tif",
        ],
    )

    evaluate.main()

    assert observed == {
        "org_img_path": "original.tif",
        "pred_img_path": "predicted.tif",
        "metrics": ["psnr"],
    }
    assert json.loads(capsys.readouterr().out) == {
        "image1": "original.tif",
        "image2": "predicted.tif",
        "metrics": {"psnr": 12.5},
    }


def test_cli_all_expands_to_sorted_metric_names(monkeypatch, capsys):
    observed = {}

    def fake_evaluation(org_img_path, pred_img_path, metrics):
        observed["metrics"] = metrics
        return {metric: float(index) for index, metric in enumerate(metrics)}

    monkeypatch.setattr(evaluate, "evaluation", fake_evaluation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image-similarity-measures",
            "--org_img_path=original.tif",
            "--pred_img_path=predicted.tif",
            "--metric=all",
        ],
    )

    evaluate.main()

    expected = sorted(evaluate.metric_functions)
    assert observed["metrics"] == expected
    assert list(json.loads(capsys.readouterr().out)["metrics"]) == expected
