import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_explicit_opencv_loader_overrides_tiff_auto_selection(monkeypatch):
    expected = np.ones((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(
        evaluate,
        "rasterio",
        SimpleNamespace(open=lambda path: (_ for _ in ()).throw(AssertionError(path))),
    )
    monkeypatch.setattr(evaluate.cv2, "imread", lambda path: expected)

    assert evaluate.read_image_with_loader("image.tif", "opencv") is expected


def test_explicit_loader_auto_delegates_to_legacy_reader(monkeypatch):
    expected = np.ones((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(evaluate, "read_image", lambda path: expected)

    assert evaluate.read_image_with_loader("image.tif") is expected


def test_explicit_rasterio_loader_handles_uppercase_suffix_and_closes(monkeypatch):
    band_first = np.arange(12).reshape(2, 2, 3)
    state = {"closed": False}

    class Dataset:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            state["closed"] = True

        def read(self):
            return band_first

    monkeypatch.setattr(
        evaluate,
        "rasterio",
        SimpleNamespace(open=lambda path: Dataset()),
    )

    actual = evaluate.read_image_with_loader("IMAGE.TIF", "rasterio")

    assert state["closed"] is True
    np.testing.assert_array_equal(actual, np.rollaxis(band_first, 0, 3))


def test_explicit_rasterio_loader_requires_optional_dependency(monkeypatch):
    monkeypatch.setattr(evaluate, "rasterio", None)

    with pytest.raises(ImportError, match="image-similarity-measures\\[rasterio\\]"):
        evaluate.read_image_with_loader("image.tif", "rasterio")


def test_explicit_loader_rejects_unknown_value():
    with pytest.raises(ValueError, match="auto, opencv, rasterio"):
        evaluate.read_image_with_loader("image.tif", "unknown")


def test_evaluation_with_explicit_loader(monkeypatch):
    original = np.array([[[1]]], dtype=np.uint8)
    predicted = np.array([[[2]]], dtype=np.uint8)
    paths = {"original.tif": original, "predicted.tif": predicted}
    calls = []

    def fake_read(path, loader):
        calls.append((path, loader))
        return paths[path]

    monkeypatch.setattr(evaluate, "read_image_with_loader", fake_read)
    monkeypatch.setitem(evaluate.metric_functions, "probe", lambda x, y: 4.5)

    assert evaluate.evaluation_with_loader(
        "original.tif", "predicted.tif", ["probe"], loader="opencv"
    ) == {"probe": 4.5}
    assert calls == [
        ("original.tif", "opencv"),
        ("predicted.tif", "opencv"),
    ]


def test_evaluation_with_auto_loader_delegates_to_legacy_api(monkeypatch):
    observed = {}

    def fake_evaluation(org_img_path, pred_img_path, metrics):
        observed.update(
            org_img_path=org_img_path,
            pred_img_path=pred_img_path,
            metrics=metrics,
        )
        return {"rmse": 1.5}

    monkeypatch.setattr(evaluate, "evaluation", fake_evaluation)

    assert evaluate.evaluation_with_loader(
        "original.tif", "predicted.tif", ["rmse"]
    ) == {"rmse": 1.5}
    assert observed == {
        "org_img_path": "original.tif",
        "pred_img_path": "predicted.tif",
        "metrics": ["rmse"],
    }


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


def test_cli_accepts_explicit_loader(monkeypatch, capsys):
    observed = {}

    def fake_evaluation(org_img_path, pred_img_path, metrics, loader):
        observed.update(metrics=metrics, loader=loader)
        return {"rmse": 1.25}

    monkeypatch.setattr(evaluate, "evaluation_with_loader", fake_evaluation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "image-similarity-measures",
            "--org_img_path=original.tif",
            "--pred_img_path=predicted.tif",
            "--metric=rmse",
            "--loader=opencv",
        ],
    )

    evaluate.main()

    assert observed == {"metrics": ["rmse"], "loader": "opencv"}
    assert json.loads(capsys.readouterr().out)["metrics"] == {"rmse": 1.25}
