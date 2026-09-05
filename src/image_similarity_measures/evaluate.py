import argparse
import json
import logging
import os
from typing import List, Literal

import cv2
import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

from image_similarity_measures.quality_metrics import metric_functions

logger = logging.getLogger(__name__)


def read_image(path: str):
    logger.info(f"Reading image {os.path.basename(path)}")
    if rasterio and path.endswith((".tif", ".tiff")):
        return np.rollaxis(rasterio.open(path).read(), 0, 3)
    return cv2.imread(path)


def read_image_with_loader(
    path: str, loader: Literal["auto", "opencv", "rasterio"] = "auto"
):
    """Read an image with an explicit loader while retaining legacy auto behavior."""
    if loader == "auto":
        return read_image(path)

    logger.info(f"Reading image {os.path.basename(path)} with {loader}")
    if loader == "opencv":
        return cv2.imread(path)
    if loader == "rasterio":
        if rasterio is None:
            raise ImportError(
                "Rasterio is required for loader='rasterio'. Install "
                "image-similarity-measures[rasterio]."
            )
        with rasterio.open(path) as dataset:
            return np.rollaxis(dataset.read(), 0, 3)
    raise ValueError("loader must be one of: auto, opencv, rasterio")


def _evaluate_images(org_img, pred_img, metrics: List[str]):
    output_dict = {}
    for metric in metrics:
        metric_func = metric_functions[metric]
        out_value = float(metric_func(org_img, pred_img))
        logger.info(f"{metric.upper()} value is: {out_value}")
        output_dict[metric] = out_value
    return output_dict


def evaluation(org_img_path: str, pred_img_path: str, metrics: List[str]):
    org_img = read_image(org_img_path)
    pred_img = read_image(pred_img_path)
    return _evaluate_images(org_img, pred_img, metrics)


def evaluation_with_loader(
    org_img_path: str,
    pred_img_path: str,
    metrics: List[str],
    loader: Literal["auto", "opencv", "rasterio"] = "auto",
):
    """Evaluate image files with a deterministic loader selection."""
    if loader == "auto":
        return evaluation(org_img_path, pred_img_path, metrics)

    org_img = read_image_with_loader(org_img_path, loader)
    pred_img = read_image_with_loader(pred_img_path, loader)
    return _evaluate_images(org_img, pred_img, metrics)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    all_metrics = sorted(metric_functions.keys())
    parser = argparse.ArgumentParser(
        description="Evaluates an Image Super Resolution Model"
    )
    parser.add_argument(
        "--org_img_path",
        help="Path to original input image",
        required=True,
        metavar="FILE",
    )
    parser.add_argument(
        "--pred_img_path", help="Path to predicted image", required=True, metavar="FILE"
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        choices=[*all_metrics, "all"],
        metavar="METRIC",
        help="select an evaluation metric (%(choices)s) (can be repeated)",
    )
    parser.add_argument(
        "--loader",
        choices=["auto", "opencv", "rasterio"],
        default="auto",
        help="select the image loader (default: %(default)s)",
    )
    args = parser.parse_args()
    if not args.metrics:
        args.metrics = ["psnr"]
    if "all" in args.metrics:
        args.metrics = all_metrics

    if args.loader == "auto":
        metric_values = evaluation(
            org_img_path=args.org_img_path,
            pred_img_path=args.pred_img_path,
            metrics=args.metrics,
        )
    else:
        metric_values = evaluation_with_loader(
            org_img_path=args.org_img_path,
            pred_img_path=args.pred_img_path,
            metrics=args.metrics,
            loader=args.loader,
        )
    result_dict = {
        "image1": args.org_img_path,
        "image2": args.pred_img_path,
        "metrics": metric_values,
    }
    print(json.dumps(result_dict, sort_keys=True))


if __name__ == "__main__":
    main()
