import argparse
import json
import logging
import os
from typing import List

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
    if rasterio and (path.endswith(".tif") or path.endswith(".tiff")):
        return np.rollaxis(rasterio.open(path).read(), 0, 3)
    return cv2.imread(path)


def evaluation(org_img_path: str, pred_img_path: str, metrics: List[str]):
    output_dict = {}
    org_img = read_image(org_img_path)
    pred_img = read_image(pred_img_path)

    for metric in metrics:
        metric_func = metric_functions[metric]
        out_value = float(metric_func(org_img, pred_img))
        logger.info(f"{metric.upper()} value is: {out_value}")
        output_dict[metric] = out_value
    return output_dict


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
        choices=all_metrics + ["all"],
        metavar="METRIC",
        help="select an evaluation metric (%(choices)s) (can be repeated)",
    )
    args = parser.parse_args()
    if not args.metrics:
        args.metrics = ["psnr"]
    if "all" in args.metrics:
        args.metrics = all_metrics

    metric_values = evaluation(
        org_img_path=args.org_img_path,
        pred_img_path=args.pred_img_path,
        metrics=args.metrics,
    )
    result_dict = {
        "image1": args.org_img_path,
        "image2": args.pred_img_path,
        "metrics": metric_values,
    }
    print(json.dumps(result_dict, sort_keys=True))


if __name__ == "__main__":
    main()
