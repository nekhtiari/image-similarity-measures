import os
import argparse
import numpy as np
import rasterio as rio
import cv2

from image_similarity_measures.quality_metrics import metric_functions

import logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name, level=logging.DEBUG):
    """
    This method creates logger object and sets the default log level to DEBUG.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # create console handler and set level to debug
    c_h = logging.StreamHandler()
    c_h.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    c_h.setFormatter(formatter)
    logger.addHandler(c_h)
    return logger


logger = get_logger(__name__)


def write_final_dict(metric, metric_dict):
    # Create a directory to save the text file of including evaluation values.
    predict_path = "metrics_value/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    with open(os.path.join(predict_path, metric + '.txt'), 'w') as f:
        f.writelines('{}\n'.format(v) for _, v in metric_dict.items())


def read_image(path):
    logger.info("Reading image %s", os.path.basename(path))
    if path.endswith(".tif") or path.endswith(".tiff"):
        return np.rollaxis(rio.open(path).read(), 0, 3)
    return cv2.imread(path)


def evaluation(org_img_path, pred_img_path, metrics, write_to_file):
    org_img = read_image(org_img_path)
    pred_img = read_image(pred_img_path)

    for metric in metrics:
        metric_func = metric_functions[metric]
        out_value = metric_func(org_img, pred_img)
        logger.info(f"{metric.upper()} value is: {out_value}")
        if write_to_file:
            metric_dict = {metric: {f"{metric.upper()}": out_value}}
            write_final_dict(metric, metric_dict)


def main():
    all_metrics = sorted(metric_functions.keys())
    parser = argparse.ArgumentParser(description="Evaluates an Image Super Resolution Model")
    parser.add_argument("--org_img_path", help="Path to original input image", required=True, metavar="FILE")
    parser.add_argument("--pred_img_path", help="Path to predicted image", required=True, metavar="FILE")
    parser.add_argument("--metric", dest="metrics", action="append",
                        choices=all_metrics + ['all'], metavar="METRIC",
                        help="select an evaluation metric (%(choices)s) (can be repeated)")
    parser.add_argument("--write_to_file", action="store_true", help="final output will be written to a file.")
    args = parser.parse_args()
    if not args.metrics:
        args.metrics = ["psnr"]
    if "all" in args.metrics:
        args.metrics = all_metrics

    evaluation(
        org_img_path=args.org_img_path,
        pred_img_path=args.pred_img_path,
        metrics=args.metrics,
        write_to_file=args.write_to_file,
    )

if __name__ == "__main__":
    main()
