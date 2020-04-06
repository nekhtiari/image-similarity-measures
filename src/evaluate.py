from __future__ import print_function, division
import os
import argparse
from glob import glob
from pathlib import Path
import numpy as np
import rasterio as rio
import cv2

from quality_metrics import psnr, ssim, fsim, issm

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


def read_tif(img_path, swap_axes=False):
    if swap_axes:
        img = rio.open(img_path).read()
        img = np.rollaxis(img, 0, 3)
    else:
        img = rio.open(img_path).read()
    return img


def read_png(img_path):
    img = cv2.imread(img_path)
    return img


def write_final_dict(metric, metric_dict):
    # Create a directory to save the text file of including evaluation values.
    predict_path = "metrics_value/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    with open(os.path.join(predict_path, metric + '_' + '.txt'), 'w') as f:
        f.writelines('{}:{}\n'.format(k, v) for k, v in metric_dict.items())


def evaluation(org_img_path, pred_img_path, mode, write_to_file):
    metric_dict = {}

    if mode == "tif":
        logging.info("Reading image %s", Path(org_img_path).stem)
        #org_img = read_tif(org_img_path, swap_axes=True)
        org_img = upsample(org_img_path, pred_img_path, swap_axes=True)
        logging.info("Reading image %s", Path(pred_img_path).stem)
        pred_img = read_tif(pred_img_path, swap_axes=True)

    if mode == "png":
        logging.info("Reading image %s", Path(org_img_path).stem)
        org_img = read_png(org_img_path)
        logging.info("Reading image %s", Path(pred_img_path).stem)
        pred_img = read_png(pred_img_path)

    if metric == 'psnr':
        psnr_value = psnr(org_img, pred_img)
        logger.info("PSNR value is: %s", psnr_value)
        if write_to_file:
            metric_dict[metric] = {'PSNR': psnr_value}

    if metric == 'ssim':
        ssim_value = ssim(org_img, pred_img)
        logger.info("SSIM value is: %s", ssim_value)
        if write_to_file:
            metric_dict[metric] = {'SSIM': ssim_value}

    if metric == 'fsim':
        fsim_value = fsim(org_img, pred_img)
        logger.info("FSIM value is: %s", fsim_value)
        if write_to_file:
            metric_dict[metric] = {'FSIM': fsim_value}

    if metric == 'issm':
        issm_value = issm(org_img, pred_img)
        logger.info("ISSM value is: %s", issm_value)
        if write_to_file:
            metric_dict[metric] = {'ISSM': issm_value}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluates an Image Super Resolution Model")
    parser.add_argument("--org_img_path", type=str, help="Path to original pan-sharpened and ms input image")
    parser.add_argument("--pred_img_path", help="Path to output of SR model")
    parser.add_argument("--metric", type=str, default="psnr", help="use psnr, ssim, fsim or issm as evaluation metric")
    parser.add_argument("--mode", type=str, default="tif", help="format of image, use either tif, or png, or jpg")
    parser.add_argument("--write_to_file", type=bool, default=True, help="final output will be written to a file.")
    args = parser.parse_args()

    orgpath = args.org_img_path
    predpath = args.pred_img_path
    metric = args.metric
    mode = args.mode
    write_to_file = args.write_to_file

    evaluation(orgpath, predpath, mode, write_to_file)

