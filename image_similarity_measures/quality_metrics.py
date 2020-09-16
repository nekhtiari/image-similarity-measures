"""
This module is a collection of metrics to assess the similarity between two images.
PSNR, SSIM, FSIM and ISSM are the current metrics that are implemented in this module.
"""
import math

import numpy as np
from skimage.metrics import structural_similarity
import phasepack.phasecong as pc
import cv2


def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    msg = (f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
           f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}")

    assert org_img.shape == pred_img.shape, msg


def rmse(org_img: np.ndarray, pred_img: np.ndarray, data_range=4096):
    """
    Root Mean Squared Error
    """

    _assert_image_shapes_equal(org_img, pred_img, "RMSE")
    rmse_final = []
    for i in range(org_img.shape[2]):
        m = np.mean(np.square((org_img[:, :, i] - pred_img[:, :, i]) / data_range))
        s = np.sqrt(m)
        rmse_final.append(s)
    return np.mean(rmse_final)


def psnr(org_img: np.ndarray, pred_img: np.ndarray, data_range=4096):
    """
    Peek Signal to Noise Ratio, a measure similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When using 12-bit imagery MaxP is 4096, for 8-bit imagery 256
    """
    _assert_image_shapes_equal(org_img, pred_img, "PSNR")

    r = []
    for i in range(org_img.shape[2]):
        val = 20 * np.log10(data_range) - 10. * np.log10(np.mean(np.square(org_img[:, :, i] - pred_img[:, :, i])))
        r.append(val)

    return np.mean(r)


def _similarity_measure(x, y, constant):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator


def _gradient_magnitude(img: np.ndarray, img_depth):
    """
    Calculate gradient magnitude based on Scharr operator
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)


def fsim(org_img: np.ndarray, pred_img: np.ndarray):
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)

    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.

    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.
    """
    _assert_image_shapes_equal(org_img, pred_img, "FSIM")

    T1 = 0.85  # a constant based on the dynamic range PC
    T2 = 160  # a constant based on the dynamic range GM
    alpha = beta = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(org_img.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        pc2_2dim = pc(pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros((pred_img.shape[0], pred_img.shape[1]), dtype=np.float64)
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)


def _ehs(x, y):
    """
    Entropy-Histogram Similarity measure
    """
    H = (np.histogram2d(x.flatten(), y.flatten()))[0]

    return -np.sum(np.nan_to_num(H * np.log2(H)))


def _edge_c(x, y):
    """
    Edge correlation coefficient based on Canny detector
    """
    # Use 100 and 200 as thresholds, no indication in the paper what was used
    g = cv2.Canny((x * 0.0625).astype(np.uint8), 100, 200)
    h = cv2.Canny((y * 0.0625).astype(np.uint8), 100, 200)

    g0 = np.mean(g)
    h0 = np.mean(h)

    numerator = np.sum((g - g0) * (h - h0))
    denominator = np.sqrt(np.sum(np.square(g-g0)) * np.sum(np.square(h-h0)))

    return numerator / denominator


def issm(org_img: np.ndarray, pred_img: np.ndarray):
    """
    Information theoretic-based Statistic Similarity Measure
    """
    _assert_image_shapes_equal(org_img, pred_img, "ISSM")

    # Variable names closely follow original paper for better readability
    x = org_img
    y = pred_img
    A = 0.3
    B = 0.5
    C = 0.7
    
    ehs_val = _ehs(x, y)
    canny_val = _edge_c(x, y)

    numerator = canny_val * ehs_val * (A + B) + math.e
    denominator = A * canny_val * ehs_val + B * ehs_val + C * ssim(x, y) + math.e

    return np.nan_to_num(numerator / denominator)


def ssim(org_img: np.ndarray, pred_img: np.ndarray, data_range=4096):
    """
    Structural SIMularity index
    """
    _assert_image_shapes_equal(org_img, pred_img, "SSIM")

    return structural_similarity(org_img, pred_img, data_range=data_range, multichannel=True)


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def uiq(org_img: np.ndarray, pred_img: np.ndarray):
    """
    Universal Image Quality index
    """
    # TODO: Apply optimization, right now it is very slow
    _assert_image_shapes_equal(org_img, pred_img, "UIQ")
    q_all = []
    for (x, y, window_org), (x, y, window_pred) in zip(sliding_window(org_img, stepSize=1, windowSize=(8, 8)),
                                                       sliding_window(pred_img, stepSize=1, windowSize=(8, 8))):
        # if the window does not meet our desired window size, ignore it
        if window_org.shape[0] != 8 or window_org.shape[1] != 8:
            continue
        org_img_mean = np.mean(org_img)
        pred_img_mean = np.mean(pred_img)
        org_img_variance = np.var(org_img)
        pred_img_variance = np.var(pred_img)
        org_pred_img_variance = np.mean((window_org - org_img_mean) * (window_pred - pred_img_mean))

        numerator = 4 * org_pred_img_variance * org_img_mean * pred_img_mean
        denominator = (org_img_variance + pred_img_variance) * (org_img_mean**2 + pred_img_mean**2)

        if denominator != 0.0:
            q = numerator / denominator
            q_all.append(q)

    return np.mean(q_all)


def sam(org_img: np.ndarray, pred_img: np.ndarray):
    """
    calculates spectral angle mapper
    """
    _assert_image_shapes_equal(org_img, pred_img, "SAM")
    org_img = org_img.reshape((org_img.shape[0] * org_img.shape[1], org_img.shape[2]))
    pred_img = pred_img.reshape((pred_img.shape[0] * pred_img.shape[1], pred_img.shape[2]))

    N = org_img.shape[1]
    sam_angles = np.zeros(N)
    for i in range(org_img.shape[1]):
        val = np.clip(np.dot(org_img[:, i], pred_img[:, i]) / (np.linalg.norm(org_img[:, i]) * np.linalg.norm(pred_img[:, i])), -1, 1)
        sam_angles[i] = np.arccos(val)

    return np.mean(sam_angles * 180.0 / np.pi)


def sre(org_img: np.ndarray, pred_img: np.ndarray):
    """
    signal to reconstruction error ratio
    """
    _assert_image_shapes_equal(org_img, pred_img, "SRE")

    sre_final = []
    for i in range(org_img.shape[2]):
        numerator = np.square(np.mean(org_img[:, :, i]))
        denominator = ((np.linalg.norm(org_img[:, :, i] - pred_img[:, :, i]))) /\
                      (org_img.shape[0] * org_img.shape[1])
        sre_final.append(10 * np.log10(numerator/denominator))

    return np.mean(sre_final)
