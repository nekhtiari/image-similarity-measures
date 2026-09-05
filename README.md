# Image Similarity Measures

[![CI](https://github.com/nekhtiari/image-similarity-measures/actions/workflows/python.yml/badge.svg)](https://github.com/nekhtiari/image-similarity-measures/actions/workflows/python.yml)
[![PyPI](https://img.shields.io/pypi/v/image-similarity-measures)](https://pypi.org/project/image-similarity-measures/)
[![Python](https://img.shields.io/pypi/pyversions/image-similarity-measures)](https://pypi.org/project/image-similarity-measures/)

A Python library and command-line tool for comparing images with eight full-reference image-quality metrics:

- Feature Similarity Index (FSIM)
- Information theoretic-based Statistic Similarity Measure (ISSM)
- Peak Signal-to-Noise Ratio (PSNR)
- Root Mean Square Error (RMSE)
- Spectral Angle Mapper (SAM)
- Signal-to-Reconstruction Error ratio (SRE)
- Structural Similarity Index (SSIM)
- Universal Image Quality Index (UIQ)

The library was created for multi-band remote-sensing imagery. Arrays are expected in channel-last order: `(rows, columns, bands)`.

## Installation

Image Similarity Measures supports Python 3.10 and newer.

```console
python -m pip install image-similarity-measures
```

Optional dependencies are available for geospatial TIFF loading and faster FSIM evaluation:

```console
python -m pip install "image-similarity-measures[rasterio]"
python -m pip install "image-similarity-measures[speedups]"
python -m pip install "image-similarity-measures[rasterio,speedups]"
```

## Command-line usage

Evaluate every metric:

```console
image-similarity-measures \
  --org_img_path=original.tif \
  --pred_img_path=prediction.tif
```

Select one or more metrics by repeating `--metric`:

```console
image-similarity-measures \
  --org_img_path=original.tif \
  --pred_img_path=prediction.tif \
  --metric=rmse \
  --metric=psnr
```

The command prints machine-readable JSON to standard output.

## Python usage

Evaluate files:

```python
from image_similarity_measures.evaluate import evaluation

results = evaluation(
    org_img_path="original.tif",
    pred_img_path="prediction.tif",
    metrics=["rmse", "psnr"],
)
```

Or call an individual metric with NumPy arrays:

```python
import numpy as np

from image_similarity_measures.quality_metrics import rmse

original = np.random.default_rng(0).random((32, 32, 3))
prediction = np.random.default_rng(1).random((32, 32, 3))
score = rmse(original, prediction, max_p=1)
```

`max_p` defaults to `4095` for RMSE, PSNR, and SSIM because the original use case was 12-bit imagery. Pass `255` for 8-bit images or `1` for normalized floating-point images where appropriate.

## Important TIFF loading behavior

For compatibility with existing releases, installing the `rasterio` extra changes how lowercase `.tif` and `.tiff` files are read:

- with Rasterio installed, TIFFs retain their bands and native dtype;
- otherwise, OpenCV reads them and may change the band count, dtype, or value range;
- other file types use OpenCV.

Consequently, installing an optional dependency can change metric results for the same TIFF files. Pin your environment and loader dependencies when reproducibility matters. See [COMPATIBILITY.md](COMPATIBILITY.md) for the compatibility policy and documented legacy behavior.

## Contributing and security

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change, especially a numerical one.

Report security problems using the private process in [SECURITY.md](SECURITY.md). For release history, see [CHANGELOG.md](CHANGELOG.md).

## Citation

If this package supports your research, please cite:

> Müller, M. U., Ekhtiari, N., Almeida, R. M., and Rieke, C. (2020). Super-resolution of multispectral satellite images using convolutional neural networks. *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, V-1-2020, 33–40. <https://doi.org/10.5194/isprs-annals-V-1-2020-33-2020>

## License

Image Similarity Measures is distributed under the [MIT License](LICENSE).
