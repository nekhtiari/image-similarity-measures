# Image Similarity Measures

Implementation of eight evaluation metrics to access the similarity between two images. The eight metrics are as follows:

 * <i><a href="https://en.wikipedia.org/wiki/Root-mean-square_deviation">Root mean square error (RMSE)</a></i>,
 * <i><a href="https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio">Peak signal-to-noise ratio (PSNR)</a></i>,
 * <i><a href="https://en.wikipedia.org/wiki/Structural_similarity">Structural Similarity Index (SSIM)</a></i>,
 * <i><a href="https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM.pdf">Feature-based similarity index (FSIM)</a></i>,
 * <i><a href="https://www.tandfonline.com/doi/full/10.1080/22797254.2019.1628617">Information theoretic-based Statistic Similarity Measure (ISSM)</a></i>,
 * <i><a href="https://www.sciencedirect.com/science/article/abs/pii/S0924271618302636">Signal to reconstruction error ratio (SRE)</a></i>,
 * <i><a href="https://ntrs.nasa.gov/citations/19940012238">Spectral angle mapper (SAM)</a></i>, and
 * <i><a href="https://ece.uwaterloo.ca/~z70wang/publications/quality_2c.pdf">Universal image quality index (UIQ)</a></i>

## Instructions

The following step-by-step instructions will guide you through installing this package and run evaluation using the command line tool.

**Note:** Supported python versions are 3.6, 3.7, 3.8, and 3.9.

### Install package

```bash
pip install image-similarity-measures
```

For faster evaluation of the FSIM metric, the `pyfftw` package is required.
You can install it separately, or via the `speedups` extra:

```bash
pip install image-similarity-measures[speedups]
```

You may also install the `rasterio` package to allow the CLI tool to use it for reading TIFF
images instead of OpenCV. It, too, is available as an extra:


```bash
pip install image-similarity-measures[rasterio]
```


### Usage

#### Parameters
```
  --org_img_path FILE   Path to original input image
  --pred_img_path FILE  Path to predicted image
  --metric METRIC       select an evaluation metric (fsim, issm, psnr, rmse,
                        sam, sre, ssim, uiq, all) (can be repeated)
```

#### Evaluation
For doing the evaluation, you can easily run the following command:

```bash
image-similarity-measures --org_img_path=a.tif --pred_img_path=b.tif
```
The results are printed in machine-readable JSON, so you can redirect the output of the command into a file.

**Note** that images that are used for evaluation should be **channel last**.

#### Usage in python
```bash
import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, psnr
```

### Install package from source

#### Clone the repository

```bash
git clone https://github.com/up42/image-similarity-measures.git
cd image-similarity-measures
```

Then navigate to the folder via `cd image-similarity-measures`.

#### Installing the required libraries

First create a new virtual environment called `similarity-measures`, for example by using
[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/):

```bash
mkvirtualenv --python=$(which python3.7) similarity-measures
```

Activate the new environment:

```bash
workon similarity-measures
```

Install the necessary Python libraries via:

```bash
bash setup.sh
```

## Citation
Please use the following for citation purposes of this codebase:

<strong>Müller, M. U., Ekhtiari, N., Almeida, R. M., and Rieke, C.: SUPER-RESOLUTION OF MULTISPECTRAL
SATELLITE IMAGES USING CONVOLUTIONAL NEURAL NETWORKS, ISPRS Ann. Photogramm. Remote Sens.
Spatial Inf. Sci., V-1-2020, 33–40, https://doi.org/10.5194/isprs-annals-V-1-2020-33-2020, 2020.</strong>
