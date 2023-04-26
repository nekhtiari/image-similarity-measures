# Image Similarity Measures

Python package and commandline tool to evaluate the similarity between two images with eight evaluation metrics:

 * <i><a href="https://en.wikipedia.org/wiki/Root-mean-square_deviation">Root mean square error (RMSE)</a></i>
 * <i><a href="https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio">Peak signal-to-noise ratio (PSNR)</a></i>
 * <i><a href="https://en.wikipedia.org/wiki/Structural_similarity">Structural Similarity Index (SSIM)</a></i>
 * <i><a href="https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM.pdf">Feature-based similarity index (FSIM)</a></i>
 * <i><a href="https://www.tandfonline.com/doi/full/10.1080/22797254.2019.1628617">Information theoretic-based Statistic Similarity Measure (ISSM)</a></i>
 * <i><a href="https://www.sciencedirect.com/science/article/abs/pii/S0924271618302636">Signal to reconstruction error ratio (SRE)</a></i>
 * <i><a href="https://ntrs.nasa.gov/citations/19940012238">Spectral angle mapper (SAM)</a></i>
 * <i><a href="https://ece.uwaterloo.ca/~z70wang/publications/quality_2c.pdf">Universal image quality index (UIQ)</a></i>

## Installation

Supports Python >=3.8.

```bash
pip install image-similarity-measures
```

*Optional*: For faster evaluation of the FSIM metric, the `pyfftw` package is required, install via:

```bash
pip install image-similarity-measures[speedups]
```

*Optional*: For reading TIFF images with `rasterio` instead of `OpenCV`, install:

```bash
pip install image-similarity-measures[rasterio]
```


## Usage on commandline

To evaluate the similarity beteween two images, run on the commandline:

```bash
image-similarity-measures --org_img_path=a.tif --pred_img_path=b.tif
```

**Note** that images that are used for evaluation should be **channel last**. The results are printed in 
machine-readable JSON, so you can redirect the output of the command into a file.

#### Parameters
```
  --org_img_path FILE   Path to original input image
  --pred_img_path FILE  Path to predicted image
  --metric METRIC       select an evaluation metric (fsim, issm, psnr, rmse,
                        sam, sre, ssim, uiq, all) (can be repeated)
```

## Usage in Python

```bash
from image_similarity_measures.evaluate import evaluation

evaluation(org_img_path="example/lafayette_org.tif", 
           pred_img_path="example/lafayette_pred.tif", 
           metrics=["rmse", "psnr"])
```

```bash
from image_similarity_measures.quality_metrics import rmse

rmse(org_img=np.random.rand(3,2,1), pred_img=np.random.rand(3,2,1))
```

## Contribute

Contributions are welcome! Please see README-dev.md for instructions.


## Citation
Please use the following for citation purposes of this codebase:

<strong>Müller, M. U., Ekhtiari, N., Almeida, R. M., and Rieke, C.: SUPER-RESOLUTION OF MULTISPECTRAL
SATELLITE IMAGES USING CONVOLUTIONAL NEURAL NETWORKS, ISPRS Ann. Photogramm. Remote Sens.
Spatial Inf. Sci., V-1-2020, 33–40, https://doi.org/10.5194/isprs-annals-V-1-2020-33-2020, 2020.</strong>
