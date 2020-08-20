# Image Similarity measure

Implementation of four evaluation metrics to access the similarity between two images. The four metrics are as follows:

<i><a href="https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio">Peak signal-to-noise ratio (PSNR)</a></i>, 
<i><a href="https://en.wikipedia.org/wiki/Structural_similarity">Structural Similarity Index (SSIM)</a></i>,
<i><a href="https://www.tandfonline.com/doi/full/10.1080/22797254.2019.1628617">Information theoretic-based Statistic Similarity Measure (ISSM)</a></i>, and 
<i><a href="https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM.pdf">Feature-based similarity index (FSIM)</a></i>, 


## Instructions

The following step-by-step instructions will guide you through setting up the proper environment and libraries to be
able to run these scripts.

### Clone the repository

```bash
git clone https://github.com/up42/image-similarity-measures.git
cd image-similarity-measures
```

Then navigate to the folder via `cd image-similarity-measures`.

### Installing the required libraries

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

## Parameters
```
--org_img_path : Path to the original image.
--pred_img_path : Path to the predicted or disordered image which is created from the original image.
--metric= : Name of the evaluation metric. Default set to be psnr. It can be one of the following: psnr, ssim, issm, fsim.
--mode : Image format. Default set to be "tif". can be one of the following: "tif", or "png", or "jpg". 
--write_to_file : The final result will be written to a file. Set to False if you don't want a final file.
```


## Evaluation
For doing the evaluation, you can easily run the following command:
```bash
python src/evaluate.py --org_img_path=path_to_first_img --pred_img_path=path_to_second_img --mode=tif
```
If you want to save the final result in a file you can add `--write_to_file` at then end of above command.

## Citation
Please use the following for citation purposes of this codebase:

<strong>Müller, M. U., Ekhtiari, N., Almeida, R. M., and Rieke, C.: SUPER-RESOLUTION OF MULTISPECTRAL
SATELLITE IMAGES USING CONVOLUTIONAL NEURAL NETWORKS, ISPRS Ann. Photogramm. Remote Sens.
Spatial Inf. Sci., V-1-2020, 33–40, https://doi.org/10.5194/isprs-annals-V-1-2020-33-2020, 2020.</strong>