# torchconvquality

[![Latest Version](https://img.shields.io/pypi/v/torchconvquality.svg?color=green)](https://pypi.python.org/pypi/torchconvquality)
![GitHub](https://img.shields.io/github/license/paulgavrikov/torchconvquality?color=green)

*A library for PyTorch model convolution quality analysis.*


## Installation
To install published releases from PyPi execute:
```bash
pip install torchconvquality
```
To update torchconvquality to the latest available version, add the `--upgrade` flag to the above commands.

If you want the latest (potentially unstable) features you can also directly install from the github main branch:
```bash
pip install git+https://github.com/paulgavrikov/torchconvquality
```

## Usage

Just import the package and run `measure_quality` on your model. This function will return a dict with quality metrics for every 2D-Conv-Layer with 3x3 kernel. Note: theoretically, we could also extend this to large kernel sites. Let us know through a GitHub issue if you are interested in that beeing implemented.

```
from torchconvquality import measure_quality

model = ... # your model
quality_dict = measure_quality(model)
```

Supported Metrics:

### Sparsity

*Sparsity* measures the ratio of 2D Filters with a $l_\infty$-norm that is lower than 1% of the highest norm in that layer. These filters will most likely not contribute to your learned function beyond noise. You should minimize this value if you are interested in exploiting all of your available model capacity. On the other hand, larger sparsity values allow you to successfully prune many weights.

### Variance Entropy

*Variance Entropy*  captures the difference in filter patterns in your conv layer. We have observed that significantly overparameterized networks learn many redundand filters in deeper layers. Hence we assume that, generally, you'd like to increase diversity. A good value is somewhere around 0.9 - this means that the layer in question has learned a filter distribution that is signifincantly different from random. A value close to 0 indicates highly redudand filters. A value over 1 indicates a random distribution that you'd find prior to any training (i.e. right after initialization) or in GAN-Discriminator at the end of training (when it can no longer distinguish between real and fake inputs).

## Citation

Please consider citing our publication if this libary was helpfull to you.
```
@InProceedings{Gavrikov_2022_CVPR,
    author    = {Gavrikov, Paul and Keuper, Janis},
    title     = {CNN Filter DB: An Empirical Investigation of Trained Convolutional Filters},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {19066-19076}
}
```
