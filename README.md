# torchconvquality

[![Latest Version](https://img.shields.io/pypi/v/torchconvquality.svg?color=green)](https://pypi.python.org/pypi/torchconvquality)
[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]


[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

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

Here is an example output (pretrained ResNet-18 on ImageNet):

```python
{'layer1.0.conv1': {'sparsity': 0.125244140625,
                    'variance_entropy': 0.8243831176467854},
 'layer1.0.conv2': {'sparsity': 0.0, 'variance_entropy': 0.8540944028708247},
 'layer1.1.conv1': {'sparsity': 0.0, 'variance_entropy': 0.880116579714338},
 'layer1.1.conv2': {'sparsity': 0.0, 'variance_entropy': 0.8770092802517852},
 'layer2.0.conv1': {'sparsity': 0.0, 'variance_entropy': 0.9162120601419921},
 'layer2.0.conv2': {'sparsity': 0.0, 'variance_entropy': 0.79917093039702},
 'layer2.1.conv1': {'sparsity': 0.0, 'variance_entropy': 0.8988180721697099},
 'layer2.1.conv2': {'sparsity': 0.0, 'variance_entropy': 0.8584897149301801},
 'layer3.0.conv1': {'sparsity': 0.0, 'variance_entropy': 0.589569852560285},
 'layer3.0.conv2': {'sparsity': 0.0, 'variance_entropy': 0.7655632562758724},
 'layer3.1.conv1': {'sparsity': 0.0, 'variance_entropy': 0.8485658915907506},
 'layer3.1.conv2': {'sparsity': 1.52587890625e-05,
                    'variance_entropy': 0.7960795856993427},
 'layer4.0.conv1': {'sparsity': 7.62939453125e-06,
                    'variance_entropy': 0.6701797219658017},
 'layer4.0.conv2': {'sparsity': 7.62939453125e-06,
                    'variance_entropy': 0.8185696588740375},
 'layer4.1.conv1': {'sparsity': 0.0, 'variance_entropy': 0.6583874160290571},
 'layer4.1.conv2': {'sparsity': 0.001796722412109375,
                    'variance_entropy': 0.21928562164990348}}
```

### Supported Metrics

#### Sparsity

*Sparsity* measures the ratio of 2D Filters with a $l_\infty$-norm that is lower than 1% of the highest norm in that layer. These filters will most likely not contribute to your learned function beyond noise. You should minimize this value if you are interested in exploiting all of your available model capacity. On the other hand, larger sparsity values allow you to successfully prune many weights.

#### Variance Entropy

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

## Legal

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

Funded by the Ministry for Science, Research and Arts, Baden-Wuerttemberg, Germany Grant 32-7545.20/45/1 (Q-AMeLiA).
