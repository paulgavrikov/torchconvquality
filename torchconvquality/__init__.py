import torch
import numpy as np


def _torch_entropy10(probs):
    e = 0
    for p in probs:
        e += p * torch.log10(p)
    return -e


def _entropy_max_threshold(n):
    l, x0, k, b = (1.2618047, 2.30436435, 0.88767525, -0.31050834)  # min distribution
    return l / (1 + np.exp(-k * (np.log2(n) - x0))) + b


def _svd_variance_ratio(x):
    s = torch.linalg.svdvals(x - x.mean(axis=0))
    variance = s ** 2 / (len(x) - 1)
    return variance / variance.sum()


def sparsity(w, sparsity_eps=0.01, **kwargs):
    n = w.shape[0]
    t = abs(w).max() * sparsity_eps
    new_layer = torch.ones_like(w)
    new_layer[abs(w) < t] = 0
    sparse_mask = (new_layer.sum(axis=1) == 0)
    sparsity = sparse_mask.sum() / n
    return sparsity.item(), sparse_mask


def variance_entropy(w, **kwargs):
    n = w.shape[0]
    ratio = _svd_variance_ratio(w)
    entropy = _torch_entropy10(ratio) / _entropy_max_threshold(n)
    return entropy.item()


def measure_conv_weight_quality(w, **kwargs):
    info_dict = {}
    n = w.shape[0]
    sparsity_ratio, sparse_mask = sparsity(w, **kwargs)

    info_dict["n"] = n
    info_dict["sparsity"] = sparsity_ratio
    info_dict["variance_entropy"] = variance_entropy(w, **kwargs)
    info_dict["variance_entropy_clean"] = variance_entropy(w[~sparse_mask], **kwargs)

    return info_dict


def measure_layer_quality(conv_layer, **kwargs):
    w = conv_layer.weight.view(-1, conv_layer.kernel_size[0] * conv_layer.kernel_size[1])
    info_dict = measure_conv_weight_quality(w, **kwargs)
    return info_dict


def measure_quality(model, **kwargs):
    info_dict = {}
    for name, conv_layer in filter(
            lambda x: type(x[1]) == torch.nn.Conv2d and x[1].kernel_size == (3, 3), model.named_modules()):
        info_dict[name] = measure_layer_quality(conv_layer, **kwargs)

    return info_dict
