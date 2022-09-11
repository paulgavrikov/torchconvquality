import torch
import numpy as np


def torch_entropy10(probs):
    e = 0
    for p in probs:
        e += p * torch.log10(p)
    return -e


def _entropy_max_threshold(n):
    l, x0, k, b = (1.2618047, 2.30436435, 0.88767525, -0.31050834)  # min distribution
    return l / (1 + np.exp(-k * (np.log2(n) - x0))) + b


def svd_variance_ratio(x):
    print(x.shape)
    s = torch.linalg.svdvals(x - x.mean(axis=0))
    print(s)
    variance = s ** 2 / (len(x) - 1)
    return variance / variance.sum()


def measure_layer_quality(conv_layer, sparsity_eps):
    info_dict = {}
    w = conv_layer.weight.view(-1, conv_layer.kernel_size[0] * conv_layer.kernel_size[1])
    n = w.shape[0]
    info_dict["n"] = n

    t = abs(w).max() * sparsity_eps
    new_layer = torch.ones_like(w)
    new_layer[abs(w) < t] = 0
    sparsity = (new_layer.sum(axis=1) == 0).sum() / n
    info_dict["sparsity"] = sparsity.item()

    entropy = torch_entropy10(svd_variance_ratio(w)) / _entropy_max_threshold(n)
    info_dict["variance_entropy"] = entropy.item()

    w_clean = w[new_layer.sum(axis=1) > 0]
    entropy_clean = torch_entropy10(svd_variance_ratio(w_clean)) / _entropy_max_threshold(len(w_clean))
    info_dict["variance_entropy_clean"] = entropy_clean.item()
    return info_dict


def measure_quality(model, sparsity_eps=0.01):
    info_dict = {}
    for name, conv_layer in filter(
            lambda x: type(x[1]) == torch.nn.Conv2d and x[1].kernel_size == (3, 3), model.named_modules()):
        info_dict[name] = measure_layer_quality(conv_layer, sparsity_eps)

    return info_dict
