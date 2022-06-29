import torch
import numpy as np
import scipy.stats


def _entropy_max_threshold(n):
    l, x0, k, b = (1.2618047, 2.30436435, 0.88767525, -0.31050834)  # min distribution
    return l / (1 + np.exp(-k * (np.log2(n) - x0))) + b


def measure_quality(model, sparsity_eps=0.01):
    info_dict = {}
    for name, conv_layer in filter(
            lambda x: type(x[1]) == torch.nn.Conv2d and x[1].kernel_size == (3, 3), model.named_modules()):
        info_dict[name] = {}

        w = conv_layer.weight.detach().cpu().numpy().reshape(-1, conv_layer.kernel_size[0] * conv_layer.kernel_size[1])

        n = w.shape[0]

        s = np.linalg.svd(w - w.mean(axis=0), full_matrices=False, compute_uv=False)
        v = s ** 2 / (n - 1)
        entropy = scipy.stats.entropy(v, base=10) / _entropy_max_threshold(n)

        info_dict[name]["variance_entropy"] = entropy

        t = np.abs(w).max() * sparsity_eps
        new_layer = np.ones_like(w)
        new_layer[np.abs(w) < t] = 0
        sparsity = (new_layer.sum(axis=1) == 0).sum() / n

        info_dict[name]["sparsity"] = sparsity

    return info_dict
