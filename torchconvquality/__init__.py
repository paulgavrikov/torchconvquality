from distutils.command.clean import clean
import torch
import numpy as np
from typing import Tuple, Union


def _torch_entropy10(probs: torch.Tensor) -> float:
    e = 0
    for p in probs:
        e += p * torch.log10(p)
    return -e.item()


def _entropy_max_threshold(n: int) -> float:
    l, x0, k, b = (1.2618047, 2.30436435, 0.88767525, -0.31050834)  # min distribution
    return l / (1 + np.exp(-k * (np.log2(n) - x0))) + b


def _svd_variance_ratio(x: torch.Tensor) -> torch.Tensor:
    s = torch.linalg.svd((x - x.mean(dim=0)), full_matrices=False).S
    variance = s ** 2 / (len(x) - 1)
    den = variance.sum()
    if den == 0:
        den = 1
    return variance / den


def sparsity(w: torch.Tensor, sparsity_eps: float = 0.01, **kwargs) -> Tuple[float, torch.Tensor]:

    assert (
        len(w.shape) == 4
    ), "w must be a tensor of shape (out_channels, in_channels, kernel_size, kernel_size)"

    w = w.view(w.shape[0] * w.shape[1], w.shape[2] * w.shape[3])
    n = w.shape[0]
    t = abs(w).max().item() * sparsity_eps
    if t == 0:
        t = 1
    new_layer = torch.ones_like(w)
    new_layer[abs(w) < t] = 0
    sparse_mask = (new_layer.sum(dim=1) == 0)
    sparsity = sparse_mask.sum() / n
    return sparsity.item(), sparse_mask


def variance_entropy(w: torch.Tensor, mask: Union[torch.Tensor, None] = None, **kwargs) -> float:

    assert (
        len(w.shape) == 4
    ), "w must be a tensor of shape (out_channels, in_channels, kernel_size, kernel_size)"

    w = w.view(w.shape[0] * w.shape[1], w.shape[2] * w.shape[3])
    if mask is not None:
        w = w[mask]
    n = w.shape[0]
    ratio = _svd_variance_ratio(w)
    entropy = 0.0
    if ratio.sum() != 0:
        entropy = _torch_entropy10(ratio)
    return entropy, entropy /_entropy_max_threshold(n)


def orthogonality(w: torch.Tensor, **kwargs) -> float:

    assert (
        len(w.shape) == 4
    ), "w must be a tensor of shape (out_channels, in_channels, kernel_size, kernel_size)"

    w = w.view(w.shape[0], -1)
    n = w.shape[0]
    t = torch.linalg.norm(w, axis=1, ord=2).unsqueeze(1)
    t[t == 0] = 1
    w_norm = w / t
    c = abs(torch.matmul(w_norm, w_norm.T) - torch.eye(n)).sum() / (n * (n - 1))
    return (1 - c).item()


def measure_conv_weight_quality(w: torch.Tensor, **kwargs) -> dict:

    assert (
        len(w.shape) == 4
    ), "w must be a tensor of shape (out_channels, in_channels, kernel_size, kernel_size)"

    info_dict = {}
    n = w.shape[0] * w.shape[1]

    with torch.no_grad():
        sparsity_ratio, sparse_mask = sparsity(w, **kwargs)
        ve, ve_norm = variance_entropy(w, **kwargs)
        ve_clean, ve_clean_norm = variance_entropy(w, mask=~sparse_mask, **kwargs)

        info_dict["n"] = n
        info_dict["sparsity"] = sparsity_ratio
        info_dict["variance_entropy"] = ve
        info_dict["variance_entropy_norm"] = ve_norm
        info_dict["variance_entropy_clean"] = ve_clean
        info_dict["variance_entropy_clean_norm"] = ve_clean_norm
        info_dict["orthogonality"] = orthogonality(w, **kwargs)

    return info_dict


def measure_layer_quality(conv_layer: torch.nn.Conv2d, **kwargs) -> dict:
    w = conv_layer.weight.detach()
    info_dict = measure_conv_weight_quality(w, **kwargs)
    return info_dict


def measure_quality(model: torch.nn.Module, **kwargs) -> dict:
    info_dict = {}
    for name, conv_layer in filter(
        lambda x: type(x[1]) == torch.nn.Conv2d and x[1].kernel_size == (3, 3),
        model.named_modules(),
    ):
        info_dict[name] = measure_layer_quality(conv_layer, **kwargs)

    return info_dict
