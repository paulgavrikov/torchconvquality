import torchvision.models
from torchconvquality import measure_quality, _torch_entropy10, sparsity, variance_entropy, orthogonality
import scipy.stats
import torch


ERROR = 1e-5


def test_sparsity():
    w = torch.zeros(64, 16, 3, 3)
    assert sparsity(w)[0] == 1
    w = torch.ones(64, 16, 3, 3)
    assert sparsity(w)[0] == 0
    w = torch.ones(2, 1, 3, 3)
    w[0] = w[0] * 0
    assert sparsity(w)[0] == 0.5


def test_variance_entropy():
    w = torch.rand(128, 128, 3, 3)
    ve, ve_norm = variance_entropy(w) 
    assert ve_norm > 1
    w = torch.ones(64, 16, 3, 3)
    assert variance_entropy(w) == (0, 0)


def test_orthogonality():
    w = torch.ones(64, 16, 3, 3)
    w_b = torch.ones(64, 16, 3, 3) * 0.1
    assert orthogonality(w) < ERROR
    assert orthogonality(w_b) < ERROR
    assert abs(orthogonality(w) - orthogonality(w_b)) < ERROR
    w = torch.zeros(64, 16, 3, 3)
    assert orthogonality(w) < ERROR


def test_torch_entropy10():
    p = torch.tensor([0.5, 0.2, 0.2, 0.1])
    assert abs(_torch_entropy10(p) - scipy.stats.entropy(p.detach().numpy(), base=10)) < ERROR
    assert _torch_entropy10(p) != scipy.stats.entropy(p.detach().numpy(), base=2)


def test_pretrained():
    model = torchvision.models.resnext50_32x4d(True)
    if torch.cuda.is_available():
        model.cuda()

    quality_dict = measure_quality(model, sparsity_eps=0.1)
    assert quality_dict is not None

    for layer_name, entry in quality_dict.items():
        assert "sparsity" in entry
        assert "orthogonality" in entry
        assert "variance_entropy" in entry
        assert "variance_entropy_norm" in entry
        assert "variance_entropy_clean" in entry
        assert "variance_entropy_clean_norm" in entry
        assert "n" in entry
        assert entry["sparsity"] <= 1
        assert entry["sparsity"] >= 0
        assert entry["variance_entropy"] >= 0
        assert entry["variance_entropy"] <= 1
        assert entry["variance_entropy_norm"] >= 0
        assert entry["variance_entropy_norm"] <= 1
        assert entry["variance_entropy_clean"] >= 0
        assert entry["variance_entropy_clean"] <= 1
        assert entry["variance_entropy_clean_norm"] >= 0
        assert entry["variance_entropy_clean_norm"] <= 1

        if entry["sparsity"] == 0:
            assert entry["variance_entropy"] == entry["variance_entropy_clean"]


def test_untrained():
    model = torchvision.models.resnext50_32x4d(False)
    if torch.cuda.is_available():
        model.cuda()

    quality_dict = measure_quality(model)
    assert quality_dict is not None

    for layer_name, entry in quality_dict.items():
        assert "sparsity" in entry
        assert "orthogonality" in entry
        assert "variance_entropy" in entry
        assert "variance_entropy_norm" in entry
        assert "variance_entropy_clean" in entry
        assert "variance_entropy_clean_norm" in entry
        assert "n" in entry
        assert entry["sparsity"] >= 0

        # conservative thresholds
        assert entry["sparsity"] <= 0.1
        assert entry["variance_entropy"] >= 0.9
        assert entry["variance_entropy_clean"] >= 0.9
        assert entry["variance_entropy_norm"] >= 0.9
        assert entry["variance_entropy_clean_norm"] >= 0.9


def test_untouched_original():
    model = torchvision.models.resnext50_32x4d(False)
    state_pre = model.state_dict().copy()
    _ = measure_quality(model)
    state_post = model.state_dict()

    for k_pre, k_post in zip(state_pre.keys(), state_post.keys()):
        assert torch.equal(state_pre[k_pre], state_post[k_post])


def test_output_no_tensors():
    model = torchvision.models.resnext50_32x4d(False)
    if torch.cuda.is_available():
        model.cuda()

    quality_dict = measure_quality(model)
    assert quality_dict is not None

    for layer_name, entry in quality_dict.items():
        for key, value in entry.items():
            assert type(value) != torch.Tensor
