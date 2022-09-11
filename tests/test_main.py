import torchvision.models
from torchconvquality import measure_quality, _torch_entropy10
import scipy.stats
import torch


def test_torch_entropy10():
    p = torch.tensor([0.5, 0.2, 0.2, 0.1])
    assert _torch_entropy10(p) == scipy.stats.entropy(p.detach().numpy(), base=10)
    assert _torch_entropy10(p) != scipy.stats.entropy(p.detach().numpy(), base=2)


def test_pretrained():
    model = torchvision.models.resnet18(True)
    if torch.cuda.is_available():
        model.cuda()

    quality_dict = measure_quality(model, sparsity_eps=0.1)
    assert quality_dict is not None

    for layer_name, entry in quality_dict.items():
        assert "sparsity" in entry
        assert "variance_entropy" in entry
        assert "variance_entropy_clean" in entry
        assert "n" in entry
        assert entry["sparsity"] <= 1
        assert entry["sparsity"] >= 0
        assert entry["variance_entropy"] >= 0
        assert entry["variance_entropy"] <= 1
        assert entry["variance_entropy_clean"] >= 0
        assert entry["variance_entropy_clean"] <= 1

        if entry["sparsity"] == 0:
            assert entry["variance_entropy"] == entry["variance_entropy_clean"]


def test_untrained():
    model = torchvision.models.resnet18(False)
    if torch.cuda.is_available():
        model.cuda()
    
    quality_dict = measure_quality(model)
    assert quality_dict is not None

    for layer_name, entry in quality_dict.items():
        assert "sparsity" in entry
        assert "variance_entropy" in entry
        assert "variance_entropy_clean" in entry
        assert "n" in entry
        assert entry["sparsity"] >= 0

        # conservative thresholds
        assert entry["sparsity"] <= 0.1
        assert entry["variance_entropy"] >= 0.9
        assert entry["variance_entropy_clean"] >= 0.9
