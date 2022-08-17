import torchvision
from torchconvquality import measure_quality


def test_pretrained():
    model = torchvision.models.resnet18(True)
    quality_dict = measure_quality(model)

    from pprint import pprint
    pprint(quality_dict)

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
