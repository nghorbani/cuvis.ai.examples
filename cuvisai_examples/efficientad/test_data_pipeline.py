"""
Test script for the refactored data pipeline using pytest.
This verifies that the new dataset classes work correctly.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from cuvisai_examples.efficientad.data import CubeDataset, EfficientADCuvisDataset, ImageNetDataset
from cuvisai_examples.efficientad.model import EfficientAdModel


def test_cube_dataset_train_mode():
    """Test CubeDataset in train mode."""
    train_dataset = CubeDataset(
        dataset_dir="data/cubes",
        mode="train",
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
        normalize=True,
        max_data_load=2,  # Limit for testing
    )

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        assert "image" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].dim() == 3  # C×H×W


def test_cube_dataset_test_mode():
    """Test CubeDataset in test mode."""
    test_dataset = CubeDataset(
        dataset_dir="data/cubes",
        mode="test",
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
        normalize=True,
        max_data_load=2,
    )

    if len(test_dataset) > 0:
        sample = test_dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert "mask" in sample
        assert "defect" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["label"], int)
        assert isinstance(sample["mask"], torch.Tensor)


def test_imagenet_dataset_loading():
    """Test ImageNetDataset loading and sampling."""
    dataset = ImageNetDataset(
        imagenet_dir="../data/ImageNet_6_channel", imagenet_file_ending=".npy", mode="train", max_data_load=2
    )

    if len(dataset) > 0:
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.dim() == 3  # C×H×W
        assert sample.shape[0] == 6  # 6 channels

        # Test random sampling
        random_sample = dataset.get_random_sample()
        assert isinstance(random_sample, torch.Tensor)
        assert random_sample.shape[0] == 6


def test_efficientad_cuvis_dataset_train_mode():
    """Test combined dataset in train mode."""
    train_dataset = EfficientADCuvisDataset(
        dataset_dir="data/cubes",
        mode="train",
        imagenet_dir="../data/ImageNet_6_channel",
        imagenet_file_ending=".npy",
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
        normalize=True,
        max_data_load=2,
    )

    if len(train_dataset) > 0:
        sample = train_dataset[0]

        # Check all required keys
        assert "image" in sample, "Missing 'image' key"
        assert "image_ae" in sample, "Missing 'image_ae' key"
        assert "imgnet_img" in sample, "Missing 'imgnet_img' key"

        # Check types
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["image_ae"], torch.Tensor)
        assert isinstance(sample["imgnet_img"], torch.Tensor)

        # Verify that image and image_ae have the same shape
        assert sample["image"].shape == sample["image_ae"].shape


def test_efficientad_cuvis_dataset_test_mode():
    """Test combined dataset in test mode."""
    test_dataset = EfficientADCuvisDataset(
        dataset_dir="data/cubes",
        mode="test",
        imagenet_dir="../data/ImageNet_6_channel",
        imagenet_file_ending=".npy",
        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
        normalize=True,
        max_data_load=2,
    )

    if len(test_dataset) > 0:
        sample = test_dataset[0]

        # Check all required keys
        assert "image" in sample
        assert "label" in sample
        assert "mask" in sample
        assert "defect" in sample

        # Check types
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["label"], int)
        assert isinstance(sample["mask"], torch.Tensor)
        assert isinstance(sample["defect"], str)


def test_model_training_forward():
    """Test model forward pass in training mode."""
    model = EfficientAdModel(teacher_out_channels=384, in_channels=6, model_size="small", use_imgnet_penalty=True)

    # Create dummy inputs
    batch_size = 2
    batch = torch.randn(batch_size, 6, 256, 256)
    batch_ae = torch.randn(batch_size, 6, 256, 256)
    batch_imagenet = torch.randn(batch_size, 6, 256, 256)

    # Test training mode
    model.train()
    loss_st, loss_ae, loss_stae = model(batch, batch_imagenet=batch_imagenet, batch_ae=batch_ae)

    assert isinstance(loss_st, torch.Tensor)
    assert isinstance(loss_ae, torch.Tensor)
    assert isinstance(loss_stae, torch.Tensor)
    assert loss_st.requires_grad
    assert loss_ae.requires_grad
    assert loss_stae.requires_grad


def test_model_eval_forward():
    """Test model forward pass in eval mode."""
    model = EfficientAdModel(teacher_out_channels=384, in_channels=6, model_size="small", use_imgnet_penalty=True)

    # Create dummy inputs
    batch_size = 2
    batch = torch.randn(batch_size, 6, 256, 256)

    # Test eval mode
    model.eval()
    with torch.no_grad():
        output = model(batch, return_all_maps=True)

    assert isinstance(output, dict)
    assert "anomaly_map" in output
    assert "map_st" in output
    assert "map_ae" in output
    assert output["anomaly_map"].shape == (batch_size, 1, 256, 256)


# Data tests run by default; keep --run-data-tests option for compatibility.
def pytest_addoption(parser):
    """Add custom pytest options (kept for compatibility)."""
    parser.addoption(
        "--run-data-tests", action="store_true", default=False, help="Run tests that require actual data files"
    )
