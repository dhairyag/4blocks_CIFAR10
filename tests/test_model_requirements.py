import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import CIFAR10Net
import pytest
from typing import List
import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def has_depthwise_separable_conv(model: nn.Module) -> bool:
    """Check if model contains depthwise separable convolution."""
    for module in model.modules():
        # Check for custom DepthwiseSeparableConv class
        if module.__class__.__name__ == 'DepthwiseSeparableConv':
            return True
        # Also check for conventional implementation
        if isinstance(module, nn.Sequential):
            layers = list(module.children())
            for i in range(len(layers) - 1):
                current = layers[i]
                next_layer = layers[i + 1]
                if (isinstance(current, nn.Conv2d) and isinstance(next_layer, nn.Conv2d) and
                    current.groups == current.in_channels and next_layer.kernel_size == (1, 1)):
                    return True
    return False

def has_dilated_conv(model: nn.Module) -> bool:
    """Check if model contains dilated convolution."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.dilation != (1, 1):
            return True
    return False

def has_gap(model: nn.Module) -> bool:
    """Check if model contains Global Average Pooling."""
    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d) and name == 'gap':
            return True
    return False

def has_fc_after_gap(model: nn.Module) -> bool:
    """Check if model contains FC layer after GAP."""
    found_gap = False
    for module in model.modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            found_gap = True
        elif found_gap and isinstance(module, nn.Linear):
            return True
    return False

def test_parameter_count():
    model = CIFAR10Net()
    param_count = count_parameters(model)
    assert param_count < 200000, f"Model has {param_count} parameters, which exceeds the limit of 200,000"

def test_depthwise_separable_conv():
    model = CIFAR10Net()
    assert has_depthwise_separable_conv(model), "Model does not use Depthwise Separable Convolution"

def test_dilated_conv():
    model = CIFAR10Net()
    assert has_dilated_conv(model), "Model does not use Dilated Convolution"

def test_gap():
    model = CIFAR10Net()
    assert has_gap(model), "Model does not use Global Average Pooling"

def test_fc_after_gap():
    model = CIFAR10Net()
    assert has_fc_after_gap(model), "Model does not have FC layer after GAP" 