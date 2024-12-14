# CIFAR10 Image Classification with Custom CNN

[![Model Architecture Checks](https://github.com/dhairyag/4blocks_CIFAR10/actions/workflows/model-checks.yml/badge.svg)](https://github.com/dhairyag/4blocks_CIFAR10/actions/workflows/model-checks.yml)

This project implements a custom Convolutional Neural Network (CNN) architecture for the CIFAR10 dataset classification task. The network achieves 85%+ accuracy while maintaining under 128k parameters through efficient architecture choices and modern convolution techniques.

## Project Structure

```bash
├── .github/
│ └── workflows/ # CI/CD workflows
├── tests/ # Unit tests
├── main.py # Entry point
├── model.py # Model architecture
├── train.py # Training logic
├── utils.py # Utility functions
└── albumentation1.py # Data augmentation
```


## Key Features

- Custom CNN architecture optimized for CIFAR10
- Efficient parameter usage  (< 128k params)
- Modern convolution techniques:
  - Depthwise Separable Convolution
  - Dilated Convolution
  - Global Average Pooling (GAP)
- Data augmentation using Albumentations:
  - Horizontal Flip
  - ShiftScaleRotate
  - CoarseDropout

## Model Architecture
```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CIFAR10Net                               [1, 10]                   --
├─Sequential: 1-1                        [1, 16, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 8, 32, 32]            224
│    └─BatchNorm2d: 2-2                  [1, 8, 32, 32]            16
│    └─ReLU: 2-3                         [1, 8, 32, 32]            --
│    └─Conv2d: 2-4                       [1, 16, 32, 32]           1,168
│    └─BatchNorm2d: 2-5                  [1, 16, 32, 32]           32
│    └─ReLU: 2-6                         [1, 16, 32, 32]           --
│    └─Conv2d: 2-7                       [1, 16, 32, 32]           2,320
│    └─BatchNorm2d: 2-8                  [1, 16, 32, 32]           32
│    └─ReLU: 2-9                         [1, 16, 32, 32]           --
├─Sequential: 1-2                        [1, 32, 16, 16]           --
│    └─DepthwiseSeparableConv: 2-10      [1, 32, 16, 16]           --
│    │    └─Conv2d: 3-1                  [1, 16, 16, 16]           160
│    │    └─Conv2d: 3-2                  [1, 32, 16, 16]           544
│    └─BatchNorm2d: 2-11                 [1, 32, 16, 16]           64
│    └─ReLU: 2-12                        [1, 32, 16, 16]           --
│    └─Conv2d: 2-13                      [1, 32, 16, 16]           9,248
│    └─BatchNorm2d: 2-14                 [1, 32, 16, 16]           64
│    └─ReLU: 2-15                        [1, 32, 16, 16]           --
│    └─Conv2d: 2-16                      [1, 32, 16, 16]           9,248
│    └─BatchNorm2d: 2-17                 [1, 32, 16, 16]           64
│    └─ReLU: 2-18                        [1, 32, 16, 16]           --
├─Sequential: 1-3                        [1, 48, 16, 16]           --
│    └─Conv2d: 2-19                      [1, 48, 16, 16]           13,872
│    └─BatchNorm2d: 2-20                 [1, 48, 16, 16]           96
│    └─ReLU: 2-21                        [1, 48, 16, 16]           --
│    └─Conv2d: 2-22                      [1, 48, 16, 16]           20,784
│    └─BatchNorm2d: 2-23                 [1, 48, 16, 16]           96
│    └─ReLU: 2-24                        [1, 48, 16, 16]           --
├─Sequential: 1-4                        [1, 64, 8, 8]             --
│    └─Conv2d: 2-25                      [1, 64, 8, 8]             27,712
│    └─BatchNorm2d: 2-26                 [1, 64, 8, 8]             128
│    └─ReLU: 2-27                        [1, 64, 8, 8]             --
│    └─Conv2d: 2-28                      [1, 64, 8, 8]             36,928
│    └─BatchNorm2d: 2-29                 [1, 64, 8, 8]             128
│    └─ReLU: 2-30                        [1, 64, 8, 8]             --
│    └─Conv2d: 2-31                      [1, 64, 8, 8]             4,160
├─AdaptiveAvgPool2d: 1-5                 [1, 64, 1, 1]             --
├─Linear: 1-6                            [1, 10]                   650
==========================================================================================
Total params: 127,738
Trainable params: 127,738
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 21.99
==========================================================================================
```

### Key Architectural Decisions

- **No MaxPooling**: Uses strided convolutions for downsampling
- **Receptive Field**: 45 pixels
- **Layer Structure**: C1-C2-C3-C4-O format
- **Special Layers**:
  - Depthwise Separable Convolution in one layer
  - Dilated Convolution in one layer
  - Global Average Pooling (GAP)
  - Final FC layer for classification

## Requirements
```bash
python>=3.8
torch>=1.7.0
torchvision>=0.8.0
albumentations>=1.3.0
numpy>=1.19.2
```


## Installation
```bash
git clone git@github.com:dhairyag/4blocks_CIFAR10.git
cd 4blocks_CIFAR10
pip install -r requirements.txt
```


## Usage

### Training
```bash
python main.py
```


## Data Augmentation

The project uses Albumentations library for data augmentation with the following transformations:
```python
transforms = A.Compose([
A.HorizontalFlip(p=0.5),
A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1,
min_height=16, min_width=16, fill_value=0.473363, p=0.5),
])
```


## Performance

- **Accuracy**: 85%+
- **Total Parameters**: < 200k
- **Receptive Field**: > 44 pixels

## Testing

Run the test suite:
```bash
python -m pytest tests/
```


## CI/CD

The project uses GitHub Actions for CI/CD. The workflow is defined in .github/workflows/model-checks.yml.

