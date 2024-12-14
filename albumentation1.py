import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision import datasets
import cv2

# CIFAR10 mean and std values
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

# Training augmentation pipeline
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT,
        value=0
    ),
    A.CoarseDropout(
        max_holes=1,
        max_height=16,
        max_width=16,
        min_holes=1,
        min_height=16,
        min_width=16,
        p=0.5
    ),
    A.Normalize(
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
    ),
    ToTensorV2(),
])

# Test transforms (only normalization)
test_transforms = A.Compose([
    A.Normalize(
        mean=CIFAR_MEAN,
        std=CIFAR_STD,
    ),
    ToTensorV2(),
])

class CIFAR10Albumentations(datasets.CIFAR10):
    def __init__(self, root, train=True, download=True, transform=None):
        super().__init__(root, train=train, download=download, transform=None)
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]
            
        return img, label
