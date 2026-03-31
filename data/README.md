# Datasets

This directory is for datasets. It is recommended to organize datasets as follows:

```
data/
├── cifar10/           # CIFAR-10 (auto-downloaded on first use)
├── cifar100/          # CIFAR-100 (auto-downloaded on first use)
└── imagenet/
    ├── train/         # ImageNet training set
    └── val/           # ImageNet validation set
```

## CIFAR-10 and CIFAR-100

Automatically downloaded by `torchvision.datasets` when first used. No manual setup required.

```python
from torchvision import datasets
datasets.CIFAR10(root='./data', train=False, download=True)
```

## ImageNet

ImageNet requires manual download due to licensing restrictions.

### Download from ImageNet Website

1. Register at [ImageNet](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
2. Download ILSVRC2012 training and validation sets
3. Extract to `data/imagenet/`

### Expected Directory Structure

```
data/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── image1.JPEG
│   │   └── ...
│   └── ... (1000 classes)
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000001.JPEG
    │   └── ...
    └── ... (1000 classes)
```

### Kaggle API

```bash
pip install kaggle
# Place ~/.kaggle/kaggle.json
kaggle competitions download -c imagenet-object-localization-challenge
```

## Quick Verification

```bash
python -c "from torchvision import datasets; print(datasets.CIFAR10(root='./data', train=False, download=True)); print('CIFAR-10 OK')"
```
