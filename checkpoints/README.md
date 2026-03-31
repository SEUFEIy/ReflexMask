# Pretrained Model Weights

Download pretrained model weights to this directory before running experiments.

## Download Links

| Model | Dataset | Checkpoint File | Download Link |
|-------|---------|----------------|---------------|
| DAJAT ResNet-18 | CIFAR-10 | `Addepalli2022Efficient_RN18.pt` | [Google Drive](https://drive.google.com/uc?id=1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f) |
| TRADES-AWP WideResNet-34-10 | CIFAR-10 | `TRADES-AWP_cifar10_linf_wrn34-10.pt` | [Google Drive](https://drive.google.com/uc?id=1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T) |
| FAT ResNet-50 | ImageNet | `imagenet_model_weights_4px.pth.tar` | [Google Drive](https://drive.google.com/uc?id=1UrNEtLWs-fjlM2GPb1JpBGtpffDuHH_4) |

## Download Script

```bash
# Install gdown if not already installed
pip install gdown

# Download all checkpoints
gdown 1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f -O checkpoints/
gdown 1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T -O checkpoints/
gdown 1UrNEtLWs-fjlM2GPb1JpBGtpffDuHH_4 -O checkpoints/
```

## Using Other Models

ReflexMask supports any model from [RobustBench model zoo](https://github.com/RobustBench/robustbench).

To use a different model:
1. Download the checkpoint to this directory
2. Add the model definition to `online/models/` (similar to existing implementations)
3. Use the checkpoint filename as the `--load-model` argument

## Verification

After downloading, verify the files:

```bash
ls -lh checkpoints/
# Should show:
#   Addepalli2022Efficient_RN18.pt
#   TRADES-AWP_cifar10_linf_wrn34-10.pt
#   imagenet_model_weights_4px.pth.tar
```
