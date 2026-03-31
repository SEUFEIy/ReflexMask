# ReflexMask

**ReflexMask: Introspective Test-Time Rewiring for Adversarial Robustness**

ReflexMask is a training-free test-time adversarial defense framework for improving adversarial robustness at inference time, without model fine-tuning.  
The method follows an offline-to-online pipeline: offline, it constructs neuron-importance artifacts and consciousness prototypes; online, it performs introspection-driven adaptive masking during inference.

---

## Setup

### 1. Create environment

```bash
conda create -n ReflexMask python=3.9 -y
conda activate ReflexMask
pip install -r requirements.txt
```

### 2. Download pretrained checkpoints

Please download the pretrained weights into `checkpoints/`.

| Model | Dataset | Checkpoint File | Download |
|-------|---------|----------------|----------|
| DAJAT ResNet-18 | CIFAR-10 | `Addepalli2022Efficient_RN18.pt` | [Google Drive](https://drive.google.com/uc?id=1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f) |
| TRADES-AWP WideResNet-34-10 | CIFAR-10 | `TRADES-AWP_cifar10_linf_wrn34-10.pt` | [Google Drive](https://drive.google.com/uc?id=1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T) |
| FAT ResNet-50 | ImageNet | `imagenet_model_weights_4px.pth.tar` | [Google Drive](https://drive.google.com/uc?id=1UrNEtLWs-fjlM2GPb1JpBGtpffDuHH_4) |

Quick download with `gdown`:

```bash
mkdir -p checkpoints

gdown 1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f -O checkpoints/Addepalli2022Efficient_RN18.pt
gdown 1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T -O checkpoints/TRADES-AWP_cifar10_linf_wrn34-10.pt
gdown 1UrNEtLWs-fjlM2GPb1JpBGtpffDuHH_4 -O checkpoints/imagenet_model_weights_4px.pth.tar
```

### 3. Prepare datasets

- **CIFAR-10 / CIFAR-100** are downloaded automatically by `torchvision`.
- **ImageNet** should be prepared manually if you want to run ImageNet experiments.

---

## Quick Start

### Step 1: Generate neuron importance rankings
Offline, one-time per model.

```bash
bash offline/rankings/scripts/run_loir_rn18_cifar10.sh
bash offline/rankings/scripts/run_loir_wrn34_cifar10.sh
```

### Step 2: Build consciousness prototypes
Offline, one-time, for the consciousness-guided defense pipeline.

```bash
bash offline/prototypes/scripts/build_proto_rn18_cifar10.sh
```

### Step 3: Evaluate

```bash
bash eval/scripts/eval_clean.sh
bash eval/scripts/eval_autoattack.sh
bash eval/scripts/eval_iwwc.sh
```

---

## Method Overview

ReflexMask follows an **offline-to-online** workflow.

### Offline stage

The offline stage prepares the static artifacts required at inference time:

- neuron importance rankings such as **LOIR / CDIR**
- masking resources derived from the rankings
- consciousness prototypes for online monitoring

### Online stage

At test time, ReflexMask performs an introspection-driven closed-loop defense:

1. extract a low-dimensional consciousness state from neuron activations,
2. compute a risk score based on prototype distance, entropy, and prediction instability,
3. apply adaptive masking according to the estimated risk,
4. refine the prediction through iterative introspection and intervention when needed.

---

## Project Structure

```text
ReflexMask/
├── offline/
│   ├── rankings/           # LOIR / CDIR neuron importance rankings (offline)
│   └── prototypes/         # Consciousness prototypes
├── online/
│   └── models/             # Model definitions with masking support
├── eval/                   # Evaluation scripts
│   └── scripts/            # Quick-run shell scripts
├── checkpoints/            # Pretrained model weights
├── data/                   # Datasets
├── saved_rankings/         # Generated rankings
└── saved_prototypes/       # Generated prototypes
```

---

## Supported Models and Datasets

| Model | Dataset |
|-------|---------|
| ResNet-18 | CIFAR-10, CIFAR-100 |
| WideResNet-34-10 | CIFAR-10, CIFAR-100 |
| ResNet-50 | ImageNet |

---

## Outputs and Artifacts

Running the pipeline will generate the following artifacts:

1. **Checkpoint files**  
   Stored in `checkpoints/`

2. **Ranking artifacts**  
   Stored in `saved_rankings/`

3. **Prototype artifacts**  
   Stored in `saved_prototypes/`
