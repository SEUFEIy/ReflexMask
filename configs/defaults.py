# Default Configurations

Central configuration file for ReflexMask experiments.
This module provides default settings for models, defenses, and evaluation.

## Model Configurations

Each entry specifies the parameters needed to run an experiment with a specific model/dataset combination.

```python
from configs.defaults import MODEL_CONFIGS, IG_DEFENSE_DEFAULTS, EVAL_DEFAULTS

config = MODEL_CONFIGS["rn18_cifar10"]
```

## Usage

```python
# Get model configuration
model_cfg = MODEL_CONFIGS["rn18_cifar10"]
# {'arch': 'rn18_val', 'checkpoint': 'Addepalli2022Efficient_RN18.pt', ...}

# Get defense configuration
defense_cfg = IG_DEFENSE_DEFAULTS
# {'mask_which': 'loir', 'important_dim': 50, 'rs': True, ...}

# Get evaluation configuration
eval_cfg = EVAL_DEFAULTS["cifar10"]
# {'epsilon': 8, 'n_ex': 10000, 'batch_size': 128}
```
"""

# =============================================================================
# Model Configurations
# =============================================================================
# Each config specifies: architecture type, checkpoint filename, layer to mask,
# layer dimensionality, and number of classes.

MODEL_CONFIGS = {
    # --- CIFAR-10 ---
    "rn18_cifar10": {
        "arch": "rn18",
        "arch_internal": "rn18_val",
        "checkpoint": "Addepalli2022Efficient_RN18.pt",
        "layer_name": "layer4",
        "layer_dim": 512,
        "num_classes": 10,
    },
    "wrn34_cifar10": {
        "arch": "wrn34_10",
        "arch_internal": "wrn34_10",
        "checkpoint": "TRADES-AWP_cifar10_linf_wrn34-10.pt",
        "layer_name": "block3",
        "layer_dim": 640,
        "num_classes": 10,
    },
    # --- CIFAR-100 ---
    "rn18_cifar100": {
        "arch": "rn18",
        "arch_internal": "rn18_val",
        "checkpoint": "Addepalli2022Efficient_RN18.pt",
        "layer_name": "layer4",
        "layer_dim": 512,
        "num_classes": 100,
    },
    "wrn34_cifar100": {
        "arch": "wrn34_10",
        "arch_internal": "wrn34_10",
        "checkpoint": "TRADES-AWP_cifar10_linf_wrn34-10.pt",
        "layer_name": "block3",
        "layer_dim": 640,
        "num_classes": 100,
    },
    # --- ImageNet ---
    "rn50_imagenet": {
        "arch": "rn50",
        "arch_internal": "rn50",
        "checkpoint": "imagenet_model_weights_4px.pth.tar",
        "layer_name": "layer4",
        "layer_dim": 2048,
        "num_classes": 1000,
    },
}

# =============================================================================
# IG-Defense Configurations
# =============================================================================
# Default parameters for the basic IG-Defense masking method.

IG_DEFENSE_DEFAULTS = {
    # Importance ranking method: 'loir' or 'cdir'
    "mask_which": "loir",
    # Number of neurons to retain (k value)
    "important_dim": 50,
    # Randomized smoothing
    "rs": True,
    "rs_sigma": 4,      # Noise sigma (will be divided by 255)
    "rs_nsmooth": 1,    # Number of smoothing samples
}

# =============================================================================
# Conscious IG-Defense Configurations
# =============================================================================
# Default parameters for the enhanced Conscious IG-Defense method.

CONSCIOUS_IG_DEFAULTS = {
    # Consciousness state extraction
    "conscious_dim": 16,      # Dimensionality of consciousness state
    "conscious_top_s": 8,     # Number of top activations to keep
    "conscious_seed": 42,     # Random seed for reproducibility
    # Risk monitoring
    "risk_threshold": 0.8,    # Threshold to trigger defense
    "risk_alpha": 0.3,        # Entropy weight in risk score
    "risk_beta": 0.2,         # Instability weight in risk score
    # Masking strategy
    "k_list": [16, 32, 64, 128],  # Granularities for adaptive masking
    "top_m": 3,              # Number of top classes for hypothesis
    # Iterative refinement
    "max_steps": 3,           # Maximum iterative steps
    # Randomized smoothing on high risk
    "rs_sigma": 8.0 / 255.0,
    "rs_nsmooth": 16,
}

# =============================================================================
# Evaluation Configurations
# =============================================================================
# Default parameters for evaluation per dataset.

EVAL_DEFAULTS = {
    "cifar10": {
        "epsilon": 8,
        "n_ex": 10000,
        "batch_size": 128,
        "threat_model": "Linf",
    },
    "cifar100": {
        "epsilon": 8,
        "n_ex": 10000,
        "batch_size": 128,
        "threat_model": "Linf",
    },
    "imagenet": {
        "epsilon": 4,
        "n_ex": 5000,
        "batch_size": 64,
        "threat_model": "Linf",
        "preprocessing": "Crop288",
    },
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_model_config(name: str) -> dict:
    """Get model configuration by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model config: {name}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[name]


def get_eval_config(dataset: str) -> dict:
    """Get evaluation configuration by dataset."""
    if dataset not in EVAL_DEFAULTS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {list(EVAL_DEFAULTS.keys())}"
        )
    return EVAL_DEFAULTS[dataset]


def build_ranking_path(model_name: str, layer_name: str, method: str = "loir") -> str:
    """Build the expected path for saved importance rankings."""
    return f"saved_rankings/{model_name}/{layer_name}"


def build_prototype_path(model_name: str, dataset: str, layer_name: str) -> str:
    """Build the expected path for saved consciousness prototypes."""
    return f"saved_prototypes/{model_name}/{dataset}/{layer_name}/prototypes.npz"
