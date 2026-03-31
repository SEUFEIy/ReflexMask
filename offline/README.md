# Offline Preprocessing Stage

This stage consists of one-time preprocessing steps that must be completed before running online defense or evaluation.

## Three Steps

### Step 1: Neuron Importance Rankings (`rankings/`)

Identify which neurons are most important for each class using two methods:

- **LOIR (Leave-One-Out Importance Ranking)**: Measures logit change when each neuron is ablated. Required for IG-Defense.
- **CDIR (CLIP-Dissect Importance Ranking)**: Uses CLIP text-image similarity to semantically describe neurons. Optional but provides additional ranking.

**Time**: ~10-30 minutes for CIFAR models, several hours for ImageNet.

### Step 2: Consciousness Prototypes (`prototypes/`)

Build class-conditional consciousness state prototypes from calibration data. Required only for Conscious IG-Defense (the enhanced iterative method).

**Time**: ~5-15 minutes.

### Step 3: Neuron Analysis (`analysis/`)

Analyze how adversarial perturbations affect neuron activations compared to clean neurons. Used for paper figure generation.

## Recommended Order

```
1. rankings/        (Required for IG-Defense)
       ↓
2. prototypes/     (Required for Conscious IG-Defense only)
       ↓
3. evaluation      (Run eval/ scripts)
```

## Output Directories

Results are saved to:

```
saved_rankings/              # LOIR/CDIR rankings
saved_prototypes/            # Consciousness prototypes
```

These directories are gitignored and can be regenerated with the scripts in each subdirectory.
