"""
Build Conscious Prototypes (Offline)
Computes class-conditional consciousness state prototypes from calibration set
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from online.models.resnet_val import ResNet18
from online.models.wideresnet_trades import WideResNet34_10
from online.models.resnet50 import InterpMaskedResNet50
from online.defense import (
    ConsciousConfig,
    ConsciousStateExtractor,
    PrototypeManager
)
from collections import OrderedDict

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def get_model(arch: str, num_classes: int, checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint"""
    # Create model
    if arch == "rn18_val":
        model = ResNet18(num_classes=num_classes)
    elif arch == "wrn34_10":
        model = WideResNet34_10(num_classes=num_classes)
    elif arch == "preactresnet18":
        from online.models.preactresnet import PreActResNet18
        model = PreActResNet18(num_classes=num_classes)
    elif arch == "rn50":
        # For ResNet50, we need to check the actual implementation
        # Using a dummy InterpMaskedResNet50 for now
        raise NotImplementedError("ResNet50 support pending - use rn18_val or wrn34_10")
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel wrapper
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model


def get_dataset(dataset_name: str, data_dir: str = "./data", train: bool = True):
    """Get dataset with appropriate transforms"""
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == "imagenet":
        # ImageNet needs different transforms
        if train:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train" if train else "val"), transform=transform)
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset, num_classes


def extract_layer_activation(model: nn.Module, images: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Extract activation from specified layer"""
    activation = None
    
    def hook_fn(module, input, output):
        nonlocal activation
        activation = output
    
    # Register hook
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(images)
    
    # Remove hook
    handle.remove()
    
    return activation


def build_prototypes(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    extractor: ConsciousStateExtractor,
    proto_manager: PrototypeManager,
    layer_name: str,
    batch_size: int = 128,
    max_samples: int = -1,
    use_pseudo_labels: bool = False,
    device: str = "cuda"
):
    """
    Build prototypes from dataset
    
    Args:
        model: Neural network model
        dataset: Dataset to use
        extractor: Consciousness state extractor
        proto_manager: Prototype manager
        layer_name: Layer to extract from
        batch_size: Batch size
        max_samples: Maximum number of samples (-1 for all)
        use_pseudo_labels: If True, use model predictions as labels
        device: Device
    """
    # Create dataloader
    if max_samples > 0:
        indices = list(range(min(max_samples, len(dataset))))
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Building prototypes from {len(dataset)} samples...")
    
    total_samples = 0
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Get pseudo labels if needed
        if use_pseudo_labels:
            with torch.no_grad():
                logits = model(images)
                labels = torch.argmax(logits, dim=1)
        
        # Extract activation
        activation = extract_layer_activation(model, images, layer_name)
        
        # Extract consciousness states
        c = extractor.extract(activation)
        
        # Update prototypes
        proto_manager.update(c, labels)
        
        total_samples += images.shape[0]
        
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = total_samples / elapsed
            print(f"Processed {total_samples}/{len(dataset)} samples ({speed:.1f} samples/s)")
    
    print(f"Finished processing {total_samples} samples in {time.time() - start_time:.1f}s")
    
    # Prototypes are automatically finalized during update()
    print(f"Prototypes computed for {proto_manager.num_classes} classes")
    
    return proto_manager


def main():
    parser = argparse.ArgumentParser(description="Build Conscious Prototypes")
    
    # Model arguments
    parser.add_argument("--arch", type=str, required=True, choices=["rn18_val", "wrn34_10", "preactresnet18", "rn50"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "imagenet"],
                        help="Dataset name")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: PROJECT_ROOT/data)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory containing checkpoints (default: PROJECT_ROOT/checkpoints)")
    
    # Extraction arguments
    parser.add_argument("--layer-names", type=str, nargs="+", default=["layer4"],
                        help="Layer names to extract from")
    parser.add_argument("--layer-dim", type=int, default=512,
                        help="Layer dimension")
    
    # Conscious state arguments
    parser.add_argument("--conscious-dim", type=int, default=16,
                        help="Consciousness state dimension")
    parser.add_argument("--conscious-top-s", type=int, default=8,
                        help="Number of top activations")
    parser.add_argument("--conscious-seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--projection-type", type=str, default="random_ortho",
                        choices=["random_ortho", "hadamard"],
                        help="Projection type")
    
    # Dataset arguments
    parser.add_argument("--use-train", action="store_true",
                        help="Use training set")
    parser.add_argument("--use-pseudo-labels", action="store_true",
                        help="Use model predictions as labels")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Maximum number of samples (-1 for all)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: saved_conscious_prototypes/{model_name})")
    
    args = parser.parse_args()

    # Set default paths relative to project root
    if args.data_dir is None:
        args.data_dir = str(_PROJECT_ROOT / "data")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(_PROJECT_ROOT / "checkpoints")
    if args.output_dir is None:
        args.output_dir = str(_PROJECT_ROOT / "saved_prototypes")

    # Resolve checkpoint path
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(args.checkpoint_dir, os.path.basename(args.checkpoint))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset
    dataset, num_classes = get_dataset(args.dataset, args.data_dir, train=args.use_train)
    print(f"Dataset: {args.dataset}, num_classes: {num_classes}, samples: {len(dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = get_model(args.arch, num_classes, args.checkpoint, device)
    print("Model loaded successfully")
    
    # Create config
    config = ConsciousConfig()
    config.conscious_dim = args.conscious_dim
    config.conscious_top_s = args.conscious_top_s
    config.conscious_seed = args.conscious_seed
    config.projection_type = args.projection_type
    
    # Get model name from checkpoint
    model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    
    # Process each layer
    for layer_name in args.layer_names:
        print(f"\n{'='*60}")
        print(f"Processing layer: {layer_name}")
        print(f"{'='*60}")
        
        # Create extractor
        extractor = ConsciousStateExtractor(args.layer_dim, config, device)
        
        # Create prototype manager
        proto_manager = PrototypeManager(num_classes, args.conscious_dim, config, device)
        
        # Build prototypes
        proto_manager = build_prototypes(
            model, dataset, extractor, proto_manager, layer_name,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            use_pseudo_labels=args.use_pseudo_labels,
            device=device
        )
        
        # Save prototypes
        if args.output_dir is None:
            save_dir = "saved_conscious_prototypes"
        else:
            save_dir = args.output_dir
        
        proto_manager.save(save_dir, args.arch, args.dataset, layer_name)
        proto_path = os.path.join(save_dir, args.arch, args.dataset, layer_name, "prototypes.npz")
        print(f"Saved prototypes to {proto_path}")
        
        # Save extractor projection matrix
        output_dir = os.path.join(save_dir, args.arch, args.dataset, layer_name)
        proj_path = os.path.join(output_dir, "projection.npz")
        extractor.save_projection(proj_path)
        print(f"Saved projection matrix to {proj_path}")
        
        # Save metadata
        metadata = {
            'arch': args.arch,
            'dataset': args.dataset,
            'layer_name': layer_name,
            'layer_dim': args.layer_dim,
            'conscious_dim': args.conscious_dim,
            'conscious_top_s': args.conscious_top_s,
            'conscious_seed': args.conscious_seed,
            'projection_type': args.projection_type,
            'num_classes': num_classes,
            'checkpoint': args.checkpoint
        }
        
        metadata_path = os.path.join(output_dir, "metadata.npz")
        np.savez(metadata_path, **metadata)
        print(f"Saved metadata to {metadata_path}")
    
    print(f"\n{'='*60}")
    print("All layers processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

