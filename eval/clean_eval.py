"""
Evaluate Defended Model Clean Accuracy
"""
import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import time

from online.models.resnet_val import InterpMaskedResNet
from online.models.resnet50 import InterpMaskedResNet50
from online.models.wideresnet_trades import InterpMaskedWideResNet
from online.defense import create_conscious_defense, ConsciousConfig

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def get_args():
    parser = argparse.ArgumentParser(description='Defended Model Clean Accuracy Evaluation')

    parser.add_argument('--arch', type=str, required=True,
                        choices=['rn18', 'rn18_val', 'rn50', 'wrn34_10'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset name')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to dataset directory')

    parser.add_argument('--proto-path', type=str, required=True,
                        help='Path to prototype file (.npz)')
    parser.add_argument('--layer-name', type=str, default='layer4',
                        choices=['layer1', 'layer2', 'layer3', 'layer4', 'block1', 'block2', 'block3'],
                        help='Layer name for consciousness extraction')
    parser.add_argument('--layer-dim', type=int, required=True,
                        help='Layer dimension (e.g., 512 for layer4, 640 for block3)')
    parser.add_argument('--ranking-method', type=str, default='loir',
                        choices=['loir', 'cdir'],
                        help='Neuron importance ranking method')

    parser.add_argument('--conscious-dim', type=int, default=16,
                        help='Consciousness state dimension')
    parser.add_argument('--k-list', type=str, default='16,32,64,128',
                        help='Comma-separated list of k values for masking')
    parser.add_argument('--max-steps', type=int, default=3,
                        help='Maximum iterative defense steps')
    parser.add_argument('--risk-threshold', type=float, default=0.8,
                        help='Risk threshold for defense activation')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to dataset (default: PROJECT_ROOT/data)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing checkpoints (default: PROJECT_ROOT/checkpoints)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: PROJECT_ROOT/results)')

    return parser.parse_args()


def load_base_model(args):
    print(f"\nLoading base model: {args.arch} on {args.dataset}")

    num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000
    }[args.dataset]

    if args.arch in ['rn18', 'rn18_val']:
        model = InterpMaskedResNet(
            layer_name=args.layer_name,
            checkpoint_name=args.checkpoint,
            mask_which=args.ranking_method,
            important_dim=128,
            num_classes=num_classes,
            rs=False
        )
    elif args.arch == 'rn50':
        model = InterpMaskedResNet50(
            layer_name=args.layer_name,
            checkpoint_name=args.checkpoint,
            mask_which=args.ranking_method,
            important_dim=128,
            num_classes=num_classes,
            rs=False
        )
    elif args.arch == 'wrn34_10':
        model = InterpMaskedWideResNet(
            layer_name=args.layer_name,
            checkpoint_name=args.checkpoint,
            mask_which=args.ranking_method,
            important_dim=128,
            num_classes=num_classes,
            rs=False
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    checkpoint_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(args.checkpoint_dir, os.path.basename(args.checkpoint))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k

        if args.arch == 'wrn34_10':
            if '.conv1.bias' in name or '.conv2.bias' in name or 'conv1.bias' == name:
                continue
            if '.shortcut.0.bias' in name:
                continue
            if 'num_batches_tracked' in name:
                continue

            if name.startswith('layer1'):
                name = name.replace('layer1', 'block1.layer')
            elif name.startswith('layer2'):
                name = name.replace('layer2', 'block2.layer')
            elif name.startswith('layer3'):
                name = name.replace('layer3', 'block3.layer')
            elif name.startswith('linear'):
                name = name.replace('linear', 'fc')
            name = name.replace('.shortcut.0.', '.convShortcut.')

            if name.startswith('sub_block1'):
                name = name.replace('sub_block1', 'block1')
            elif name.startswith('sub_block2'):
                name = name.replace('sub_block2', 'block2')
            elif name.startswith('sub_block3'):
                name = name.replace('sub_block3', 'block3')

        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    print("Base model loaded successfully")
    return model, num_classes


def create_defended_model(base_model, args, num_classes, device):
    print(f"\nWrapping with defense")
    print(f"  Prototype path: {args.proto_path}")
    print(f"  Layer: {args.layer_name} (dim={args.layer_dim})")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Risk threshold: {args.risk_threshold}")

    config = ConsciousConfig()
    config.conscious_dim = args.conscious_dim
    config.k_list = [int(k) for k in args.k_list.split(',')]
    config.max_steps = args.max_steps
    config.risk_threshold = args.risk_threshold
    config.default_layers = [args.layer_name]

    for risk_level in config.action_rules:
        config.action_rules[risk_level]['layers'] = [args.layer_name]

    print(f"  K values: {config.k_list}")

    defended_model = create_conscious_defense(
        base_model=base_model,
        proto_path=args.proto_path,
        arch=args.arch,
        checkpoint_name=os.path.basename(args.checkpoint),
        layer_name=args.layer_name,
        layer_dim=args.layer_dim,
        num_classes=num_classes,
        ranking_method=args.ranking_method,
        config=config,
        device=device,
        record_trajectory=False
    )

    print("Defense wrapper loaded successfully")
    return defended_model


def evaluate_clean_accuracy(model, test_loader, num_samples, device):
    print(f"\n" + "="*80)
    print("Clean Accuracy Evaluation")
    print("="*80)

    model.eval()
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if total >= num_samples:
                break

            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if (batch_idx + 1) % 10 == 0:
                current_acc = 100. * correct / total
                print(f"  Progress: {total}/{num_samples} samples, "
                      f"Current accuracy: {current_acc:.2f}%")

    elapsed_time = time.time() - start_time
    accuracy = 100. * correct / total

    print(f"\n" + "="*80)
    print("Results")
    print("="*80)
    print(f"Clean Accuracy: {accuracy:.2f}%")
    print(f"Correctly classified: {correct}/{total}")
    print(f"Evaluation time: {elapsed_time:.2f}s")
    print(f"Throughput: {total/elapsed_time:.2f} samples/s")
    print("="*80)

    return accuracy


def main():
    args = get_args()

    # Set default paths relative to project root
    if args.data_dir is None:
        args.data_dir = str(_PROJECT_ROOT / "data")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(_PROJECT_ROOT / "checkpoints")
    if args.output_dir is None:
        args.output_dir = str(_PROJECT_ROOT / "results")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print(f"\nLoading dataset: {args.dataset}")
    test_transform = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(
            root=args.data_dir, train=False, download=False, transform=test_transform
        )
    elif args.dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(
            root=args.data_dir, train=False, download=False, transform=test_transform
        )
    elif args.dataset == 'imagenet':
        raise NotImplementedError("ImageNet support coming soon")

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print(f"Dataset loaded: {len(test_dataset)} samples")

    base_model, num_classes = load_base_model(args)
    base_model = base_model.to(device)

    defended_model = create_defended_model(base_model, args, num_classes, device)
    defended_model = defended_model.to(device)

    accuracy = evaluate_clean_accuracy(
        defended_model, test_loader, args.num_samples, device
    )

    print(f"\nEvaluation complete. Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
