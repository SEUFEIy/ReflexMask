"""
Image-Wise Worst-Case (IW-WC) Robust Accuracy Evaluation
Evaluates model robustness under multiple attack methods.
"""
import argparse
import json
import os
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from online.models import resnet_val
from online.models.resnet50 import ResNet50
from online.models.wideresnet_trades import WideResNet34_10

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def get_args():
    parser = argparse.ArgumentParser(description='IW-WC Robust Accuracy Evaluation')
    
    # Model config
    parser.add_argument('--arch', type=str, required=True, 
                        choices=['rn18_val', 'rn50', 'wrn34_10'],
                        help='Architecture name')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    
    # Dataset config
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset name')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to dataset directory (default: PROJECT_ROOT/data)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory containing checkpoints (default: PROJECT_ROOT/checkpoints)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: PROJECT_ROOT/results)')
    
    # Evaluation config
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (recommend 1 for IW-WC)')
    parser.add_argument('--epsilon', type=int, default=8,
                        help='Attack epsilon (will be divided by 255)')
    
    # Attack config
    parser.add_argument('--attacks', type=str, nargs='+',
                        default=['fgsm', 'pgd-10', 'pgd-20', 'pgd-50', 'mifgsm'],
                        help='List of attacks to use')
    parser.add_argument('--use-autoattack', action='store_true',
                        help='Include AutoAttack in evaluation (very slow)')
    parser.add_argument('--save-detailed', action='store_true',
                        help='Save detailed per-sample results')
    
    # Device config
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def load_model(args):
    """Load model based on architecture"""
    num_classes = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000}[args.dataset]

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(args.checkpoint_dir, os.path.basename(args.checkpoint))

    # Create model
    if args.arch == 'rn18_val':
        from online.models.resnet_val import ResNet18
        model = ResNet18(num_classes=num_classes)
    elif args.arch == 'rn50':
        model = ResNet50(num_classes=num_classes)
    elif args.arch == 'wrn34_10':
        model = WideResNet34_10(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
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
    model = model.to(args.device)
    model.eval()
    
    return model


def get_dataloader(args):
    """Get dataset and dataloader"""
    if args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=args.data_dir, train=False, 
                                   download=True, transform=transform)
    elif args.dataset == 'cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR100(root=args.data_dir, train=False,
                                    download=True, transform=transform)
    elif args.dataset == 'imagenet':
        # ImageNet preprocessing
        transform = transforms.Compose([
            transforms.CenterCrop(288),
            transforms.ToTensor(),
        ])
        testset = datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'imagenet', 'val'),
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Limit to num_samples
    if args.num_samples > 0 and args.num_samples < len(testset):
        indices = list(range(args.num_samples))
        testset = Subset(testset, indices)
    
    loader = DataLoader(testset, batch_size=args.batch_size, 
                       shuffle=False, num_workers=4)
    
    return loader


def fgsm_attack(model, images, labels, eps=8/255):
    """FGSM Attack"""
    images_adv = images.clone().detach()
    images_adv.requires_grad = True
    
    outputs = model(images_adv)
    loss = F.cross_entropy(outputs, labels)
    
    grad = torch.autograd.grad(loss, images_adv)[0]
    images_adv = images_adv.detach() + eps * grad.sign()
    images_adv = torch.clamp(images_adv, 0, 1)
    
    return images_adv


def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=20):
    """PGD Attack"""
    images_adv = images.clone().detach()
    
    for _ in range(steps):
        images_adv.requires_grad = True
        
        outputs = model(images_adv)
        loss = F.cross_entropy(outputs, labels)
        
        grad = torch.autograd.grad(loss, images_adv)[0]
        images_adv = images_adv.detach() + alpha * grad.sign()
        
        # Project back to epsilon ball
        delta = torch.clamp(images_adv - images, -eps, eps)
        images_adv = torch.clamp(images + delta, 0, 1).detach()
    
    return images_adv


def mifgsm_attack(model, images, labels, eps=8/255, alpha=2/255, 
                  steps=10, decay=1.0):
    """MI-FGSM Attack (Momentum Iterative FGSM)"""
    images_adv = images.clone().detach()
    momentum = torch.zeros_like(images)
    
    for _ in range(steps):
        images_adv.requires_grad = True
        
        outputs = model(images_adv)
        loss = F.cross_entropy(outputs, labels)
        
        grad = torch.autograd.grad(loss, images_adv)[0]
        
        # Update momentum
        grad_norm = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = decay * momentum + grad_norm
        
        images_adv = images_adv.detach() + alpha * momentum.sign()
        
        # Project back to epsilon ball
        delta = torch.clamp(images_adv - images, -eps, eps)
        images_adv = torch.clamp(images + delta, 0, 1).detach()
    
    return images_adv


def run_attack(model, images, labels, attack_name, eps):
    """Run a specific attack"""
    if attack_name == 'fgsm':
        return fgsm_attack(model, images, labels, eps)
    elif attack_name.startswith('pgd-'):
        steps = int(attack_name.split('-')[1])
        return pgd_attack(model, images, labels, eps, alpha=2/255, steps=steps)
    elif attack_name == 'mifgsm':
        return mifgsm_attack(model, images, labels, eps)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def evaluate_iwwc(model, loader, args):
    """
    Evaluate IW-WC (Image-Wise Worst-Case) robust accuracy
    For each image, find the worst attack and count if model is robust to all attacks
    """
    print(f"\n{'='*80}")
    print("IW-WC Evaluation")
    print(f"{'='*80}")
    print(f"Model: {args.arch}")
    print(f"Dataset: {args.dataset}")
    print(f"Epsilon: {args.epsilon}/255")
    print(f"Attacks: {args.attacks}")
    print(f"Use AutoAttack: {args.use_autoattack}")
    print(f"Number of samples: {args.num_samples}")
    print(f"{'='*80}\n")
    
    eps = args.epsilon / 255.0
    
    # Statistics
    total_samples = 0
    iwwc_robust_count = 0  # Robust to ALL attacks
    individual_robust_counts = {attack: 0 for attack in args.attacks}
    if args.use_autoattack:
        individual_robust_counts['autoattack'] = 0
    
    # Detailed results (optional)
    detailed_results = [] if args.save_detailed else None
    
    # AutoAttack setup (if needed)
    if args.use_autoattack:
        try:
            from autoattack import AutoAttack
            adversary = AutoAttack(model, norm='Linf', eps=eps, 
                                  version='standard', device=args.device)
            print("✅ AutoAttack loaded successfully")
        except ImportError:
            print("⚠️  AutoAttack not installed, skipping")
            print("   Install: pip install git+https://github.com/fra31/auto-attack")
            args.use_autoattack = False
    
    model.eval()
    
    # Evaluate each sample
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Evaluating")):
        if total_samples >= args.num_samples:
            break
        
        images = images.to(args.device)
        labels = labels.to(args.device)
        
        batch_size = images.shape[0]
        
        # Get clean prediction
        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
        
        # Track if sample is robust to each attack
        is_robust_to_attack = {attack: torch.ones(batch_size, dtype=torch.bool, device=args.device)
                              for attack in args.attacks}
        
        # Run each attack
        for attack_name in args.attacks:
            images_adv = run_attack(model, images, labels, attack_name, eps)
            
            with torch.no_grad():
                adv_outputs = model(images_adv)
                adv_preds = adv_outputs.argmax(dim=1)
            
            # Check if prediction is correct (robust)
            is_robust = (adv_preds == labels)
            is_robust_to_attack[attack_name] = is_robust
            individual_robust_counts[attack_name] += is_robust.sum().item()
        
        # Run AutoAttack if enabled
        if args.use_autoattack:
            try:
                # AutoAttack requires numpy arrays
                x_np = images.cpu().numpy()
                y_np = labels.cpu().numpy()
                
                x_adv = adversary.run_standard_evaluation(
                    torch.from_numpy(x_np).to(args.device),
                    torch.from_numpy(y_np).to(args.device),
                    bs=batch_size
                )
                
                with torch.no_grad():
                    aa_outputs = model(x_adv)
                    aa_preds = aa_outputs.argmax(dim=1)
                
                is_robust = (aa_preds == labels)
                is_robust_to_attack['autoattack'] = is_robust
                individual_robust_counts['autoattack'] += is_robust.sum().item()
            except Exception as e:
                print(f"\n⚠️  AutoAttack failed for batch {batch_idx}: {e}")
                is_robust_to_attack['autoattack'] = torch.zeros(batch_size, dtype=torch.bool, device=args.device)
        
        # Check if robust to ALL attacks (IW-WC)
        is_iwwc_robust = torch.ones(batch_size, dtype=torch.bool, device=args.device)
        for attack_robust in is_robust_to_attack.values():
            is_iwwc_robust = is_iwwc_robust & attack_robust
        
        iwwc_robust_count += is_iwwc_robust.sum().item()
        total_samples += batch_size
        
        # Save detailed results if requested
        if args.save_detailed:
            for i in range(batch_size):
                sample_result = {
                    'sample_idx': total_samples - batch_size + i,
                    'true_label': labels[i].item(),
                    'clean_pred': clean_preds[i].item(),
                    'iwwc_robust': is_iwwc_robust[i].item(),
                }
                for attack_name, robust in is_robust_to_attack.items():
                    sample_result[f'{attack_name}_robust'] = robust[i].item()
                detailed_results.append(sample_result)
        
        # Progress update
        if (batch_idx + 1) % 50 == 0:
            current_iwwc_acc = 100.0 * iwwc_robust_count / total_samples
            print(f"\n  Progress: {total_samples}/{args.num_samples} samples")
            print(f"  Current IW-WC Robust Acc: {current_iwwc_acc:.2f}%")
    
    # Compute final accuracies
    iwwc_robust_accuracy = 100.0 * iwwc_robust_count / total_samples
    individual_accuracies = {
        attack: 100.0 * count / total_samples
        for attack, count in individual_robust_counts.items()
    }
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"\n IW-WC Robust Accuracy: {iwwc_robust_accuracy:.2f}%")
    print(f"   (Robust to ALL attacks: {iwwc_robust_count}/{total_samples})")
    print(f"\n Individual Attack Accuracies:")
    for attack, acc in individual_accuracies.items():
        count = individual_robust_counts[attack]
        print(f"   {attack:20s}: {acc:6.2f}% ({count}/{total_samples})")
    print(f"{'='*80}\n")
    
    # Prepare results dict
    results = {
        'model': args.arch,
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'epsilon': args.epsilon,
        'num_samples': args.num_samples,
        'total_tested': total_samples,
        'attacks': args.attacks,
        'use_autoattack': args.use_autoattack,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'results': {
            'iwwc_robust_accuracy': iwwc_robust_accuracy,
            'iwwc_robust_count': iwwc_robust_count,
            'individual_accuracies': individual_accuracies,
            'individual_counts': individual_robust_counts,
        }
    }
    
    if args.save_detailed:
        results['detailed_results'] = detailed_results
    
    return results


def save_results(results, args):
    """Save evaluation results"""
    # Create output directory
    model_name = args.arch
    dataset_name = args.dataset
    output_dir = os.path.join(args.output_dir, model_name, dataset_name, 'iwwc')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f'iwwc_results_{timestamp}.json')
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {json_path}")
    
    # Save summary text file
    txt_path = os.path.join(output_dir, f'iwwc_summary_{timestamp}.txt')
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("IW-WC ROBUST ACCURACY EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {results['model']}\n")
        f.write(f"Checkpoint: {results['checkpoint']}\n")
        f.write(f"Dataset: {results['dataset']}\n")
        f.write(f"Epsilon: {results['epsilon']}/255\n")
        f.write(f"Attacks: {', '.join(results['attacks'])}\n")
        f.write(f"Use AutoAttack: {results['use_autoattack']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Total samples: {results['total_tested']}\n\n")
        f.write("="*80 + "\n")
        f.write("RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f" IW-WC Robust Accuracy: {results['results']['iwwc_robust_accuracy']:.2f}%\n")
        f.write(f"   Robust to ALL attacks: {results['results']['iwwc_robust_count']}/{results['total_tested']}\n\n")
        f.write(" Individual Attack Accuracies:\n")
        for attack, acc in results['results']['individual_accuracies'].items():
            count = results['results']['individual_counts'][attack]
            f.write(f"   {attack:20s}: {acc:6.2f}% ({count}/{results['total_tested']})\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Summary saved to: {txt_path}")
    
    return json_path, txt_path


def main():
    args = get_args()

    # Set default paths relative to project root
    if args.data_dir is None:
        args.data_dir = str(_PROJECT_ROOT / "data")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(_PROJECT_ROOT / "checkpoints")
    if args.output_dir is None:
        args.output_dir = str(_PROJECT_ROOT / "results")

    print(f"\n{'='*80}")
    print("IW-WC (Image-Wise Worst-Case) Robust Accuracy Evaluation")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args)
    print(f"✅ Model loaded: {args.arch}")
    
    # Get dataloader
    print("\nLoading dataset...")
    loader = get_dataloader(args)
    print(f"✅ Dataset loaded: {args.dataset}")
    print(f"   Samples to evaluate: {args.num_samples}")
    
    # Run evaluation
    results = evaluate_iwwc(model, loader, args)
    
    # Save results
    save_results(results, args)
    
    print("\n Evaluation complete!")


if __name__ == "__main__":
    main()
