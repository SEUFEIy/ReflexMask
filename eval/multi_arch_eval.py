"""
Multi-Architecture Evaluation
Evaluates defense performance across ResNet-18 and WideResNet-34-10 on CIFAR-10.
"""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
import os
import json
import time
from datetime import datetime
from pathlib import Path

from online.models.resnet_val import InterpMaskedResNet
from online.models.wideresnet_trades import InterpMaskedWideResNet
from online.defense import create_conscious_defense

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = str(_PROJECT_ROOT / "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_CONFIGS = {
    'ResNet-18': {
        'checkpoint': 'Addepalli2022Efficient_RN18.pt',
        'arch': 'rn18_val',
        'layer_name': 'layer4',
        'layer_dim': 512,
        'num_classes': 10,
        'mask_which': 'loir',
        'important_dim': 128,
        'proto_path': 'rn18_val/cifar10/layer4/prototypes.npz',
        'model_class': InterpMaskedResNet
    },
    'WideResNet-34-10': {
        'checkpoint': 'TRADES-AWP_cifar10_linf_wrn34-10.pt',
        'arch': 'wrn34_10',
        'layer_name': 'block3',
        'layer_dim': 640,
        'num_classes': 10,
        'mask_which': 'loir',
        'important_dim': 128,
        'proto_path': 'wrn34_10/cifar10/block3/prototypes.npz',
        'model_class': InterpMaskedWideResNet
    }
}


def load_model(config):
    model = config['model_class'](
        layer_name=config['layer_name'],
        checkpoint_name=os.path.basename(config['checkpoint']),
        mask_which=config['mask_which'],
        important_dim=config['important_dim'],
        num_classes=config['num_classes']
    )

    checkpoint_path = str(_PROJECT_ROOT / "checkpoints" / config['checkpoint'])
    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k

        if config.get('arch') == 'wrn34_10':
            # Skip bias terms not in model
            if '.conv1.bias' in name or '.conv2.bias' in name or 'conv1.bias' == name:
                continue
            if '.shortcut.0.bias' in name:
                continue
            if 'num_batches_tracked' in name:
                continue
            # Map layerN -> blockN.layer
            if name.startswith('layer1'):
                name = name.replace('layer1', 'block1.layer')
            elif name.startswith('layer2'):
                name = name.replace('layer2', 'block2.layer')
            elif name.startswith('layer3'):
                name = name.replace('layer3', 'block3.layer')
            elif name.startswith('linear'):
                name = name.replace('linear', 'fc')
            name = name.replace('.shortcut.0.', '.convShortcut.')

        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def test_clean_accuracy(model, test_loader, device, num_samples=None):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if num_samples and total >= num_samples:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy, total


def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=20):
    images_adv = images.clone().detach()

    for step in range(steps):
        images_adv.requires_grad = True
        try:
            logits = model(images_adv)
            loss = F.cross_entropy(logits, labels)
            grad = torch.autograd.grad(loss, images_adv, retain_graph=False, create_graph=False)[0]
            images_adv = images_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(images_adv - images, min=-eps, max=eps)
            images_adv = torch.clamp(images + delta, min=0, max=1).detach()
        except RuntimeError:
            break

    return images_adv


def fgsm_attack(model, images, labels, eps=8/255):
    images_adv = images.clone().detach()
    images_adv.requires_grad = True
    try:
        logits = model(images_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, images_adv)[0]
        images_adv = images_adv.detach() + eps * grad.sign()
        images_adv = torch.clamp(images_adv, min=0, max=1)
    except RuntimeError:
        pass
    return images_adv


def test_adversarial_robustness(model, test_loader, device, attack_type='pgd-20',
                                 eps=8/255, num_samples=None):
    model.eval()
    correct = 0
    total = 0

    attack_params = {
        'fgsm': {'eps': eps},
        'pgd-10': {'eps': eps, 'alpha': 2/255, 'steps': 10},
        'pgd-20': {'eps': eps, 'alpha': 2/255, 'steps': 20},
        'pgd-50': {'eps': eps, 'alpha': 2/255, 'steps': 50},
    }

    for images, labels in test_loader:
        if num_samples and total >= num_samples:
            break
        images, labels = images.to(device), labels.to(device)

        if attack_type == 'fgsm':
            images_adv = fgsm_attack(model, images, labels, **attack_params[attack_type])
        elif attack_type.startswith('pgd'):
            images_adv = pgd_attack(model, images, labels, **attack_params[attack_type])

        with torch.no_grad():
            logits = model(images_adv)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy, total


def evaluate_single_model(model, test_loader, device, model_name, config,
                           num_samples_clean=5000, num_samples_adv=1000):
    results = {
        'model_name': model_name,
        'arch': config['arch'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")

    print(f"  Testing clean accuracy...")
    start_time = time.time()
    clean_acc, clean_total = test_clean_accuracy(model, test_loader, device, num_samples_clean)
    clean_time = time.time() - start_time

    results['clean_accuracy'] = clean_acc
    results['clean_samples'] = clean_total
    results['clean_time'] = clean_time
    print(f"  Clean Acc: {clean_acc:.2f}% ({clean_total} samples, {clean_time:.1f}s)")

    attack_types = ['fgsm', 'pgd-10', 'pgd-20', 'pgd-50']
    results['adversarial'] = {}

    for attack_type in attack_types:
        print(f"  Testing {attack_type.upper()}...")
        start_time = time.time()
        rob_acc, rob_total = test_adversarial_robustness(
            model, test_loader, device, attack_type=attack_type, num_samples=num_samples_adv
        )
        rob_time = time.time() - start_time

        results['adversarial'][attack_type] = {
            'accuracy': rob_acc,
            'samples': rob_total,
            'time': rob_time
        }
        print(f"  {attack_type.upper()}: {rob_acc:.2f}% ({rob_total} samples, {rob_time:.1f}s)")

    return results


def main():
    print("="*80)
    print("Multi-Architecture Defense Evaluation")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\nLoading CIFAR-10 Test Set...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root=str(_PROJECT_ROOT / "data"), train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    print(f"Total samples: {len(test_dataset)}")

    NUM_SAMPLES_CLEAN = 5000
    NUM_SAMPLES_ADV = 1000

    all_results = []

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Architecture: {model_name}")
        print(f"{'='*80}")

        print(f"\n[1/2] Base {model_name}")
        base_model = load_model(config)
        base_model = base_model.to(device)

        base_results = evaluate_single_model(
            base_model, test_loader, device,
            f"Base {model_name}", config,
            num_samples_clean=NUM_SAMPLES_CLEAN,
            num_samples_adv=NUM_SAMPLES_ADV
        )
        all_results.append(base_results)

        print(f"\n[2/2] Conscious {model_name}")
        conscious_model = create_conscious_defense(
            base_model=base_model,
            proto_path=str(_PROJECT_ROOT / "saved_prototypes" / config['proto_path']),
            arch=config['arch'],
            checkpoint_name=os.path.basename(config['checkpoint']),
            layer_name=config['layer_name'],
            layer_dim=config['layer_dim'],
            num_classes=config['num_classes'],
            ranking_method=config['mask_which'],
            device=device,
            record_trajectory=False
        )

        conscious_results = evaluate_single_model(
            conscious_model, test_loader, device,
            f"Conscious {model_name}", config,
            num_samples_clean=NUM_SAMPLES_CLEAN,
            num_samples_adv=NUM_SAMPLES_ADV
        )
        all_results.append(conscious_results)

    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"{RESULTS_DIR}/multi_arch_results_{timestamp}.json"

    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved: {json_file}")
    print_summary(all_results)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")


def print_summary(results):
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}\n")

    print(f"{'Model':<35} {'Clean Acc':<15} {'FGSM':<12} {'PGD-20':<12}")
    print("-" * 80)
    for result in results:
        model_name = result['model_name']
        clean_acc = result['clean_accuracy']
        fgsm_acc = result['adversarial']['fgsm']['accuracy']
        pgd20_acc = result['adversarial']['pgd-20']['accuracy']
        print(f"{model_name:<35} {clean_acc:>6.2f}%{'':<8} {fgsm_acc:>6.2f}%{'':<5} {pgd20_acc:>6.2f}%")

    print()


if __name__ == "__main__":
    main()
