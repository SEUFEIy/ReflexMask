"""
WideResNet-34-10 Evaluation
Evaluates base vs defended WideResNet-34-10 on CIFAR-10.
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

from online.models.wideresnet_trades import InterpMaskedWideResNet
from online.defense import create_conscious_defense

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = str(_PROJECT_ROOT / "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model(checkpoint_path):
    model = InterpMaskedWideResNet(
        layer_name='block3',
        checkpoint_name='TRADES-AWP_cifar10_linf_wrn34-10.pt',
        mask_which='loir',
        important_dim=128,
        num_classes=10
    )

    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
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
            if total % 500 == 0:
                print(f"    Progress: {total} samples, Acc: {100.0*correct/total:.2f}%")

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

        if total % 100 == 0:
            print(f"    Progress: {total} samples, Robust Acc: {100.0*correct/total:.2f}%")

    accuracy = 100.0 * correct / total
    return accuracy, total


def evaluate_model(model, test_loader, device, model_name, num_samples_clean=5000, num_samples_adv=1000):
    results = {
        'model_name': model_name,
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
    print("WideResNet-34-10 Evaluation")
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

    results_list = []

    print(f"\n[1/2] Base WideResNet-34-10")
    checkpoint_path = str(_PROJECT_ROOT / "checkpoints" / "TRADES-AWP_cifar10_linf_wrn34-10.pt")
    base_model = load_model(checkpoint_path)
    base_model = base_model.to(device)

    base_results = evaluate_model(
        base_model, test_loader, device,
        "Base WideResNet-34-10",
        num_samples_clean=NUM_SAMPLES_CLEAN,
        num_samples_adv=NUM_SAMPLES_ADV
    )
    results_list.append(base_results)

    print(f"\n[2/2] Conscious WideResNet-34-10")
    proto_path = str(_PROJECT_ROOT / "saved_prototypes" / "wrn34_10/cifar10/block3/prototypes.npz")

    conscious_model = create_conscious_defense(
        base_model=base_model,
        proto_path=proto_path,
        arch='wrn34_10',
        checkpoint_name='TRADES-AWP_cifar10_linf_wrn34-10.pt',
        layer_name='block3',
        layer_dim=640,
        num_classes=10,
        ranking_method='loir',
        device=device,
        record_trajectory=False
    )

    conscious_results = evaluate_model(
        conscious_model, test_loader, device,
        "Conscious WideResNet-34-10",
        num_samples_clean=NUM_SAMPLES_CLEAN,
        num_samples_adv=NUM_SAMPLES_ADV
    )
    results_list.append(conscious_results)

    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"{RESULTS_DIR}/wrn_results_{timestamp}.json"

    with open(json_file, 'w') as f:
        json.dump(results_list, f, indent=2)

    print(f"Results saved: {json_file}")

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}\n")

    print(f"{'Model':<35} {'Clean Acc':<15} {'FGSM':<12} {'PGD-20':<12}")
    print("-" * 80)
    for result in results_list:
        model_name = result['model_name']
        clean_acc = result['clean_accuracy']
        fgsm_acc = result['adversarial']['fgsm']['accuracy']
        pgd20_acc = result['adversarial']['pgd-20']['accuracy']
        print(f"{model_name:<35} {clean_acc:>6.2f}%{'':<8} {fgsm_acc:>6.2f}%{'':<5} {pgd20_acc:>6.2f}%")

    if len(results_list) == 2:
        print(f"\n{'='*80}")
        print("Improvement Analysis")
        print(f"{'='*80}\n")
        base = results_list[0]
        conscious = results_list[1]

        clean_diff = conscious['clean_accuracy'] - base['clean_accuracy']
        print(f"Clean Accuracy:  {base['clean_accuracy']:.2f}% -> {conscious['clean_accuracy']:.2f}% ({clean_diff:+.2f}%)")

        for attack in ['fgsm', 'pgd-10', 'pgd-20', 'pgd-50']:
            base_acc = base['adversarial'][attack]['accuracy']
            cons_acc = conscious['adversarial'][attack]['accuracy']
            diff = cons_acc - base_acc
            print(f"{attack.upper():<12}: {base_acc:.2f}% -> {cons_acc:.2f}% ({diff:+.2f}%)")

    print(f"\n{'='*80}")
    print("WideResNet-34-10 Evaluation Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
