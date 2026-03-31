"""
Complete Evaluation Pipeline
Evaluates base vs defended model on CIFAR-10 with multiple attack methods.
"""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import OrderedDict
import os
import json
import time
from datetime import datetime
from pathlib import Path

from online.models.resnet_val import InterpMaskedResNet
from online.defense import create_conscious_defense

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = str(_PROJECT_ROOT / "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model(checkpoint_path, num_classes=10):
    model = InterpMaskedResNet(
        layer_name='layer4',
        checkpoint_name='Addepalli2022Efficient_RN18.pt',
        mask_which='loir',
        important_dim=128,
        num_classes=num_classes
    )

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(_PROJECT_ROOT / "checkpoints", os.path.basename(checkpoint_path))
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


def test_clean_accuracy(model, test_loader, device, num_samples=None, desc=""):
    model.eval()
    correct = 0
    total = 0
    print(f"  Testing clean accuracy{' (' + desc + ')' if desc else ''}...")

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


def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=20, device='cuda'):
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


def fgsm_attack(model, images, labels, eps=8/255, device='cuda'):
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


def test_adversarial_robustness(model, test_loader, device, attack_type='pgd',
                                 eps=8/255, num_samples=None, desc=""):
    model.eval()
    correct = 0
    total = 0

    attack_params = {
        'fgsm': {'eps': eps},
        'pgd-10': {'eps': eps, 'alpha': 2/255, 'steps': 10},
        'pgd-20': {'eps': eps, 'alpha': 2/255, 'steps': 20},
        'pgd-50': {'eps': eps, 'alpha': 2/255, 'steps': 50},
    }

    print(f"  Testing {attack_type} attack{' (' + desc + ')' if desc else ''}...")

    for images, labels in test_loader:
        if num_samples and total >= num_samples:
            break
        images, labels = images.to(device), labels.to(device)

        if attack_type == 'fgsm':
            images_adv = fgsm_attack(model, images, labels, **attack_params[attack_type])
        elif attack_type.startswith('pgd'):
            images_adv = pgd_attack(model, images, labels, device=device, **attack_params[attack_type])
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        with torch.no_grad():
            logits = model(images_adv)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        if total % 100 == 0:
            print(f"    Progress: {total} samples, Robust Acc: {100.0*correct/total:.2f}%")

    accuracy = 100.0 * correct / total
    return accuracy, total


def evaluate_model(model, test_loader, device, model_name, num_samples_clean=5000,
                   num_samples_adv=1000):
    results = {
        'model_name': model_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_samples_clean': num_samples_clean,
        'num_samples_adv': num_samples_adv,
    }

    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")

    start_time = time.time()
    clean_acc, clean_total = test_clean_accuracy(
        model, test_loader, device, num_samples_clean, desc=model_name
    )
    clean_time = time.time() - start_time

    results['clean_accuracy'] = clean_acc
    results['clean_samples'] = clean_total
    results['clean_time'] = clean_time
    print(f"  Clean Accuracy: {clean_acc:.2f}% ({clean_total} samples, {clean_time:.1f}s)")

    attack_types = ['fgsm', 'pgd-10', 'pgd-20', 'pgd-50']
    results['adversarial'] = {}

    for attack_type in attack_types:
        start_time = time.time()
        rob_acc, rob_total = test_adversarial_robustness(
            model, test_loader, device, attack_type=attack_type,
            num_samples=num_samples_adv, desc=model_name
        )
        rob_time = time.time() - start_time

        results['adversarial'][attack_type] = {
            'accuracy': rob_acc,
            'samples': rob_total,
            'time': rob_time
        }
        print(f"  {attack_type.upper()} Robust Acc: {rob_acc:.2f}% ({rob_total} samples, {rob_time:.1f}s)")

    return results


def print_comparison(results_list):
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Model':<30} {'Clean Acc':<15} {'Samples':<10} {'Time (s)':<10}")
    print("-" * 80)
    for results in results_list:
        print(f"{results['model_name']:<30} {results['clean_accuracy']:>6.2f}%{'':<8} "
              f"{results['clean_samples']:<10} {results['clean_time']:<10.1f}")
    print()

    attack_types = ['fgsm', 'pgd-10', 'pgd-20', 'pgd-50']
    for attack_type in attack_types:
        print(f"\n{attack_type.upper()} Attack Results:")
        print(f"{'Model':<30} {'Robust Acc':<15} {'Samples':<10} {'Time (s)':<10}")
        print("-" * 80)
        for results in results_list:
            adv_result = results['adversarial'][attack_type]
            print(f"{results['model_name']:<30} {adv_result['accuracy']:>6.2f}%{'':<8} "
                  f"{adv_result['samples']:<10} {adv_result['time']:<10.1f}")

    if len(results_list) == 2:
        print(f"\n{'='*80}")
        print("IMPROVEMENT (Defended vs Base)")
        print(f"{'='*80}\n")

        base_results = results_list[0]
        conscious_results = results_list[1]

        clean_diff = conscious_results['clean_accuracy'] - base_results['clean_accuracy']
        print(f"Clean Accuracy:     {base_results['clean_accuracy']:.2f}% -> "
              f"{conscious_results['clean_accuracy']:.2f}% ({clean_diff:+.2f}%)")

        for attack_type in attack_types:
            base_rob = base_results['adversarial'][attack_type]['accuracy']
            conscious_rob = conscious_results['adversarial'][attack_type]['accuracy']
            rob_diff = conscious_rob - base_rob
            print(f"{attack_type.upper():<12}:    {base_rob:.2f}% -> "
                  f"{conscious_rob:.2f}% ({rob_diff:+.2f}%)")


def save_results(results_list, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{RESULTS_DIR}/evaluation_results_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results_list, f, indent=2)

    print(f"\nResults saved to: {filename}")
    return filename


def save_summary_report(results_list, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{RESULTS_DIR}/evaluation_report_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEFENSE EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Models Evaluated: {len(results_list)}\n\n")

        f.write("-"*80 + "\n")
        f.write("CLEAN ACCURACY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<30} {'Clean Acc':<15} {'Samples':<10} {'Time (s)':<10}\n")
        f.write("-"*80 + "\n")
        for results in results_list:
            f.write(f"{results['model_name']:<30} {results['clean_accuracy']:>6.2f}%{'':<8} "
                   f"{results['clean_samples']:<10} {results['clean_time']:<10.1f}\n")

        attack_types = ['fgsm', 'pgd-10', 'pgd-20', 'pgd-50']
        for attack_type in attack_types:
            f.write("\n" + "-"*80 + "\n")
            f.write(f"{attack_type.upper()} ATTACK\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Model':<30} {'Robust Acc':<15} {'Samples':<10} {'Time (s)':<10}\n")
            f.write("-"*80 + "\n")
            for results in results_list:
                adv_result = results['adversarial'][attack_type]
                f.write(f"{results['model_name']:<30} {adv_result['accuracy']:>6.2f}%{'':<8} "
                       f"{adv_result['samples']:<10} {adv_result['time']:<10.1f}\n")

        if len(results_list) == 2:
            f.write("\n" + "="*80 + "\n")
            f.write("IMPROVEMENT ANALYSIS\n")
            f.write("="*80 + "\n\n")

            base_results = results_list[0]
            conscious_results = results_list[1]

            clean_diff = conscious_results['clean_accuracy'] - base_results['clean_accuracy']
            f.write(f"Clean Accuracy:\n")
            f.write(f"  Base:       {base_results['clean_accuracy']:.2f}%\n")
            f.write(f"  Defended:   {conscious_results['clean_accuracy']:.2f}%\n")
            f.write(f"  Difference: {clean_diff:+.2f}%\n\n")

            for attack_type in attack_types:
                base_rob = base_results['adversarial'][attack_type]['accuracy']
                conscious_rob = conscious_results['adversarial'][attack_type]['accuracy']
                rob_diff = conscious_rob - base_rob
                f.write(f"{attack_type.upper()} Attack:\n")
                f.write(f"  Base:       {base_rob:.2f}%\n")
                f.write(f"  Defended:   {conscious_rob:.2f}%\n")
                f.write(f"  Difference: {rob_diff:+.2f}%\n\n")

    print(f"Summary report saved to: {filename}")
    return filename


def main():
    print("="*80)
    print("COMPLETE DEFENSE EVALUATION")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Results directory: {RESULTS_DIR}")

    print("\n" + "="*80)
    print("Loading CIFAR-10 Test Set")
    print("="*80)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root=str(_PROJECT_ROOT / "data"), train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    print(f"Total test samples: {len(test_dataset)}")

    NUM_SAMPLES_CLEAN = 5000
    NUM_SAMPLES_ADV = 1000
    print(f"Clean accuracy samples: {NUM_SAMPLES_CLEAN}")
    print(f"Adversarial robustness samples: {NUM_SAMPLES_ADV}")

    results_list = []

    print("\n" + "="*80)
    print("STEP 1: Evaluating Base Model")
    print("="*80)
    base_model = load_model('checkpoints/Addepalli2022Efficient_RN18.pt', num_classes=10)
    base_model = base_model.to(device)

    base_results = evaluate_model(
        base_model, test_loader, device,
        "Base Model",
        num_samples_clean=NUM_SAMPLES_CLEAN,
        num_samples_adv=NUM_SAMPLES_ADV
    )
    results_list.append(base_results)

    print("\n" + "="*80)
    print("STEP 2: Evaluating Defended Model")
    print("="*80)

    proto_path = str(_PROJECT_ROOT / "saved_prototypes/rn18_val/cifar10/layer4/prototypes.npz")

    conscious_model = create_conscious_defense(
        base_model=base_model,
        proto_path=proto_path,
        arch='rn18_val',
        checkpoint_name='Addepalli2022Efficient_RN18.pt',
        layer_name='layer4',
        layer_dim=512,
        num_classes=10,
        ranking_method='loir',
        device=device,
        record_trajectory=False
    )

    conscious_results = evaluate_model(
        conscious_model, test_loader, device,
        "Defended Model",
        num_samples_clean=NUM_SAMPLES_CLEAN,
        num_samples_adv=NUM_SAMPLES_ADV
    )
    results_list.append(conscious_results)

    print_comparison(results_list)

    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)

    json_file = save_results(results_list)
    report_file = save_summary_report(results_list)

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - JSON:   {json_file}")
    print(f"  - Report: {report_file}")
    print()


if __name__ == "__main__":
    main()
