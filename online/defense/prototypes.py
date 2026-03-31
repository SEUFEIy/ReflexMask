"""
Prototype Manager for Conscious States
Manages class-conditional consciousness prototypes (mean and optional covariance)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from .config import ConsciousConfig, DEFAULT_CONFIG


class PrototypeManager:
    """
    Manage class-conditional prototypes for consciousness states.
    Stores mean (and optionally covariance) for each class.
    """
    
    def __init__(
        self,
        num_classes: int,
        conscious_dim: int,
        config: Optional[ConsciousConfig] = None,
        device: str = "cuda"
    ):
        """
        Args:
            num_classes: Number of classes
            conscious_dim: Dimension of consciousness state
            config: Configuration
            device: Device to run on
        """
        self.num_classes = num_classes
        self.conscious_dim = conscious_dim
        self.config = config or DEFAULT_CONFIG
        self.device = device
        
        # Prototypes: [num_classes, conscious_dim]
        self.means = torch.zeros(num_classes, conscious_dim).to(device)
        
        # Optional covariance: [num_classes, conscious_dim, conscious_dim]
        if self.config.use_covariance:
            self.covariances = torch.zeros(num_classes, conscious_dim, conscious_dim).to(device)
        else:
            self.covariances = None
        
        # Sample counts per class
        self.counts = torch.zeros(num_classes).to(device)
        
        self.is_computed = False
    
    def update(self, c: torch.Tensor, labels: torch.Tensor):
        """
        Update prototypes with new consciousness states (online)
        
        Args:
            c: Consciousness states [batch, conscious_dim]
            labels: Class labels [batch]
        """
        for cls in range(self.num_classes):
            mask = (labels == cls)
            if mask.sum() == 0:
                continue
            
            c_cls = c[mask]  # [n_cls, conscious_dim]
            n = c_cls.shape[0]
            
            # Update mean (incremental)
            old_count = self.counts[cls]
            new_count = old_count + n
            
            if old_count == 0:
                self.means[cls] = c_cls.mean(dim=0)
            else:
                self.means[cls] = (old_count * self.means[cls] + c_cls.sum(dim=0)) / new_count
            
            # Update covariance if needed
            if self.covariances is not None:
                if old_count == 0:
                    centered = c_cls - c_cls.mean(dim=0, keepdim=True)
                    self.covariances[cls] = torch.matmul(centered.T, centered) / n
                else:
                    # Incremental covariance update (simplified)
                    centered = c_cls - self.means[cls]
                    cov_batch = torch.matmul(centered.T, centered) / n
                    self.covariances[cls] = (old_count * self.covariances[cls] + n * cov_batch) / new_count
            
            self.counts[cls] = new_count
    
    def compute_from_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        extractor: 'ConsciousStateExtractor',
        layer_name: str = "layer4",
        use_pseudo_labels: bool = False
    ):
        """
        Compute prototypes from a calibration dataset
        
        Args:
            dataloader: Data loader for calibration set
            model: Neural network model
            extractor: Consciousness state extractor
            layer_name: Which layer to extract from
            use_pseudo_labels: Use model predictions as labels if True
        """
        model.eval()
        
        all_states = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass to get activations
                # Assume model has been modified to return activations
                if hasattr(model, 'forward') and 'return_feats' in model.forward.__code__.co_varnames:
                    logits, feats = model(images, return_feats=True)
                    h = feats[layer_name] if isinstance(feats, dict) else feats
                else:
                    # Fallback: hook-based extraction
                    activations = {}
                    def hook_fn(name):
                        def hook(module, input, output):
                            activations[name] = output
                        return hook
                    
                    # Register hook
                    layer = dict(model.named_modules())[layer_name]
                    handle = layer.register_forward_hook(hook_fn(layer_name))
                    
                    logits = model(images)
                    h = activations[layer_name]
                    
                    handle.remove()
                
                # Extract consciousness states
                c = extractor.extract(h)
                
                # Use pseudo labels if requested
                if use_pseudo_labels:
                    labels = torch.argmax(logits, dim=1)
                
                all_states.append(c.cpu())
                all_labels.append(labels.cpu())
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{len(dataloader)} batches")
        
        # Concatenate all
        all_states = torch.cat(all_states, dim=0).to(self.device)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)
        
        # Compute prototypes
        for cls in range(self.num_classes):
            mask = (all_labels == cls)
            n_samples = mask.sum().item()
            
            if n_samples < self.config.min_samples_per_class:
                print(f"Warning: Class {cls} has only {n_samples} samples (min: {self.config.min_samples_per_class})")
                if n_samples == 0:
                    continue
            
            c_cls = all_states[mask]
            
            # Compute mean
            self.means[cls] = c_cls.mean(dim=0)
            
            # Compute covariance if needed
            if self.covariances is not None:
                centered = c_cls - self.means[cls]
                self.covariances[cls] = torch.matmul(centered.T, centered) / n_samples
                # Add small regularization for numerical stability
                self.covariances[cls] += torch.eye(self.conscious_dim).to(self.device) * 1e-4
            
            self.counts[cls] = n_samples
        
        self.is_computed = True
        print(f"Prototypes computed for {self.num_classes} classes")
        print(f"Sample counts: {self.counts.cpu().numpy()}")
    
    def get_prototype(self, class_idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get prototype for a specific class
        
        Args:
            class_idx: Class index
        
        Returns:
            mean: [conscious_dim]
            covariance: [conscious_dim, conscious_dim] or None
        """
        mean = self.means[class_idx]
        cov = self.covariances[class_idx] if self.covariances is not None else None
        return mean, cov
    
    def save(self, save_dir: str, arch: str, dataset: str, layer_name: str):
        """
        Save prototypes to disk
        
        Args:
            save_dir: Base directory (e.g., saved_conscious_prototypes)
            arch: Architecture name
            dataset: Dataset name
            layer_name: Layer name
        """
        save_path = os.path.join(save_dir, arch, dataset, layer_name)
        os.makedirs(save_path, exist_ok=True)
        
        filename = os.path.join(save_path, "prototypes.npz")
        
        save_dict = {
            'means': self.means.cpu().numpy(),
            'counts': self.counts.cpu().numpy(),
            'num_classes': self.num_classes,
            'conscious_dim': self.conscious_dim,
            'use_covariance': self.config.use_covariance
        }
        
        if self.covariances is not None:
            save_dict['covariances'] = self.covariances.cpu().numpy()
        
        np.savez(filename, **save_dict)
        print(f"Prototypes saved to {filename}")
    
    @classmethod
    def load(
        cls,
        save_dir: str,
        arch: str,
        dataset: str,
        layer_name: str,
        config: Optional[ConsciousConfig] = None,
        device: str = "cuda"
    ) -> 'PrototypeManager':
        """
        Load prototypes from disk
        
        Args:
            save_dir: Base directory
            arch: Architecture name
            dataset: Dataset name
            layer_name: Layer name
            config: Configuration
            device: Device
        
        Returns:
            Loaded PrototypeManager
        """
        filename = os.path.join(save_dir, arch, dataset, layer_name, "prototypes.npz")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Prototypes not found: {filename}")
        
        data = np.load(filename)
        
        num_classes = int(data['num_classes'])
        conscious_dim = int(data['conscious_dim'])
        
        manager = cls(num_classes, conscious_dim, config, device)
        
        manager.means = torch.from_numpy(data['means']).float().to(device)
        manager.counts = torch.from_numpy(data['counts']).float().to(device)
        
        if 'covariances' in data:
            manager.covariances = torch.from_numpy(data['covariances']).float().to(device)
        
        manager.is_computed = True
        
        print(f"Prototypes loaded from {filename}")
        return manager
    
    def get_all_means(self) -> torch.Tensor:
        """Get all prototype means: [num_classes, conscious_dim]"""
        return self.means
    
    def distance_to_prototype(
        self,
        c: torch.Tensor,
        class_idx: int,
        metric: str = "cosine"
    ) -> torch.Tensor:
        """
        Compute distance from consciousness state to prototype
        
        Args:
            c: Consciousness state [batch, conscious_dim]
            class_idx: Target class
            metric: "cosine" or "euclidean"
        
        Returns:
            distance: [batch]
        """
        mean, cov = self.get_prototype(class_idx)
        
        if metric == "cosine":
            # Cosine distance: 1 - cosine_similarity
            similarity = F.cosine_similarity(c, mean.unsqueeze(0), dim=1)
            distance = 1.0 - similarity
        elif metric == "euclidean":
            # Euclidean distance
            distance = torch.norm(c - mean, dim=1)
        elif metric == "mahalanobis" and cov is not None:
            # Mahalanobis distance (requires covariance)
            diff = c - mean  # [batch, d]
            cov_inv = torch.inverse(cov)
            distance = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return distance
    
    def distance_to_all_prototypes(
        self,
        c: torch.Tensor,
        metric: str = "cosine"
    ) -> torch.Tensor:
        """
        Compute distances to all prototypes
        
        Args:
            c: Consciousness state [batch, conscious_dim]
            metric: Distance metric
        
        Returns:
            distances: [batch, num_classes]
        """
        if metric == "cosine":
            # Batch cosine similarity
            c_norm = F.normalize(c, p=2, dim=1)  # [batch, d]
            means_norm = F.normalize(self.means, p=2, dim=1)  # [num_classes, d]
            similarity = torch.matmul(c_norm, means_norm.T)  # [batch, num_classes]
            distances = 1.0 - similarity
        elif metric == "euclidean":
            # Batch euclidean distance
            c_expanded = c.unsqueeze(1)  # [batch, 1, d]
            means_expanded = self.means.unsqueeze(0)  # [1, num_classes, d]
            distances = torch.norm(c_expanded - means_expanded, dim=2)  # [batch, num_classes]
        else:
            raise ValueError(f"Batch computation not supported for metric: {metric}")
        
        return distances


def test_prototype_manager():
    """Test prototype manager"""
    print("Testing PrototypeManager...")
    
    num_classes = 10
    conscious_dim = 16
    manager = PrototypeManager(num_classes, conscious_dim, device="cpu")
    
    # Simulate some consciousness states
    for cls in range(num_classes):
        # Generate samples around a random center
        center = torch.randn(conscious_dim)
        samples = center.unsqueeze(0) + torch.randn(20, conscious_dim) * 0.1
        labels = torch.full((20,), cls, dtype=torch.long)
        manager.update(samples, labels)
    
    print(f"Prototype means shape: {manager.means.shape}")
    print(f"Sample counts: {manager.counts}")
    
    # Test distance computation
    test_c = torch.randn(5, conscious_dim)
    dist = manager.distance_to_prototype(test_c, 0, "cosine")
    print(f"Distance to class 0: {dist}")
    
    all_dist = manager.distance_to_all_prototypes(test_c, "cosine")
    print(f"Distances to all classes shape: {all_dist.shape}")
    
    # Test save/load
    manager.save("/tmp", "test_arch", "test_dataset", "layer4")
    loaded_manager = PrototypeManager.load("/tmp", "test_arch", "test_dataset", "layer4", device="cpu")
    print(f"Loaded prototype means match: {torch.allclose(manager.means, loaded_manager.means)}")
    
    print("PrototypeManager test passed!")


if __name__ == "__main__":
    test_prototype_manager()


