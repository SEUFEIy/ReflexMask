"""
Mask Bank for Conscious Defense
Generates and manages neuron masks from importance rankings with multiple granularities
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from .config import ConsciousConfig, DEFAULT_CONFIG


class MaskBank:
    """
    Manage a bank of neuron masks with different granularities and strategies.
    Loads from existing importance rankings (loir/cdir) and generates masks on-demand.
    """
    
    def __init__(
        self,
        arch: str,
        checkpoint_name: str,
        ranking_method: str,  # "loir" or "cdir"
        config: Optional[ConsciousConfig] = None,
        device: str = "cuda"
    ):
        """
        Args:
            arch: Model architecture (e.g., "rn18_val")
            checkpoint_name: Checkpoint filename
            ranking_method: "loir" or "cdir"
            config: Configuration
            device: Device
        """
        self.arch = arch
        self.checkpoint_name = checkpoint_name
        self.ranking_method = ranking_method
        self.config = config or DEFAULT_CONFIG
        self.device = device
        
        # Extract model name from checkpoint
        self.model_name = os.path.splitext(os.path.basename(checkpoint_name))[0]
        
        # Load importance rankings
        self.rankings = {}  # {layer_name: rankings}
        self.layer_dims = {}  # {layer_name: dimension}
        
        # Mask cache
        self._mask_cache = {}
    
    def load_rankings(self, layer_names: List[str], num_classes: int):
        """
        Load neuron importance rankings from disk
        
        Args:
            layer_names: List of layer names to load (e.g., ["layer3", "layer4"])
            num_classes: Number of classes
        """
        if self.ranking_method == "loir":
            base_dir = "saved_loir_rankings"
        elif self.ranking_method == "cdir":
            base_dir = "saved_cdir_rankings"
        else:
            raise ValueError(f"Unknown ranking method: {self.ranking_method}")
        
        for layer_name in layer_names:
            ranking_dir = os.path.join(base_dir, self.model_name, layer_name)
            
            if not os.path.exists(ranking_dir):
                print(f"Warning: Rankings not found at {ranking_dir}")
                continue
            
            # Load importance scores for each neuron
            # Each unit{k}.npy contains scores for all classes: [num_classes]
            unit_files = sorted([f for f in os.listdir(ranking_dir) if f.startswith('unit') and f.endswith('.npy')])
            
            if len(unit_files) == 0:
                print(f"Warning: No ranking files found in {ranking_dir}")
                continue
            
            layer_dim = len(unit_files)
            self.layer_dims[layer_name] = layer_dim
            
            # Load all rankings: [layer_dim, num_classes]
            rankings = np.zeros((layer_dim, num_classes))
            for i, unit_file in enumerate(unit_files):
                unit_idx = int(unit_file.replace('unit', '').replace('.npy', ''))
                rankings[unit_idx] = np.load(os.path.join(ranking_dir, unit_file))
            
            # Convert to torch tensor
            self.rankings[layer_name] = torch.from_numpy(rankings).float().to(self.device)
            
            print(f"Loaded rankings for {layer_name}: shape {self.rankings[layer_name].shape}")
    
    def get_top_k_neurons(
        self,
        layer_name: str,
        class_idx: int,
        k: int,
        descending: bool = True
    ) -> torch.Tensor:
        """
        Get top-k important neurons for a specific class
        
        Args:
            layer_name: Layer name
            class_idx: Class index
            k: Number of neurons to select
            descending: If True, higher scores = more important
        
        Returns:
            indices: [k] neuron indices
        """
        if layer_name not in self.rankings:
            raise ValueError(f"Layer {layer_name} not loaded")
        
        # Get scores for this class: [layer_dim]
        scores = self.rankings[layer_name][:, class_idx]
        
        # Get top-k indices
        _, indices = torch.topk(scores, k, largest=descending)
        
        return indices
    
    def generate_single_class_mask(
        self,
        layer_name: str,
        class_idx: int,
        k: int,
        binary: bool = True
    ) -> torch.Tensor:
        """
        Generate mask for a single class (keeps top-k, zeros others)
        
        Args:
            layer_name: Layer name
            class_idx: Class index
            k: Number of neurons to keep
            binary: If True, binary mask (0/1), else soft mask
        
        Returns:
            mask: [layer_dim] mask tensor
        """
        layer_dim = self.layer_dims[layer_name]
        
        # Get top-k neurons
        top_k_indices = self.get_top_k_neurons(layer_name, class_idx, k, descending=True)
        
        if binary:
            # Binary mask
            mask = torch.zeros(layer_dim, device=self.device)
            mask[top_k_indices] = 1.0
        else:
            # Soft mask based on scores
            scores = self.rankings[layer_name][:, class_idx]
            # Normalize scores to [0, 1]
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            # Zero out non-top-k
            mask = torch.zeros(layer_dim, device=self.device)
            mask[top_k_indices] = scores_norm[top_k_indices]
        
        return mask
    
    def generate_top_m_union_mask(
        self,
        layer_name: str,
        class_probabilities: torch.Tensor,
        k: int,
        top_m: int = 3,
        binary: bool = True
    ) -> torch.Tensor:
        """
        Generate mask as union of top-m predicted classes
        
        Args:
            layer_name: Layer name
            class_probabilities: [num_classes] probability distribution
            k: Number of neurons per class
            top_m: Number of top classes to consider
            binary: Binary or soft mask
        
        Returns:
            mask: [layer_dim] mask tensor
        """
        layer_dim = self.layer_dims[layer_name]
        
        # Get top-m classes
        top_probs, top_classes = torch.topk(class_probabilities, top_m)
        
        # Initialize mask
        if binary:
            mask = torch.zeros(layer_dim, device=self.device)
        else:
            mask = torch.zeros(layer_dim, device=self.device)
        
        # Union of top-k neurons from each top-m class
        for i, cls in enumerate(top_classes):
            cls_mask = self.generate_single_class_mask(layer_name, cls.item(), k, binary=binary)
            
            if binary:
                mask = torch.maximum(mask, cls_mask)
            else:
                # Weight by class probability
                mask = mask + top_probs[i] * cls_mask
        
        if not binary:
            # Normalize
            mask = mask / mask.max().clamp(min=1e-8)
        
        return mask
    
    def generate_mixed_mask(
        self,
        layer_name: str,
        class_probabilities: torch.Tensor,
        k: int,
        binary: bool = True
    ) -> torch.Tensor:
        """
        Generate mask weighted by class probabilities (soft mixture)
        
        Args:
            layer_name: Layer name
            class_probabilities: [num_classes] probability distribution
            k: Number of neurons to keep per class
            binary: Binary or soft mask
        
        Returns:
            mask: [layer_dim] mask tensor
        """
        layer_dim = self.layer_dims[layer_name]
        num_classes = class_probabilities.shape[0]
        
        # Generate mask for each class
        class_masks = []
        for cls in range(num_classes):
            cls_mask = self.generate_single_class_mask(layer_name, cls, k, binary=False)
            class_masks.append(cls_mask)
        
        class_masks = torch.stack(class_masks, dim=0)  # [num_classes, layer_dim]
        
        # Weighted mixture
        mask = torch.matmul(class_probabilities, class_masks)  # [layer_dim]
        
        if binary:
            # Keep top-k of the mixed mask
            _, top_indices = torch.topk(mask, k)
            binary_mask = torch.zeros(layer_dim, device=self.device)
            binary_mask[top_indices] = 1.0
            return binary_mask
        else:
            # Normalize
            mask = mask / mask.max().clamp(min=1e-8)
            return mask
    
    def generate_mask(
        self,
        layer_name: str,
        logits: torch.Tensor,
        k: int,
        strategy: str = "single_class",
        binary: bool = True,
        top_m: int = 3
    ) -> torch.Tensor:
        """
        Generate mask with specified strategy
        
        Args:
            layer_name: Layer name
            logits: Model logits [num_classes] for single sample
            k: Number of neurons to keep
            strategy: "single_class", "top_m", or "mixed"
            binary: Binary or soft mask
            top_m: For top_m strategy
        
        Returns:
            mask: [layer_dim] mask tensor
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=0)
        
        cache_key = f"{layer_name}_{k}_{strategy}_{binary}_{top_m}_{logits.argmax().item()}"
        
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        
        if strategy == "single_class":
            predicted_class = torch.argmax(logits)
            mask = self.generate_single_class_mask(layer_name, predicted_class.item(), k, binary)
        elif strategy == "top_m":
            mask = self.generate_top_m_union_mask(layer_name, probs, k, top_m, binary)
        elif strategy == "mixed":
            mask = self.generate_mixed_mask(layer_name, probs, k, binary)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Cache mask
        self._mask_cache[cache_key] = mask
        
        return mask
    
    def generate_batch_masks(
        self,
        layer_name: str,
        logits_batch: torch.Tensor,
        k: int,
        strategy: str = "single_class",
        binary: bool = True,
        top_m: int = 3
    ) -> torch.Tensor:
        """
        Generate masks for a batch
        
        Args:
            layer_name: Layer name
            logits_batch: [batch, num_classes]
            k: Number of neurons
            strategy: Masking strategy
            binary: Binary or soft
            top_m: For top_m strategy
        
        Returns:
            masks: [batch, layer_dim]
        """
        batch_size = logits_batch.shape[0]
        layer_dim = self.layer_dims[layer_name]
        
        masks = torch.zeros(batch_size, layer_dim, device=self.device)
        
        for i in range(batch_size):
            masks[i] = self.generate_mask(
                layer_name, logits_batch[i], k, strategy, binary, top_m
            )
        
        return masks
    
    def generate_multi_layer_masks(
        self,
        layer_names: List[str],
        logits: torch.Tensor,
        k: int,
        strategy: str = "single_class",
        binary: bool = True,
        top_m: int = 3
    ) -> Dict[str, torch.Tensor]:
        """
        Generate masks for multiple layers
        
        Args:
            layer_names: List of layer names
            logits: Model logits [num_classes]
            k: Number of neurons
            strategy: Masking strategy
            binary: Binary or soft
            top_m: For top_m strategy
        
        Returns:
            masks: Dict {layer_name: mask tensor}
        """
        masks = {}
        for layer_name in layer_names:
            if layer_name in self.rankings:
                masks[layer_name] = self.generate_mask(
                    layer_name, logits, k, strategy, binary, top_m
                )
        return masks
    
    def get_mask_sparsity(self, mask: torch.Tensor) -> float:
        """Get sparsity of mask (fraction of zeros)"""
        return (mask == 0).float().mean().item()
    
    def clear_cache(self):
        """Clear mask cache"""
        self._mask_cache = {}


def test_mask_bank():
    """Test mask bank"""
    print("Testing MaskBank...")
    
    # This test assumes rankings exist - create dummy rankings for testing
    test_dir = "/tmp/test_rankings/saved_loir_rankings/test_model"
    os.makedirs(os.path.join(test_dir, "layer4"), exist_ok=True)
    
    # Create dummy rankings
    layer_dim = 512
    num_classes = 10
    for k in range(layer_dim):
        scores = np.random.randn(num_classes).astype(np.float32)
        np.save(os.path.join(test_dir, "layer4", f"unit{k}.npy"), scores)
    
    # Create mask bank
    bank = MaskBank("test_arch", "test_model.pt", "loir", device="cpu")
    
    # Change base directory for testing
    import sys
    original_dir = os.getcwd()
    os.chdir("/tmp/test_rankings")
    
    try:
        bank.load_rankings(["layer4"], num_classes)
        
        # Test single class mask
        mask = bank.generate_single_class_mask("layer4", 0, k=64, binary=True)
        print(f"Single class mask shape: {mask.shape}")
        print(f"Mask sparsity: {bank.get_mask_sparsity(mask):.2f}")
        print(f"Number of non-zero: {(mask != 0).sum()}")
        
        # Test batch masks
        logits_batch = torch.randn(4, num_classes)
        masks_batch = bank.generate_batch_masks("layer4", logits_batch, k=64, strategy="single_class")
        print(f"Batch masks shape: {masks_batch.shape}")
        
        # Test top-m strategy
        logits = torch.randn(num_classes)
        mask_topm = bank.generate_mask("layer4", logits, k=64, strategy="top_m", top_m=3)
        print(f"Top-m mask sparsity: {bank.get_mask_sparsity(mask_topm):.2f}")
        
        print("MaskBank test passed!")
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    test_mask_bank()

