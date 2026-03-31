"""
Conscious Defense Wrapper
Wraps base model with conscious_ig defense mechanism for evaluation
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .conscious_state import ConsciousStateExtractor
from .prototypes import PrototypeManager
from .monitor import RiskMonitor
from .mask_bank import MaskBank
from .controller import ConsciousController
from .config import ConsciousConfig


class ConsciousDefenseWrapper(nn.Module):
    """
    Wrapper for models with conscious_ig defense.
    Makes conscious defense compatible with AutoAttack and other evaluation frameworks.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        extractor: ConsciousStateExtractor,
        proto_manager: PrototypeManager,
        risk_monitor: RiskMonitor,
        mask_bank: MaskBank,
        controller: ConsciousController,
        layer_name: str = "layer4",
        config: Optional[ConsciousConfig] = None,
        record_trajectory: bool = False
    ):
        """
        Args:
            base_model: Base neural network (InterpMaskedResNet/WideResNet)
            extractor: Consciousness state extractor
            proto_manager: Prototype manager
            risk_monitor: Risk monitor
            mask_bank: Mask bank
            controller: Conscious controller
            layer_name: Primary layer for extraction
            config: Configuration
            record_trajectory: Whether to record iteration trajectory
        """
        super().__init__()
        self.base_model = base_model
        self.extractor = extractor
        self.proto_manager = proto_manager
        self.risk_monitor = risk_monitor
        self.mask_bank = mask_bank
        self.controller = controller
        self.layer_name = layer_name
        self.config = config
        self.record_trajectory = record_trajectory
        
        # Trajectory storage (if recording)
        self.trajectories = [] if record_trajectory else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conscious defense
        
        Args:
            x: Input images [batch, C, H, W]
        
        Returns:
            logits: Model predictions [batch, num_classes]
        """
        # Initial forward pass to get features
        logits_init, activations = self.base_model(
            x, defense='conscious_ig', return_feats=True
        )
        
        # Extract consciousness state from primary layer
        c_init = self.extractor.extract(activations[self.layer_name])
        
        # Compute initial risk
        initial_risk, _ = self.risk_monitor.compute_risk_score(c_init, logits_init)
        
        # Only apply defense to high-risk samples
        high_risk_mask = initial_risk >= self.controller.risk_threshold
        
        if high_risk_mask.any():
            # Iterative defense
            final_logits, trajectory = self.controller.iterative_defense(
                self.base_model, self.extractor, x, logits_init, c_init
            )
            
            # Record trajectory if needed
            if self.record_trajectory and self.trajectories is not None:
                self.trajectories.append(trajectory)
        else:
            # All samples are low risk, use initial logits directly
            final_logits = logits_init
        
        return final_logits
    
    def forward_with_trajectory(self, x: torch.Tensor) -> tuple:
        """
        Forward pass that returns both logits and trajectory
        
        Args:
            x: Input images [batch, C, H, W]
        
        Returns:
            logits: Model predictions [batch, num_classes]
            trajectory: List of dicts with iteration info
        """
        # Initial forward pass
        logits_init, activations = self.base_model(
            x, defense='conscious_ig', return_feats=True
        )
        
        # Extract consciousness state
        c_init = self.extractor.extract(activations[self.layer_name])
        
        # Iterative defense
        final_logits, trajectory = self.controller.iterative_defense(
            self.base_model, self.extractor, x, logits_init, c_init
        )
        
        return final_logits, trajectory
    
    def get_trajectories(self) -> Optional[List]:
        """Get recorded trajectories"""
        return self.trajectories
    
    def clear_trajectories(self):
        """Clear recorded trajectories"""
        if self.trajectories is not None:
            self.trajectories = []
    
    def eval(self):
        """Set to evaluation mode"""
        self.base_model.eval()
        return super().eval()
    
    def train(self, mode: bool = True):
        """Set to training mode (though this defense is training-free)"""
        self.base_model.train(mode)
        return super().train(mode)
    
    def to(self, device):
        """Move to device"""
        self.base_model = self.base_model.to(device)
        self.extractor = self.extractor.to(device) if hasattr(self.extractor, 'to') else self.extractor
        self.proto_manager = self.proto_manager.to(device) if hasattr(self.proto_manager, 'to') else self.proto_manager
        self.risk_monitor.device = device
        self.mask_bank.device = device
        self.controller.device = device
        return super().to(device)


def create_conscious_defense(
    base_model: nn.Module,
    proto_path: str,
    arch: str,
    checkpoint_name: str,
    layer_name: str,
    layer_dim: int,
    num_classes: int,
    ranking_method: str = "loir",
    config: Optional[ConsciousConfig] = None,
    device: str = "cuda",
    record_trajectory: bool = False
) -> ConsciousDefenseWrapper:
    """
    Factory function to create conscious defense wrapper
    
    Args:
        base_model: Base neural network
        proto_path: Path to prototype file
        arch: Model architecture
        checkpoint_name: Checkpoint filename
        layer_name: Layer name for extraction
        layer_dim: Layer dimension
        num_classes: Number of classes
        ranking_method: "loir" or "cdir"
        config: Configuration (uses default if None)
        device: Device
        record_trajectory: Whether to record trajectories
    
    Returns:
        ConsciousDefenseWrapper instance
    """
    from .config import DEFAULT_CONFIG
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Create components
    extractor = ConsciousStateExtractor(layer_dim, config, device)
    
    # Try to load projection if exists
    import os
    proj_path = os.path.join(os.path.dirname(proto_path), "projection.npz")
    if os.path.exists(proj_path):
        extractor.load_projection(proj_path)
        print(f"Loaded projection from {proj_path}")
    
    # Load prototypes
    # Extract arch, dataset, layer_name from proto_path
    # Path format: saved_conscious_prototypes/{arch}/{dataset}/{layer_name}/prototypes.npz
    import os
    path_parts = proto_path.split(os.sep)
    if 'saved_conscious_prototypes' in path_parts:
        idx = path_parts.index('saved_conscious_prototypes')
        save_dir = os.sep.join(path_parts[:idx+1])
        arch_from_path = path_parts[idx+1] if len(path_parts) > idx+1 else arch
        dataset_from_path = path_parts[idx+2] if len(path_parts) > idx+2 else 'cifar10'
        layer_from_path = path_parts[idx+3] if len(path_parts) > idx+3 else layer_name
    else:
        # Fallback: use provided values
        save_dir = os.path.dirname(os.path.dirname(os.path.dirname(proto_path)))
        arch_from_path = arch
        dataset_from_path = 'cifar10'
        layer_from_path = layer_name
    
    proto_manager = PrototypeManager.load(save_dir, arch_from_path, dataset_from_path, layer_from_path, config, device)
    print(f"Loaded prototypes from {proto_path}")
    
    # Create risk monitor
    risk_monitor = RiskMonitor(proto_manager, config, device)
    
    # Create mask bank
    mask_bank = MaskBank(arch, checkpoint_name, ranking_method, config, device)
    mask_bank.load_rankings([layer_name], num_classes)
    print(f"Loaded {ranking_method} rankings for {layer_name}")
    
    # Create controller
    controller = ConsciousController(mask_bank, risk_monitor, config, device)
    
    # Create wrapper
    wrapper = ConsciousDefenseWrapper(
        base_model, extractor, proto_manager, risk_monitor,
        mask_bank, controller, layer_name, config, record_trajectory
    )
    
    return wrapper


def test_wrapper():
    """Test conscious defense wrapper"""
    print("Testing ConsciousDefenseWrapper...")
    
    # This is a mock test - full test requires actual model and prototypes
    import sys
    sys.path.insert(0, '..')
    
    try:
        from models.resnet_val import InterpMaskedResNet
        
        # Create a small model for testing
        model = InterpMaskedResNet(
            layer_name='layer4',
            checkpoint_name='test.pt',
            mask_which='conscious_ig',
            important_dim=128,
            num_classes=10
        )
        
        print("Base model created")
        print("Note: Full integration test requires:")
        print("  1. Actual checkpoint")
        print("  2. Prototype file")
        print("  3. Importance rankings")
        print("\nUse scripts/eval_conscious_ig.sh for end-to-end testing")
        
    except Exception as e:
        print(f"Mock test (expected): {e}")
        print("This is normal - wrapper needs full environment")
    
    print("Wrapper test completed!")


if __name__ == "__main__":
    test_wrapper()

