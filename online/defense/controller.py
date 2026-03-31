"""
Conscious Controller
Rule-based controller (no learning) that selects actions based on risk scores
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .config import ConsciousConfig, DEFAULT_CONFIG
from .monitor import RiskMonitor
from .mask_bank import MaskBank


class ConsciousController:
    """
    Rule-based controller for conscious defense.
    Selects actions (k, strategy, layers, RS) based on risk level.
    No trainable parameters - purely rule-based.
    """
    
    def __init__(
        self,
        mask_bank: MaskBank,
        risk_monitor: RiskMonitor,
        config: Optional[ConsciousConfig] = None,
        device: str = "cuda"
    ):
        """
        Args:
            mask_bank: Mask bank for generating masks
            risk_monitor: Risk monitor for scoring
            config: Configuration
            device: Device
        """
        self.mask_bank = mask_bank
        self.risk_monitor = risk_monitor
        self.config = config or DEFAULT_CONFIG
        self.device = device
        
        self.k_list = self.config.k_list
        self.max_steps = self.config.max_steps
        self.risk_threshold = self.config.risk_threshold
    
    def select_action(
        self,
        risk_score: float,
        entropy: float,
        margin: float,
        step: int = 0
    ) -> Dict[str, any]:
        """
        Select action based on current risk level
        
        Args:
            risk_score: Current risk score
            entropy: Prediction entropy
            margin: Prediction margin
            step: Current step in iterative loop
        
        Returns:
            action: Dict with keys:
                - k_idx: Index in k_list
                - k: Actual k value
                - strategy: "single_class", "top_m", or "mixed"
                - layers: List of layer names to apply mask
                - use_rs: Whether to use randomized smoothing
                - top_m: Number of classes for top_m strategy
        """
        # Determine risk level
        risk_level = self.risk_monitor.get_risk_level(risk_score, entropy, margin)
        
        # Get base action from config
        action = self.config.get_action(risk_level).copy()
        
        # Get k value from index
        k_idx = action['k_idx']
        if k_idx >= len(self.k_list):
            k_idx = len(self.k_list) - 1
        action['k'] = self.k_list[k_idx]
        
        # Add top_m if not present
        if 'top_m' not in action:
            action['top_m'] = self.config.top_m_classes
        
        # Progressive escalation based on step
        if step > 0:
            # Increase k granularity
            new_k_idx = min(k_idx + step, len(self.k_list) - 1)
            action['k'] = self.k_list[new_k_idx]
            
            # More aggressive strategies in later steps
            if step >= 2 and action['strategy'] == 'single_class':
                action['strategy'] = 'top_m'
            
            # Enable RS in later steps if not already
            if step >= 2 and self.config.use_rs_on_high_risk:
                action['use_rs'] = True
        
        return action
    
    def apply_mask_to_activation(
        self,
        activation: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mask to activation tensor
        
        Args:
            activation: [batch, channels] or [batch, channels, H, W]
            mask: [batch, channels] or [channels]
        
        Returns:
            masked_activation: Same shape as activation
        """
        if activation.dim() == 4:
            # Spatial activation: [batch, C, H, W]
            if mask.dim() == 1:
                # [C] -> [1, C, 1, 1]
                mask = mask.view(1, -1, 1, 1)
            elif mask.dim() == 2:
                # [batch, C] -> [batch, C, 1, 1]
                mask = mask.unsqueeze(-1).unsqueeze(-1)
        else:
            # Flattened activation: [batch, C]
            if mask.dim() == 1:
                # [C] -> [1, C]
                mask = mask.unsqueeze(0)
        
        return activation * mask
    
    def execute_action(
        self,
        action: Dict[str, any],
        model: torch.nn.Module,
        extractor: 'ConsciousStateExtractor',
        images: torch.Tensor,
        current_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Execute selected action (generate masks and forward pass)
        
        Args:
            action: Action dictionary from select_action
            model: Neural network model
            extractor: Consciousness state extractor
            images: Input images [batch, C, H, W]
            current_logits: Current model logits [batch, num_classes]
        
        Returns:
            new_logits: [batch, num_classes] after applying masks
            new_c: [batch, d] new consciousness state
            new_activations: Dict of layer activations
        """
        k = action['k']
        strategy = action['strategy']
        layer_names = action['layers']
        use_rs = action.get('use_rs', False)
        top_m = action.get('top_m', 3)
        
        batch_size = images.shape[0]
        
        # Generate masks for specified layers
        masks = {}
        for layer_name in layer_names:
            if layer_name in self.mask_bank.layer_dims:
                # Generate batch masks
                batch_masks = self.mask_bank.generate_batch_masks(
                    layer_name, current_logits, k, strategy, binary=not self.config.soft_mask, top_m=top_m
                )
                masks[layer_name] = batch_masks
        
        # Forward pass with masking
        if use_rs:
            # Randomized smoothing
            new_logits, new_activations = self._forward_with_rs_and_masks(
                model, images, masks
            )
        else:
            # Try using external_masks parameter first (if model supports it)
            try:
                new_logits, new_activations = model(
                    images, defense='conscious_ig', return_feats=True, external_masks=masks
                )
            except TypeError:
                # Fallback to hooks-based approach
                new_logits, new_activations = self._forward_with_masks(
                    model, images, masks
                )
        
        # Extract new consciousness state
        # Use the primary layer (first in layer_names)
        primary_layer = layer_names[0] if layer_names else "layer4"
        
        def get_best_layer(activations_dict, target_layer):
            """Get the best layer from activations, avoiding _spatial versions"""
            if target_layer in activations_dict:
                return target_layer
            # Avoid spatial versions
            non_spatial_layers = [k for k in activations_dict.keys() if not k.endswith('_spatial')]
            if non_spatial_layers:
                # Prefer target layer if available
                if target_layer in non_spatial_layers:
                    return target_layer
                # Otherwise, prefer layers with matching name prefix
                matching = [k for k in non_spatial_layers if target_layer in k or k in target_layer]
                if matching:
                    return matching[0]
                # Last resort: use last non-spatial layer (usually the deepest)
                return non_spatial_layers[-1]
            # Last resort: use any layer
            return list(activations_dict.keys())[0] if activations_dict else None
        
        if primary_layer in new_activations:
            new_c = extractor.extract(new_activations[primary_layer])
        elif len(new_activations) > 0:
            # Fallback: use any available layer (avoiding spatial versions)
            avail_layer = get_best_layer(new_activations, primary_layer)
            if avail_layer:
                new_c = extractor.extract(new_activations[avail_layer])
            else:
                raise RuntimeError("No suitable activations found")
        else:
            # No activations available, re-do forward pass
            print(f"Warning: No activations returned, re-doing forward pass")
            new_logits, new_activations = model(images, defense='conscious_ig', return_feats=True)
            avail_layer = get_best_layer(new_activations, primary_layer)
            if avail_layer:
                new_c = extractor.extract(new_activations[avail_layer])
            else:
                raise RuntimeError("No suitable activations found after re-forward")
        
        return new_logits, new_c, new_activations
    
    def _forward_with_masks(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with masks applied via hooks
        
        Args:
            model: Model
            images: Input images
            masks: Dict {layer_name: mask tensor}
        
        Returns:
            logits: Model output
            activations: Dict of layer activations (after masking)
        """
        activations = {}
        handles = []
        
        def make_hook(layer_name, mask):
            def hook(module, input, output):
                # Apply mask
                masked_output = self.apply_mask_to_activation(output, mask)
                activations[layer_name] = masked_output
                return masked_output
            return hook
        
        # Register hooks
        for layer_name, mask in masks.items():
            try:
                layer = dict(model.named_modules())[layer_name]
                handle = layer.register_forward_hook(make_hook(layer_name, mask))
                handles.append(handle)
            except KeyError:
                print(f"Warning: Layer {layer_name} not found in model")
        
        # Forward pass - need to request features
        with torch.no_grad():
            try:
                # Try with return_feats parameter
                result = model(images, defense='conscious_ig', return_feats=True)
                if isinstance(result, tuple):
                    logits, model_activations = result
                    # Merge model_activations with our hook-collected activations
                    activations.update(model_activations)
                else:
                    logits = result
            except TypeError:
                # Fallback to plain forward
                logits = model(images)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return logits, activations
    
    def _forward_with_rs_and_masks(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with randomized smoothing and masks
        
        Args:
            model: Model
            images: Input images
            masks: Dict of masks
        
        Returns:
            smoothed_logits: Averaged logits
            activations: Dict of layer activations (from last sample)
        """
        sigma = self.config.rs_sigma
        n_smooth = self.config.rs_nsmooth
        
        batch_size = images.shape[0]
        
        # Collect logits from all noisy samples
        all_logits = []
        activations = None
        
        for i in range(n_smooth):
            # Add Gaussian noise
            noise = torch.randn_like(images) * sigma
            noisy_images = torch.clamp(images + noise, 0, 1)
            
            # Forward with masks
            logits, acts = self._forward_with_masks(model, noisy_images, masks)
            all_logits.append(logits)
            
            # Keep last activations
            if i == n_smooth - 1:
                activations = acts
        
        # Average logits
        smoothed_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        
        return smoothed_logits, activations
    
    def iterative_defense(
        self,
        model: torch.nn.Module,
        extractor: 'ConsciousStateExtractor',
        images: torch.Tensor,
        initial_logits: torch.Tensor,
        initial_c: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict[str, any]]]:
        """
        Iterative conscious defense loop (up to max_steps)
        
        Args:
            model: Model
            extractor: Extractor
            images: Input images [batch, C, H, W]
            initial_logits: Initial logits [batch, num_classes]
            initial_c: Initial consciousness state [batch, d]
        
        Returns:
            final_logits: [batch, num_classes]
            trajectory: List of dicts with step information
        """
        current_logits = initial_logits
        current_c = initial_c
        
        trajectory = []
        
        # Compute initial risk
        initial_risk, components = self.risk_monitor.compute_risk_score(current_c, current_logits)
        
        trajectory.append({
            'step': 0,
            'risk_score': initial_risk.cpu(),
            'entropy': components['entropy'].cpu(),
            'margin': components['margin'].cpu(),
            'logits': current_logits.cpu()
        })
        
        # Check if ANY sample needs defense
        needs_defense = (initial_risk >= self.risk_threshold).any()
        
        if not needs_defense:
            # All samples are low risk, return initial predictions
            return current_logits, trajectory
        
        # Iterative refinement
        for step in range(1, self.max_steps + 1):
            # Compute current risk for all samples
            risk_scores, components = self.risk_monitor.compute_risk_score(current_c, current_logits)
            
            # Check convergence: if all samples below threshold, stop
            if (risk_scores < self.risk_threshold).all():
                break
            
            # Select action based on average risk
            avg_risk = risk_scores.mean().item()
            avg_entropy = components['entropy'].mean().item()
            avg_margin = components['margin'].mean().item()
            
            action = self.select_action(avg_risk, avg_entropy, avg_margin, step)
            
            # Execute action
            new_logits, new_c, new_activations = self.execute_action(
                action, model, extractor, images, current_logits
            )
            
            # Update state
            current_logits = new_logits
            current_c = new_c
            
            # Record trajectory
            new_risk, new_components = self.risk_monitor.compute_risk_score(current_c, current_logits)
            trajectory.append({
                'step': step,
                'risk_score': new_risk.cpu(),
                'entropy': new_components['entropy'].cpu(),
                'margin': new_components['margin'].cpu(),
                'action': action,
                'logits': current_logits.cpu()
            })
        
        return current_logits, trajectory


def test_controller():
    """Test conscious controller"""
    print("Testing ConsciousController...")
    
    from .prototypes import PrototypeManager
    from .conscious_state import ConsciousStateExtractor
    
    # Create components
    num_classes = 10
    conscious_dim = 16
    layer_dim = 512
    
    proto_manager = PrototypeManager(num_classes, conscious_dim, device="cpu")
    for cls in range(num_classes):
        samples = torch.randn(20, conscious_dim)
        labels = torch.full((20,), cls, dtype=torch.long)
        proto_manager.update(samples, labels)
    
    risk_monitor = RiskMonitor(proto_manager, device="cpu")
    
    # Create dummy mask bank
    # (In practice, would load real rankings)
    config = ConsciousConfig()
    config.k_list = [16, 32, 64, 128]
    
    # This is a simplified test - full test would require actual rankings
    print("Controller action selection test:")
    
    # Test action selection at different risk levels
    test_cases = [
        (0.3, 1.0, 0.5, "Low risk"),
        (0.6, 1.8, 0.3, "Medium risk"),
        (1.0, 2.0, 0.1, "High risk"),
        (1.5, 2.5, 0.05, "Critical risk")
    ]
    
    # Create a mock controller for action selection testing
    class MockMaskBank:
        def __init__(self):
            self.layer_dims = {"layer4": 512}
    
    mock_bank = MockMaskBank()
    controller = ConsciousController(mock_bank, risk_monitor, config, device="cpu")
    
    for risk, entropy, margin, desc in test_cases:
        action = controller.select_action(risk, entropy, margin, step=0)
        print(f"{desc}: k={action['k']}, strategy={action['strategy']}, layers={action['layers']}, use_rs={action.get('use_rs', False)}")
    
    # Test progressive escalation
    print("\nProgressive escalation test:")
    for step in range(3):
        action = controller.select_action(0.8, 1.5, 0.2, step=step)
        print(f"Step {step}: k={action['k']}, strategy={action['strategy']}, use_rs={action.get('use_rs', False)}")
    
    print("ConsciousController test passed!")


if __name__ == "__main__":
    test_controller()

