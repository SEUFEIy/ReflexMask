"""
Risk Monitor for Conscious Defense
Computes risk score r based on distance to prototypes, entropy, and augmentation instability
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .config import ConsciousConfig, DEFAULT_CONFIG
from .prototypes import PrototypeManager


class RiskMonitor:
    """
    Monitor and compute risk scores for consciousness states.
    r = distance_to_prototype + alpha * entropy + beta * augmentation_instability
    """
    
    def __init__(
        self,
        prototype_manager: PrototypeManager,
        config: Optional[ConsciousConfig] = None,
        device: str = "cuda"
    ):
        """
        Args:
            prototype_manager: Prototype manager with class prototypes
            config: Configuration
            device: Device to run on
        """
        self.prototype_manager = prototype_manager
        self.config = config or DEFAULT_CONFIG
        self.device = device
        
        self.alpha = self.config.risk_alpha
        self.beta = self.config.risk_beta
        self.distance_metric = self.config.distance_metric
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction entropy (normalized)
        
        Args:
            logits: Model logits [batch, num_classes]
        
        Returns:
            entropy: [batch] normalized entropy (0 to 1)
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)  # [batch]
        
        # Normalize by max entropy (log(num_classes))
        max_entropy = torch.log(torch.tensor(logits.shape[1], dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def compute_margin(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction margin (difference between top-1 and top-2)
        
        Args:
            logits: Model logits [batch, num_classes]
        
        Returns:
            margin: [batch] margin values
        """
        top2 = torch.topk(logits, 2, dim=1)[0]  # [batch, 2]
        margin = top2[:, 0] - top2[:, 1]
        return margin
    
    def compute_augmentation_instability(
        self,
        images: torch.Tensor,
        model: torch.nn.Module,
        extractor: 'ConsciousStateExtractor',
        layer_name: str = "layer4",
        n_augmentations: int = 4
    ) -> torch.Tensor:
        """
        Compute instability under random augmentations (similar to consistency check)
        
        Args:
            images: Input images [batch, C, H, W]
            model: Neural network model
            extractor: Consciousness state extractor
            layer_name: Which layer to extract from
            n_augmentations: Number of random augmentations
        
        Returns:
            instability: [batch] instability scores
        """
        batch_size = images.shape[0]
        
        # Get original consciousness state
        with torch.no_grad():
            # Extract activation
            activations = {}
            def hook_fn(module, input, output):
                activations[layer_name] = output
            
            layer = dict(model.named_modules())[layer_name]
            handle = layer.register_forward_hook(hook_fn)
            
            _ = model(images)
            c_orig = extractor.extract(activations[layer_name])
            
            handle.remove()
        
        # Apply random augmentations and compute variance
        c_aug_list = []
        for _ in range(n_augmentations):
            # Random augmentation: add small Gaussian noise
            noise = torch.randn_like(images) * 0.02
            images_aug = torch.clamp(images + noise, 0, 1)
            
            with torch.no_grad():
                activations = {}
                handle = layer.register_forward_hook(hook_fn)
                
                _ = model(images_aug)
                c_aug = extractor.extract(activations[layer_name])
                
                handle.remove()
            
            c_aug_list.append(c_aug)
        
        # Stack augmented states
        c_aug_stack = torch.stack(c_aug_list, dim=1)  # [batch, n_aug, d]
        
        # Compute variance across augmentations
        c_mean = c_aug_stack.mean(dim=1)  # [batch, d]
        c_var = torch.var(c_aug_stack, dim=1).mean(dim=1)  # [batch]
        
        # Also compute distance to original
        dist_to_orig = torch.norm(c_orig - c_mean, dim=1)  # [batch]
        
        # Instability = variance + distance_to_orig
        instability = c_var + dist_to_orig
        
        return instability
    
    def compute_risk_score(
        self,
        c: torch.Tensor,
        logits: torch.Tensor,
        predicted_class: Optional[torch.Tensor] = None,
        augmentation_instability: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute overall risk score
        r = distance_to_prototype + alpha * entropy + beta * augmentation_instability
        
        Args:
            c: Consciousness state [batch, d]
            logits: Model logits [batch, num_classes]
            predicted_class: Predicted class indices [batch], if None use argmax
            augmentation_instability: Pre-computed instability [batch], if None set to 0
        
        Returns:
            risk_score: [batch] overall risk scores
            components: Dict with individual components
        """
        batch_size = c.shape[0]
        
        # Get predicted class if not provided
        if predicted_class is None:
            predicted_class = torch.argmax(logits, dim=1)
        
        # Compute distance to predicted class prototype
        distance = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            cls = predicted_class[i].item()
            distance[i] = self.prototype_manager.distance_to_prototype(
                c[i:i+1], cls, self.distance_metric
            )
        
        # Compute entropy
        entropy = self.compute_entropy(logits)
        
        # Augmentation instability
        if augmentation_instability is None:
            augmentation_instability = torch.zeros(batch_size, device=self.device)
        
        # Compute overall risk
        risk_score = distance + self.alpha * entropy + self.beta * augmentation_instability
        
        # Package components
        components = {
            'distance': distance,
            'entropy': entropy,
            'augmentation_instability': augmentation_instability,
            'margin': self.compute_margin(logits)
        }
        
        return risk_score, components
    
    def compute_risk_batch(
        self,
        c: torch.Tensor,
        logits: torch.Tensor,
        use_augmentation: bool = False,
        images: Optional[torch.Tensor] = None,
        model: Optional[torch.nn.Module] = None,
        extractor: Optional['ConsciousStateExtractor'] = None,
        layer_name: str = "layer4"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute risk scores for a batch (wrapper for convenience)
        
        Args:
            c: Consciousness states [batch, d]
            logits: Model logits [batch, num_classes]
            use_augmentation: Whether to compute augmentation instability
            images: Input images (required if use_augmentation)
            model: Model (required if use_augmentation)
            extractor: Extractor (required if use_augmentation)
            layer_name: Layer name
        
        Returns:
            risk_scores: [batch]
            components: Dict of risk components
        """
        aug_instability = None
        
        if use_augmentation:
            if images is None or model is None or extractor is None:
                raise ValueError("Must provide images, model, and extractor for augmentation instability")
            
            aug_instability = self.compute_augmentation_instability(
                images, model, extractor, layer_name
            )
        
        risk_scores, components = self.compute_risk_score(c, logits, augmentation_instability=aug_instability)
        
        return risk_scores, components
    
    def is_high_risk(
        self,
        risk_score: torch.Tensor,
        entropy: torch.Tensor,
        margin: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine if samples are high risk based on thresholds
        
        Args:
            risk_score: Risk scores [batch]
            entropy: Entropy values [batch]
            margin: Margin values [batch]
        
        Returns:
            is_high_risk: [batch] boolean mask
        """
        high_risk_mask = (
            (risk_score >= self.config.risk_threshold) |
            (entropy >= self.config.entropy_threshold) |
            (margin <= self.config.margin_threshold)
        )
        
        return high_risk_mask
    
    def get_risk_level(
        self,
        risk_score: float,
        entropy: float,
        margin: float
    ) -> str:
        """
        Get risk level string for a single sample
        
        Args:
            risk_score: Risk score
            entropy: Entropy
            margin: Margin
        
        Returns:
            risk_level: "low_risk", "medium_risk", "high_risk", or "critical_risk"
        """
        return self.config.get_risk_level(risk_score, entropy, margin)
    
    def get_nearest_prototypes(
        self,
        c: torch.Tensor,
        top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get k nearest prototypes for consciousness state
        
        Args:
            c: Consciousness state [batch, d]
            top_k: Number of nearest prototypes
        
        Returns:
            distances: [batch, top_k] distances to nearest prototypes
            indices: [batch, top_k] class indices of nearest prototypes
        """
        # Compute distances to all prototypes
        all_distances = self.prototype_manager.distance_to_all_prototypes(c, self.distance_metric)
        
        # Get top-k nearest
        distances, indices = torch.topk(all_distances, top_k, dim=1, largest=False)
        
        return distances, indices


def test_risk_monitor():
    """Test risk monitor"""
    print("Testing RiskMonitor...")
    
    from .prototypes import PrototypeManager
    
    # Create prototype manager
    num_classes = 10
    conscious_dim = 16
    proto_manager = PrototypeManager(num_classes, conscious_dim, device="cpu")
    
    # Initialize with some random prototypes
    for cls in range(num_classes):
        samples = torch.randn(20, conscious_dim)
        labels = torch.full((20,), cls, dtype=torch.long)
        proto_manager.update(samples, labels)
    
    # Create risk monitor
    monitor = RiskMonitor(proto_manager, device="cpu")
    
    # Test risk computation
    c = torch.randn(4, conscious_dim)
    logits = torch.randn(4, num_classes)
    
    risk_scores, components = monitor.compute_risk_score(c, logits)
    print(f"Risk scores shape: {risk_scores.shape}")
    print(f"Risk scores: {risk_scores}")
    print(f"Components: {list(components.keys())}")
    print(f"Entropy: {components['entropy']}")
    print(f"Margin: {components['margin']}")
    
    # Test high risk detection
    is_high = monitor.is_high_risk(
        risk_scores,
        components['entropy'],
        components['margin']
    )
    print(f"High risk mask: {is_high}")
    
    # Test nearest prototypes
    distances, indices = monitor.get_nearest_prototypes(c, top_k=3)
    print(f"Nearest prototype distances: {distances}")
    print(f"Nearest prototype indices: {indices}")
    
    print("RiskMonitor test passed!")


if __name__ == "__main__":
    test_risk_monitor()

