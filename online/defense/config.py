"""
Configuration for ReflexMask Defense
All default parameters centralized here.
"""
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class ConsciousConfig:
    """Configuration for conscious defense mechanism"""
    
    # Conscious state extraction
    conscious_dim: int = 16  # Dimensionality of consciousness state c
    conscious_top_s: int = 8  # Number of top activations to keep
    conscious_seed: int = 42  # Random seed for reproducibility
    projection_type: str = "random_ortho"  # "random_ortho" or "hadamard"
    
    # Prototype parameters
    use_covariance: bool = False  # Whether to compute covariance (mean only by default)
    min_samples_per_class: int = 10  # Minimum samples needed per class
    
    # Risk monitoring
    risk_alpha: float = 0.3  # Weight for entropy term in risk score
    risk_beta: float = 0.2  # Weight for augmentation instability term
    distance_metric: str = "cosine"  # "cosine" or "euclidean"
    
    # Mask bank configuration
    k_list: List[int] = field(default_factory=lambda: [16, 32, 64, 128])  # Granularities
    mask_layers: List[str] = field(default_factory=lambda: ["layer4"])  # Which layers to mask
    mask_strategies: List[str] = field(default_factory=lambda: ["single_class", "top_m"])
    top_m_classes: int = 3  # For top-m hypothesis strategy
    soft_mask: bool = False  # Use soft (float) masks instead of binary
    
    # Controller parameters
    risk_threshold: float = 0.5  # Threshold to trigger conscious defense
    max_steps: int = 3  # Maximum iterative refinement steps
    entropy_threshold: float = 1.5  # High entropy threshold
    margin_threshold: float = 0.2  # Low margin threshold
    use_rs_on_high_risk: bool = True  # Enable randomized smoothing on high risk
    rs_sigma: float = 8.0 / 255.0  # RS noise level
    rs_nsmooth: int = 16  # RS samples
    
    # Action selection rules
    action_rules: dict = field(default_factory=lambda: {
        "low_risk": {"k_idx": 0, "strategy": "single_class", "layers": ["layer4"], "use_rs": False},
        "medium_risk": {"k_idx": 1, "strategy": "top_m", "layers": ["layer4"], "use_rs": False},
        "high_risk": {"k_idx": 2, "strategy": "top_m", "layers": ["layer3", "layer4"], "use_rs": True},
        "critical_risk": {"k_idx": 3, "strategy": "top_m", "layers": ["layer3", "layer4"], "use_rs": True}
    })
    
    # Paths
    prototype_dir: str = "saved_conscious_prototypes"
    projection_dir: str = "saved_conscious_projections"
    
    def get_risk_level(self, risk_score: float, entropy: float, margin: float) -> str:
        """Determine risk level based on multiple factors"""
        if risk_score < self.risk_threshold and entropy < self.entropy_threshold:
            return "low_risk"
        elif risk_score < self.risk_threshold * 1.5 and margin > self.margin_threshold:
            return "medium_risk"
        elif risk_score < self.risk_threshold * 2.0:
            return "high_risk"
        else:
            return "critical_risk"
    
    def get_action(self, risk_level: str) -> dict:
        """Get action parameters based on risk level"""
        return self.action_rules.get(risk_level, self.action_rules["medium_risk"])


# Default configuration instance
DEFAULT_CONFIG = ConsciousConfig()


