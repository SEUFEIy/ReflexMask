"""
Conscious State Extraction Module
Extracts low-dimensional consciousness state c from layer activations using fixed projections
c = normalize(top_s(P @ phi(h)))
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from scipy.linalg import hadamard
from .config import ConsciousConfig, DEFAULT_CONFIG


class ConsciousStateExtractor:
    """
    Extract consciousness state from neural activations using fixed random projections.
    No trainable parameters - purely deterministic given seed.
    """
    
    def __init__(
        self,
        layer_dim: int,
        config: Optional[ConsciousConfig] = None,
        device: str = "cuda"
    ):
        """
        Args:
            layer_dim: Dimension of input layer activations (e.g., 512 for ResNet18 layer4)
            config: Conscious configuration
            device: Device to run on
        """
        self.layer_dim = layer_dim
        self.config = config or DEFAULT_CONFIG
        self.device = device
        
        self.d = self.config.conscious_dim
        self.s = self.config.conscious_top_s
        self.seed = self.config.conscious_seed
        self.projection_type = self.config.projection_type
        
        # Initialize projection matrix P: [d, layer_dim]
        self.P = self._create_projection_matrix()
        self.P = torch.from_numpy(self.P).float().to(device)
        
    def _create_projection_matrix(self) -> np.ndarray:
        """
        Create fixed projection matrix (reproducible)
        
        Returns:
            P: [d, layer_dim] projection matrix
        """
        np.random.seed(self.seed)
        
        if self.projection_type == "random_ortho":
            # Random orthogonal projection via QR decomposition
            if self.d <= self.layer_dim:
                A = np.random.randn(self.layer_dim, self.d)
                P, _ = np.linalg.qr(A)
                P = P.T  # [d, layer_dim]
            else:
                # If d > layer_dim, use random Gaussian (normalized)
                P = np.random.randn(self.d, self.layer_dim)
                P = P / np.linalg.norm(P, axis=1, keepdims=True)
                
        elif self.projection_type == "hadamard":
            # Hadamard projection (only works for power-of-2 dimensions)
            n = 2 ** int(np.ceil(np.log2(max(self.d, self.layer_dim))))
            H = hadamard(n) / np.sqrt(n)
            
            # Take first d rows and layer_dim columns
            P = H[:self.d, :self.layer_dim]
            
        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")
        
        return P.astype(np.float32)
    
    def save_projection(self, save_path: str):
        """Save projection matrix for reproducibility"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, self.P.cpu().numpy())
        # Also save metadata
        meta_path = save_path.replace('.npy', '_meta.npz')
        np.savez(meta_path,
                 layer_dim=self.layer_dim,
                 d=self.d,
                 s=self.s,
                 seed=self.seed,
                 projection_type=self.projection_type)
    
    @classmethod
    def load_projection(cls, load_path: str, device: str = "cuda") -> 'ConsciousStateExtractor':
        """Load projection matrix from file"""
        # Try loading as .npz first (if saved with npz format)
        if load_path.endswith('.npz'):
            data = np.load(load_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                # If it's an npz file, try to get the array
                # Check if there's a .npy file with same base name
                npy_path = load_path + '.npy'
                if os.path.exists(npy_path):
                    P = np.load(npy_path)
                else:
                    # Otherwise just use the first array in npz
                    P = data[list(data.keys())[0]] if len(data.keys()) > 0 else None
                    if P is None:
                        raise ValueError(f"Could not find projection matrix in {load_path}")
            else:
                P = data
        else:
            P = np.load(load_path)
        
        meta_path = load_path.replace('.npz', '_meta.npz').replace('.npy', '_meta.npz')
        
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            layer_dim = int(meta['layer_dim'])
            config = ConsciousConfig(
                conscious_dim=int(meta['d']),
                conscious_top_s=int(meta['s']),
                conscious_seed=int(meta['seed']),
                projection_type=str(meta['projection_type'])
            )
        else:
            # Infer from shape
            layer_dim = P.shape[1]
            config = ConsciousConfig()
        
        extractor = cls(layer_dim, config, device)
        extractor.P = torch.from_numpy(P).float().to(device)
        return extractor
    
    def phi(self, h: torch.Tensor) -> torch.Tensor:
        """
        Nonlinear transformation phi(h)
        Default: ReLU (can be changed to other activations)
        
        Args:
            h: [batch, layer_dim] or [batch, layer_dim, H, W]
        
        Returns:
            phi_h: same shape as h
        """
        return F.relu(h)
    
    def top_s_selection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Keep only top-s activations, zero out others
        
        Args:
            x: [batch, d] projected features
        
        Returns:
            x_sparse: [batch, d] with only top-s non-zero
        """
        if self.s >= self.d:
            return x
        
        # Get top-s indices
        top_vals, top_ids = torch.topk(torch.abs(x), self.s, dim=1)
        
        # Create sparse tensor
        x_sparse = torch.zeros_like(x)
        x_sparse.scatter_(1, top_ids, x.gather(1, top_ids))
        
        return x_sparse
    
    def extract(self, h: torch.Tensor) -> torch.Tensor:
        """
        Extract consciousness state: c = normalize(top_s(P @ phi(h)))
        
        Args:
            h: Layer activations [batch, layer_dim] or [batch, layer_dim, H, W]
        
        Returns:
            c: Consciousness state [batch, d]
        """
        # Handle spatial activations (conv layers)
        if h.dim() == 4:
            # Global average pooling: [batch, layer_dim, H, W] -> [batch, layer_dim]
            h = F.adaptive_avg_pool2d(h, 1).squeeze(-1).squeeze(-1)
        
        # Apply nonlinear transformation
        phi_h = self.phi(h)  # [batch, layer_dim]
        
        # Project: [batch, layer_dim] @ [layer_dim, d]^T -> [batch, d]
        projected = torch.matmul(phi_h, self.P.T)  # [batch, d]
        
        # Top-s selection
        sparse = self.top_s_selection(projected)  # [batch, d]
        
        # L2 normalization
        c = F.normalize(sparse, p=2, dim=1)  # [batch, d]
        
        return c
    
    def extract_multi_layer(self, activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract consciousness states from multiple layers
        
        Args:
            activations: Dict mapping layer_name -> activation tensor
        
        Returns:
            states: Dict mapping layer_name -> consciousness state
        """
        states = {}
        for layer_name, h in activations.items():
            states[layer_name] = self.extract(h)
        return states
    
    def get_feature_importance(self, h: torch.Tensor) -> torch.Tensor:
        """
        Get importance scores of input features based on projection magnitudes
        
        Args:
            h: [batch, layer_dim]
        
        Returns:
            importance: [batch, layer_dim] importance scores
        """
        if h.dim() == 4:
            h = F.adaptive_avg_pool2d(h, 1).squeeze(-1).squeeze(-1)
        
        phi_h = self.phi(h)
        
        # Compute contribution to consciousness state
        # importance_i = sum_j |P_ji * phi(h_i)|
        importance = torch.abs(phi_h.unsqueeze(1) * self.P.unsqueeze(0))  # [batch, d, layer_dim]
        importance = importance.sum(dim=1)  # [batch, layer_dim]
        
        return importance


def test_extractor():
    """Test consciousness state extraction"""
    print("Testing ConsciousStateExtractor...")
    
    # Create extractor
    layer_dim = 512
    extractor = ConsciousStateExtractor(layer_dim, device="cpu")
    
    # Test single batch
    h = torch.randn(4, 512)
    c = extractor.extract(h)
    print(f"Input shape: {h.shape}, Output shape: {c.shape}")
    print(f"Consciousness state norm: {torch.norm(c, dim=1)}")
    print(f"Sparsity: {(c == 0).float().mean()}")
    
    # Test spatial input
    h_spatial = torch.randn(4, 512, 4, 4)
    c_spatial = extractor.extract(h_spatial)
    print(f"Spatial input shape: {h_spatial.shape}, Output shape: {c_spatial.shape}")
    
    # Test save/load
    save_path = "/tmp/test_projection.npy"
    extractor.save_projection(save_path)
    loaded_extractor = ConsciousStateExtractor.load_projection(save_path, device="cpu")
    c_loaded = loaded_extractor.extract(h)
    print(f"Projection preserved: {torch.allclose(c, c_loaded)}")
    
    print("ConsciousStateExtractor test passed!")


if __name__ == "__main__":
    test_extractor()


