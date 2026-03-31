"""
ReflexMask: Training-free adversarial defense with introspection-detection-reconstruction loop.
"""

from .config import ConsciousConfig, DEFAULT_CONFIG
from .conscious_state import ConsciousStateExtractor
from .prototypes import PrototypeManager
from .monitor import RiskMonitor
from .mask_bank import MaskBank
from .controller import ConsciousController
from .wrapper import ConsciousDefenseWrapper, create_conscious_defense

__version__ = "1.0.0"
__all__ = [
    "ConsciousConfig",
    "DEFAULT_CONFIG",
    "ConsciousStateExtractor",
    "PrototypeManager",
    "RiskMonitor",
    "MaskBank",
    "ConsciousController",
    "ConsciousDefenseWrapper",
    "create_conscious_defense"
]


