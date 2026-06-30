"""cv_flow.dam — Direct Access Memory layer (zero-copy IPC)."""
from cv_flow.dam.bus import PortBus

try:
    from cv_flow.dam.round_robin import RoundRobinBus
except ImportError:
    RoundRobinBus = None  # type: ignore

try:
    from cv_flow.dam.merge import MergeBus
except ImportError:
    MergeBus = None  # type: ignore

try:
    from cv_flow.dam.cuda_bus import CudaPortBus
    _CUDA_AVAILABLE = True
except ImportError:
    CudaPortBus = None  # type: ignore
    _CUDA_AVAILABLE = False

__all__ = ["PortBus", "RoundRobinBus", "MergeBus", "CudaPortBus", "_CUDA_AVAILABLE"]
