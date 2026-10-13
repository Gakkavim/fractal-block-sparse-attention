"""
Fractal Block-Sparse Attention
DOI: 10.5281/zenodo.23351708
"""

from .layout import FractalBlockLayoutGenerator
from .ops_pytorch import FractalBlockSparseAttention

__all__ = [
    "FractalBlockLayoutGenerator",
    "FractalBlockSparseAttention",
]

try:
    import torch
    if torch.cuda.is_available():
        from .ops_triton import fractal_triton_attention
        __all__.append("fractal_triton_attention")
except Exception:
    pass
