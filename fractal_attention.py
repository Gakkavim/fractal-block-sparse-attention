"""
Sub-Quadratic Block-Sparse Attention via Odd Binomial Fractal Tree Topologies
Reference Implementation: Layout Generator and Multi-Head Attention Layer
Paper DOI: 10.5281/zenodo.23351708
"""

import math
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FractalBlockLayoutGenerator:
    """
    Compiles token-level fractal odd binomial tree connections into
    hardware-aligned (M, M) block masks and compact Lookup Tables (LUT).
    """
    def __init__(self, block_size: int = 32, local_window_blocks: int = 1) -> None:
        self.block_size = block_size
        self.local_window_blocks = local_window_blocks

    def generate(
        self,
        seq_len: int,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert seq_len > 0, "Sequence length must be positive."
        padded_seq_len = ((seq_len + self.block_size - 1) // self.block_size) * self.block_size
        M = padded_seq_len // self.block_size

        # 1. Base block adjacency mask
        mask = np.zeros((M, M), dtype=bool)

        # 2. Local sliding-window band diagonal
        for bi in range(M):
            start = max(0, bi - self.local_window_blocks)
            end = min(M, bi + self.local_window_blocks + 1)
            mask[bi, start:end] = True

        # 3. Deterministic Fractal Odd-Number Tree (n = 3; 2n-1, 2n+1)
        root = 3
        if root < seq_len:
            queue = [root]
            while queue:
                curr = queue.pop(0)
                left_child = 2 * curr - 1
                right_child = 2 * curr + 1

                if left_child < seq_len:
                    b_curr = curr // self.block_size
                    b_child = left_child // self.block_size
                    mask[b_curr, b_child] = True
                    mask[b_child, b_curr] = True
                    queue.append(left_child)

                if right_child < seq_len:
                    b_curr = curr // self.block_size
                    b_child = right_child // self.block_size
                    mask[b_curr, b_child] = True
                    mask[b_child, b_curr] = True
                    queue.append(right_child)

        # 4. Compile compact Lookup Table (LUT) and active block counts
        active_counts_np = mask.sum(axis=1).astype(np.int64)
        max_active = int(active_counts_np.max())
        block_lut_np = np.zeros((M, max_active), dtype=np.int64)

        for i in range(M):
            active_cols = np.where(mask[i])[0]
            block_lut_np[i, :len(active_cols)] = active_cols

        block_mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        block_lut_t = torch.tensor(block_lut_np, dtype=torch.long, device=device)
        active_counts_t = torch.tensor(active_counts_np, dtype=torch.long, device=device)

        return block_mask_t, block_lut_t, active_counts_t


class FractalBlockSparseAttention(nn.Module):
    """
    Block-Sparse Attention evaluating sparse GEMMs and block-softmax
    exclusively on active (Q_block, K_block) pairs from the compiled LUT.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        block_size: int = 32,
        local_window_blocks: int = 1,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.local_window_blocks = local_window_blocks
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.layout_gen = FractalBlockLayoutGenerator(block_size, local_window_blocks)
        self._cached_layout = None
        self._cached_seq_len = -1

    def _get_layout(self, seq_len: int, device: Optional[torch.device]):
        if self._cached_layout is None or self._cached_seq_len != seq_len:
            self._cached_layout = self.layout_gen.generate(seq_len, device=device)
            self._cached_seq_len = seq_len
        return self._cached_layout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_sz, orig_seq_len, C = x.shape
        device = x.device

        # Padding sequence to multiple of block_size
        pad_len = (self.block_size - (orig_seq_len % self.block_size)) % self.block_size
        if pad_len > 0:
            pad = torch.zeros(B_sz, pad_len, C, dtype=x.dtype, device=device)
            x_padded = torch.cat([x, pad], dim=1)