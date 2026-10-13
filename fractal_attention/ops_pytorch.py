import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layout import FractalBlockLayoutGenerator

class FractalBlockSparseAttention(nn.Module):
    """
    Reference PyTorch Multi-Head Block-Sparse Attention.
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

        pad_len = (self.block_size - (orig_seq_len % self.block_size)) % self.block_size
        if pad_len > 0:
            pad = torch.zeros(B_sz, pad_len, C, dtype=x.dtype, device=device)
            x_padded = torch.cat([x, pad], dim=1)
        else:
            x_padded = x

        seq_len = x_padded.shape[1]
        M = seq_len // self.block_size

        q = self.q_proj(x_padded).view(B_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_padded).view(B_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_padded).view(B_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_blocks = q.view(B_sz, self.num_heads, M, self.block_size, self.head_dim)
        k_blocks = k.view(B_sz, self.num_heads, M, self.block_size, self.head_dim)
        v_blocks = v.view(B_sz, self.num_heads, M, self.block_size, self.head_dim)

        _, block_lut, active_counts = self._get_layout(seq_len, device)

        out_blocks_list = []
        for i in range(M):
            c_i = int(active_counts[i].item())
            active_k_indices = block_lut[i, :c_i].tolist()

            qi = q_blocks[:, :, i, :, :]
            ki = k_blocks[:, :, active_k_indices, :, :]
            vi = v_blocks[:, :, active_k_indices, :, :]

            ki_flat = ki.reshape(B_sz, self.num_heads, c_i * self.block_size, self.head_dim)
            vi_flat = vi.reshape(B_sz, self.num_heads, c_i * self.block_size, self.head_dim)

            scores_i = torch.matmul(qi, ki_flat.transpose(-1, -2)) * self.scale
            attn_i = F.softmax(scores_i, dim=-1)
            attn_i = self.dropout(attn_i)

            out_i = torch.matmul(attn_i, vi_flat)
            out_blocks_list.append(out_i)

        out_blocks = torch.stack(out_blocks_list, dim=2)
        out = out_blocks.view(B_sz, self.num_heads, seq_len, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(B_sz, seq_len, C)
        output = self.out_proj(out)

        if pad_len > 0:
            output = output[:, :orig_seq_len, :]

        return output
