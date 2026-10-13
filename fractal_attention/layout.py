import numpy as np
import torch
from typing import Optional, Tuple

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

        mask = np.zeros((M, M), dtype=bool)

        for bi in range(M):
            start = max(0, bi - self.local_window_blocks)
            end = min(M, bi + self.local_window_blocks + 1)
            mask[bi, start:end] = True

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
