import math
import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _fractal_sparse_fwd_kernel(
        Q, K, V, sm_scale,
        Out,
        block_lut, active_counts,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        stride_lut_m, stride_lut_k,
        Z, H, N,
        MAX_ACTIVE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        mask_m = offs_m[:, None] < N
        q = tl.load(q_ptrs, mask=mask_m, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        num_active = tl.load(active_counts + start_m)

        for step in range(MAX_ACTIVE):
            if step < num_active:
                col_block = tl.load(block_lut + start_m * stride_lut_m + step * stride_lut_k)

                if not IS_CAUSAL or (col_block * BLOCK_N <= (start_m + 1) * BLOCK_M):
                    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
                    k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
                    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
                    mask_n = offs_n[None, :] < N

                    k = tl.load(k_ptrs, mask=mask_n, other=0.0)
                    v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

                    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                    qk += tl.dot(q, k)
                    qk *= sm_scale

                    if IS_CAUSAL:
                        causal_mask = offs_m[:, None] >= offs_n[None, :]
                        qk = tl.where(causal_mask, qk, float("-inf"))

                    m_ij = tl.maximum(m_i, tl.max(qk, 1))
                    p = tl.exp(qk - m_ij[:, None])
                    alpha = tl.exp(m_i - m_ij)

                    acc = acc * alpha[:, None]
                    l_i = l_i * alpha + tl.sum(p, 1)

                    p = p.to(v.dtype)
                    acc += tl.dot(p, v)
                    m_i = m_ij

        acc = acc / l_i[:, None]
        out_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(out_ptrs, acc.to(Out.dtype), mask=mask_m)

def fractal_triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_lut: torch.Tensor,
    active_counts: torch.Tensor,
    is_causal: bool = False,
    block_size: int = 32
) -> torch.Tensor:
    if not HAS_TRITON:
        raise ImportError("Triton is not installed. Please run `pip install triton`.")
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA."

    B, H, N, D = q.shape
    M = (N + block_size - 1) // block_size
    MAX_ACTIVE = block_lut.shape[1]

    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)

    grid = (M, B * H)

    _fractal_sparse_fwd_kernel[grid](
        q, k, v, sm_scale,
        out,
        block_lut.to(torch.int32), active_counts.to(torch.int32),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        block_lut.stride(0), block_lut.stride(1),
        B, H, N,
        MAX_ACTIVE=MAX_ACTIVE,
        BLOCK_M=block_size,
        BLOCK_DMODEL=D,
        BLOCK_N=block_size,
        IS_CAUSAL=is_causal,
        num_warps=4 if D <= 64 else 8,
        num_stages=2
    )

    return out
