# fractal-block-sparse-attention
Sub-quadratic Block-Sparse Attention via odd binomial fractal trees (n=3; 2n±1). Achieves 10.7x speedup and 13x memory reduction with zero context loss.

# Sub-Quadratic Block-Sparse Attention via Odd Binomial Fractal Tree Topologies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23351708.svg)](https://doi.org/10.5281/zenodo.23351708)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official PyTorch implementation of **Fractal Block-Sparse Attention**, a deterministic, hardware-aligned sub-quadratic self-attention mechanism governed by odd binomial fractal tree dynamics.

📖 **Paper / Preprint:** [DOI: 10.5281/zenodo.23351708](https://doi.org/10.5281/zenodo.23351708)

---

## 🚀 Key Highlights

- **Deterministic Expander Topology:** Replaces randomized heuristics with an odd-number binary tree rooted at $n=3$ with recursive branching $n \mapsto \{2n-1, 2n+1\}$.
- **Sub-Quadratic Scaling:** Effective communication diameter scales strictly as $\mathcal{O}(\log N)$.
- **Hardware-Aligned Block Projection:** Projects graph paths onto coarse $B \times B$ tiles ($B \in \{32, 64\}$) and prunes inactive blocks via compact Lookup Tables (LUT).
- **Zero Information Blind Spots:** Matches dense attention with **100% accuracy** on long-range Associative Recall at $N=1024$, completely eliminating the fatal failure mode of local-window attention (6.0%).
- **Empirical Efficiency:** Up to **10.73x CPU acceleration** and **13.13x memory reduction** at $N=2048$ (92.38% block sparsity).

---

## 📊 Empirical Benchmarks

### 1. Latency & Memory Scaling ($B=32$)

| Sequence Length ($N$) | Grid ($M \times M$) | Block Sparsity | Dense Attention | Fractal Sparse Attention | Speedup | Peak Memory Footprint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **256** | $8 \times 8$ | 50.00% | 38.50 ms | 15.32 ms | **2.51x** | 1.00 MB vs 2.00 MB |
| **512** | $16 \times 16$ | 71.88% | 116.94 ms | 37.13 ms | **3.15x** | 2.25 MB vs 8.00 MB |
| **1024** | $32 \times 32$ | 85.16% | 782.61 ms | 141.29 ms | **4.42x** | 4.75 MB vs 32.00 MB |
| **2048** | $64 \times 64$ | 92.38% | 2425.26 ms | 225.93 ms | **10.73x** | 9.75 MB vs 128.00 MB |

### 2. Long-Range Associative Recall ($N=1024$, 4 Layers, 500 Steps)

| Model Architecture | Graph Topology | Validation Loss | Validation Accuracy | Information Reach |
| :--- | :---: | :---: | :---: | :--- |
| **Standard Dense** | $\mathcal{O}(N^2)$ Complete | 0.0124 | **100.0%** | Full Direct (1 Step) |
| **Local-Window Block** | $\mathcal{O}(N)$ Band ($w=1$) | 2.7684 | 6.0% | Failed (Blind Spot) |
| **Fractal Block-Sparse** | $\mathcal{O}(N \log N)$ Fractal | 0.0381 | **100.0%** | Full Context (4-Hop Highway) |

---

## 🛠️ Quickstart Usage

```python
import torch
from fractal_attention import FractalBlockSparseAttention

# Batch size 1, sequence length 2048, hidden dimension 256, 8 heads
x = torch.randn(1, 2048, 256)

layer = FractalBlockSparseAttention(
    embed_dim=256,
    num_heads=8,
    block_size=32,
    local_window_blocks=1
)

output = layer(x)
print(output.shape) # torch.Size([1, 2048, 256])

## Citation

If you use this architecture or find our findings useful in your research, please cite our preprint:

```bibtex
@article{akkavim2026fractal,
  title={Sub-Quadratic Block-Sparse Attention via Odd Binomial Fractal Tree Topologies},
  author={Akkavim, G{\"o}ksenin},
  journal={Zenodo Preprints},
  year={2026},
  doi={10.5281/zenodo.23351708},
  url={[https://doi.org/10.5281/zenodo.23351708](https://doi.org/10.5281/zenodo.23351708)}
}
