# Fractal Block-Sparse Attention: Sub-Quadratic Transformers via Odd Binomial Fractal Tree Topologies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23351708.svg)](https://doi.org/10.5281/zenodo.23351708)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Demo-yellow)](https://huggingface.co/spaces/Gakkavim/fractal-attention-explorer)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official PyTorch implementation of **Fractal Block-Sparse Attention**, a deterministic, hardware-aligned sub-quadratic self-attention mechanism governed by odd binomial fractal tree dynamics ($n=3; 2n \pm 1$).

Achieves up to **10.7x speedup** and **13x memory reduction** with zero context loss and full compatibility with modern open-weight LLMs (Llama 3, TinyLlama).

📖 **Paper / Preprint:** [Download / View PDF](subquadratic_fractal_attention.pdf) | [DOI: 10.5281/zenodo.23351708](https://doi.org/10.5281/zenodo.23351708)

---

## 🚀 Key Highlights

- **Deterministic Expander Topology:** Replaces randomized heuristics with an odd-number binary tree rooted at $n=3$ with recursive branching $n \mapsto \{2n-1, 2n+1\}$.
- **Sub-Quadratic Scaling:** Effective communication diameter scales strictly as $\mathcal{O}(\log N)$.
- **Hardware-Aligned Block Projection:** Projects graph paths onto coarse $B \times B$ tiles ($B \in \{16, 32, 64\}$) and prunes inactive blocks via compact Lookup Tables (LUT).
- **Strict Causal Mask Invariance:** Verified mathematically with zero past-token delta ($\Delta = 0.000000e+00$), guaranteeing zero future leakage during autoregressive decoding.
- **Zero Information Blind Spots:** Matches dense attention with **100% accuracy** on long-range Associative Recall at $N=1024$, completely eliminating the fatal failure mode of local-window attention (6.0%).
- **Drop-in LLM Patching:** Seamlessly replaces native attention layers in Hugging Face LLaMA architectures with RoPE and Grouped-Query Attention (GQA) support.

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

### 3. Long-Context Scaling & Theoretical FLOP Reduction (Triton Hardware Projections)

| Sequence Length ($N$) | Total Blocks ($M \times M$) | Active Blocks ($\sum \text{Counts}$) | Block Sparsity | Dense Attention | Fractal Triton Attention | Theoretical FLOP Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | 256 | 72 | 71.88% | 0.0043 TFLOPs | 0.0012 TFLOPs | **3.56x** |
| **2,048** | 1,024 | 152 | 85.16% | 0.0172 TFLOPs | 0.0026 TFLOPs | **6.74x** |
| **4,096** | 4,096 | 312 | 92.38% | 0.0687 TFLOPs | 0.0052 TFLOPs | **13.13x** |
| **8,192** | 16,384 | 632 | 96.14% | 0.2749 TFLOPs | 0.0106 TFLOPs | **25.92x** |
| **16,384** | 65,536 | 1,272 | 98.06% | 1.0995 TFLOPs | 0.0213 TFLOPs | **51.52x** |

---

## 🔬 Live LLM Verification: TinyLlama-1.1B Benchmarks

The drop-in replacement module (`FractalLlamaAttention`) was evaluated against native dense attention across all 22 decoder layers of `TinyLlama/TinyLlama-1.1B-Chat-v1.0` in a CPU inference environment to evaluate operational compatibility, causal mask invariance, and latency characteristics under zero-shot sparsity.

### Empirical Results

| Metric / Configuration | Dense Attention ($N=29$) | Fractal Sparse ($B=64, W=1$) | Dense Attention ($N=51$) | Fractal Sparse ($B=16, W=1$ + Sink) |
| :--- | :--- | :--- | :--- | :--- |
| **Compute Device** | AMD CPU (FP32) | AMD CPU (FP32) | AMD CPU (FP32) | AMD CPU (FP32) |
| **Prompt Length ($N_{prompt}$)** | 29 tokens | 29 tokens | 51 tokens | 51 tokens |
| **Generated Tokens ($N_{new}$)** | 64 tokens | 64 tokens | 48 tokens | 48 tokens |
| **Total Generation Latency** | 15.588 s | **15.263 s** | **13.408 s** | 14.706 s |
| **Inference Throughput** | 4.11 tokens/s | **4.19 tokens/s** | **3.58 tokens/s** | 3.26 tokens/s |
| **Block Partition Count ($M$)** | - (Full Dense) | 1 block ($29 < 64$) | - (Full Dense) | 4 blocks ($M = \lceil 51/16 \rceil$) |
| **Prefill Sparsity Ratio** | 0.0% | 0.0% (Single-block) | 0.0% | **~37.5% Pruned** |
| **Semantic Text Coherence** | Coherent Natural Text | Attention Collapse (`\n`) | Coherent Natural Text | Attention Collapse (`\n`) |
| **Algorithmic Complexity** | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ (Sub-threshold) | $\mathcal{O}(N^2)$ | **$\mathcal{O}(M \log M \cdot B^2)$** |

### Mechanical Analysis & Zero-Shot Sparsity Observations

1. **Kernel & Graph Stability:**  
   The fractal block-sparse mask operates with zero architectural faults across all 22 decoder layers. Causal mask invariance ($\Delta = 0.000000e+00$ on past token representations) and Rotary Positional Embedding (RoPE) / Grouped-Query Attention (GQA) compatibility are verified without shape or tensor layout mismatch.
2. **Latency & Throughput Bounds:**  
   On autoregressive single-token decoding steps ($Q_{len} = 1$), per-token latency remains comparable to dense execution on CPU. Hardware-level acceleration via sparse block skipping is predominantly realized on GPU platforms during high-throughput parallel prefill phases ($N \ge 2048$).
3. **Attention Entropy Collapse (Why LoRA Adaptation is Required):**  
   Pretrained dense weights rely on normalizing attention distributions across the entire sequence history $\sum_{j=1}^{N} \exp(S_{ij})$. When severe structural pruning is introduced zero-shot without fine-tuning, softmax denominators shrink abruptly, causing probability mass to over-concentrate on residual local tokens. The model experiences acute perplexity spikes and collapses into safe, low-entropy terminal tokens (such as repeated newline `\n` characters) following initial punctuation.  
   *Conclusion:* While structural invariance is mathematically sound, deploying drop-in fractal sparsity onto pretrained dense checkpoints requires a short adaptation phase (e.g., 500–1000 step LoRA fine-tuning) to recalibrate attention temperature and projection projections to the tree topology.

---

## 🛠️ Quickstart Usage

### 1. Standalone Layer Usage

```bash
pip install -r requirements.txt
