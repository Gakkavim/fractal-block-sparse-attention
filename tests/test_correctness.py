import torch
from fractal_attention.layout import FractalBlockLayoutGenerator
from fractal_attention.ops_pytorch import FractalBlockSparseAttention

def test_layout_generation():
    gen = FractalBlockLayoutGenerator(block_size=32)
    mask, lut, counts = gen.generate(seq_len=256)
    assert mask.shape == (8, 8)
    assert counts.shape == (8,)
    print("✓ Layout Generator sanity test passed!")

def test_pytorch_layer_forward():
    x = torch.randn(1, 128, 64)
    layer = FractalBlockSparseAttention(embed_dim=64, num_heads=4, block_size=32)
    out = layer(x)
    assert out.shape == x.shape
    print("✓ PyTorch Attention forward sanity test passed!")

if __name__ == "__main__":
    test_layout_generation()
    test_pytorch_layer_forward()
