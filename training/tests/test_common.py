"""
Common Modules Unit Tests

Phase 1の共通モジュール（ConvNeXt, Attention, Layers）のテスト
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.common import (
    # Attention
    LARoPE,
    MultiHeadAttention,
    # ConvNeXt
    ConvNeXtBlock,
    ConvNeXtStack,
    InitialConvNeXt,
    # Layers
    CausalConv1d,
    FiLM,
    LayerScale,
    StyleTokenLayer,
    TimeEmbedding,
    # Utils
    get_activation,
    init_weights,
    sequence_mask,
)


# ============================================================
# Attention Tests
# ============================================================

def test_larope():
    """
    LARoPEのテスト
    """
    print("\nTesting LARoPE...")

    dim = 64
    seq_len = 100
    text_len = 50

    # Model
    rope = LARoPE(dim=dim, rotary_base=10000, rotary_scale=10.0)

    # Forward
    cos, sin = rope(seq_len, text_len, device=torch.device("cpu"))

    # Check shape
    expected_dim = dim // 2
    assert cos.shape == (seq_len, expected_dim), f"Expected ({seq_len}, {expected_dim}), got {cos.shape}"
    assert sin.shape == (seq_len, expected_dim), f"Expected ({seq_len}, {expected_dim}), got {sin.shape}"

    print(f"[OK] LARoPE cos shape: {cos.shape}")
    print(f"[OK] LARoPE sin shape: {sin.shape}")

    # Test apply_rotary_pos_emb
    batch_size = 2
    n_heads = 4
    x = torch.randn(batch_size, n_heads, seq_len, dim)
    x_rotated = LARoPE.apply_rotary_pos_emb(x, cos, sin)

    assert x_rotated.shape == x.shape, f"Expected {x.shape}, got {x_rotated.shape}"
    print(f"[OK] apply_rotary_pos_emb output shape: {x_rotated.shape}")


def test_multi_head_attention():
    """
    MultiHeadAttentionのテスト
    """
    print("\nTesting MultiHeadAttention...")

    batch_size = 2
    seq_len = 50
    hidden_channels = 256
    filter_channels = 1024
    n_heads = 4

    # Model
    attn = MultiHeadAttention(
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=2,
        p_dropout=0.0,
        use_rope=True,
        rotary_base=10000,
        rotary_scale=10.0,
    )

    # Input
    x = torch.randn(batch_size, hidden_channels, seq_len)
    mask = torch.ones(batch_size, seq_len)
    text_len = seq_len

    # Forward
    output = attn(x, mask=mask, text_len=text_len)

    # Check shape
    assert output.shape == (batch_size, hidden_channels, seq_len), \
        f"Expected ({batch_size}, {hidden_channels}, {seq_len}), got {output.shape}"
    print(f"[OK] MultiHeadAttention output shape: {output.shape}")


# ============================================================
# ConvNeXt Tests
# ============================================================

def test_convnext_block():
    """
    ConvNeXtBlockのテスト
    """
    print("\nTesting ConvNeXtBlock...")

    batch_size = 2
    channels = 256
    seq_len = 100

    # Model
    block = ConvNeXtBlock(
        dim=channels,
        kernel_size=7,
        dilation=1,
        expansion_factor=4,
        layer_scale_init=1e-6,
        dropout=0.0,
    )

    # Input
    x = torch.randn(batch_size, channels, seq_len)

    # Forward
    output = block(x)

    # Check shape
    assert output.shape == (batch_size, channels, seq_len), \
        f"Expected ({batch_size}, {channels}, {seq_len}), got {output.shape}"
    print(f"[OK] ConvNeXtBlock output shape: {output.shape}")


def test_convnext_stack():
    """
    ConvNeXtStackのテスト
    """
    print("\nTesting ConvNeXtStack...")

    batch_size = 2
    channels = 256
    seq_len = 100

    # Model
    stack = ConvNeXtStack(
        idim=channels,
        ksz=5,
        intermediate_dim=1024,
        num_layers=4,
        dilation_lst=[1, 2, 4, 8],
    )

    # Input
    x = torch.randn(batch_size, channels, seq_len)

    # Forward
    output = stack(x)

    # Check shape
    assert output.shape == (batch_size, channels, seq_len), \
        f"Expected ({batch_size}, {channels}, {seq_len}), got {output.shape}"
    print(f"[OK] ConvNeXtStack output shape: {output.shape}")


def test_initial_convnext():
    """
    InitialConvNeXtのテスト
    """
    print("\nTesting InitialConvNeXt...")

    batch_size = 2
    in_channels = 128
    out_channels = 256
    seq_len = 100

    # Model
    init_conv = InitialConvNeXt(
        in_channels=in_channels,
        out_channels=out_channels,
        ksz_init=7,
        ksz=7,
        intermediate_dim=512,
        num_layers=2,
        dropout=0.0,
    )

    # Input
    x = torch.randn(batch_size, in_channels, seq_len)

    # Forward
    output = init_conv(x)

    # Check shape
    assert output.shape == (batch_size, out_channels, seq_len), \
        f"Expected ({batch_size}, {out_channels}, {seq_len}), got {output.shape}"
    print(f"[OK] InitialConvNeXt output shape: {output.shape}")


# ============================================================
# Layers Tests
# ============================================================

def test_causal_conv1d():
    """
    CausalConv1dのテスト
    """
    print("\nTesting CausalConv1d...")

    batch_size = 2
    in_channels = 64
    out_channels = 128
    seq_len = 100

    # Model
    causal_conv = CausalConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        dilation=1,
    )

    # Input
    x = torch.randn(batch_size, in_channels, seq_len)

    # Forward
    output = causal_conv(x)

    # Check shape (output length should be same as input for causal conv)
    assert output.shape == (batch_size, out_channels, seq_len), \
        f"Expected ({batch_size}, {out_channels}, {seq_len}), got {output.shape}"
    print(f"[OK] CausalConv1d output shape: {output.shape}")


def test_layer_scale():
    """
    LayerScaleのテスト
    """
    print("\nTesting LayerScale...")

    batch_size = 2
    dim = 256
    seq_len = 100

    # Model
    layer_scale = LayerScale(dim=dim, init_value=1e-6)

    # Input
    x = torch.randn(batch_size, dim, seq_len)

    # Forward
    output = layer_scale(x)

    # Check shape
    assert output.shape == (batch_size, dim, seq_len), \
        f"Expected ({batch_size}, {dim}, {seq_len}), got {output.shape}"

    # Check that scaling is applied
    assert not torch.allclose(output, x), "LayerScale should modify the input"
    print(f"[OK] LayerScale output shape: {output.shape}")


def test_film():
    """
    FiLMのテスト
    """
    print("\nTesting FiLM...")

    batch_size = 2
    feature_dim = 256
    cond_dim = 128
    seq_len = 100

    # Model
    film = FiLM(cond_dim=cond_dim, feature_dim=feature_dim)

    # Input
    x = torch.randn(batch_size, feature_dim, seq_len)
    cond = torch.randn(batch_size, cond_dim)

    # Forward
    output = film(x, cond)

    # Check shape
    assert output.shape == (batch_size, feature_dim, seq_len), \
        f"Expected ({batch_size}, {feature_dim}, {seq_len}), got {output.shape}"

    # Check that modulation is applied
    assert not torch.allclose(output, x), "FiLM should modify the input"
    print(f"[OK] FiLM output shape: {output.shape}")


def test_time_embedding():
    """
    TimeEmbeddingのテスト
    """
    print("\nTesting TimeEmbedding...")

    batch_size = 4
    dim = 128

    # Model
    time_emb = TimeEmbedding(dim=dim)

    # Input (time values between 0 and 1)
    t = torch.rand(batch_size)

    # Forward
    output = time_emb(t)

    # Check shape
    assert output.shape == (batch_size, dim), \
        f"Expected ({batch_size}, {dim}), got {output.shape}"
    print(f"[OK] TimeEmbedding output shape: {output.shape}")


def test_style_token_layer():
    """
    StyleTokenLayerのテスト
    """
    print("\nTesting StyleTokenLayer...")

    batch_size = 2
    input_dim = 256
    seq_len = 100
    n_style = 50
    style_value_dim = 256

    # Model (with key-value style tokens)
    style_layer_kv = StyleTokenLayer(
        input_dim=input_dim,
        n_style=n_style,
        style_key_dim=256,
        style_value_dim=style_value_dim,
        prototype_dim=256,
        n_units=256,
        n_heads=2,
    )

    # Model (prototype-based)
    style_layer_proto = StyleTokenLayer(
        input_dim=input_dim,
        n_style=n_style,
        style_key_dim=0,  # prototype-based
        style_value_dim=style_value_dim,
        prototype_dim=256,
        n_units=256,
        n_heads=2,
    )

    # Input
    x = torch.randn(batch_size, input_dim, seq_len)
    mask = torch.ones(batch_size, seq_len)

    # Forward (key-value)
    output_kv = style_layer_kv(x, mask=mask)
    assert output_kv.shape == (batch_size, style_value_dim), \
        f"Expected ({batch_size}, {style_value_dim}), got {output_kv.shape}"
    print(f"[OK] StyleTokenLayer (key-value) output shape: {output_kv.shape}")

    # Forward (prototype-based)
    output_proto = style_layer_proto(x, mask=mask)
    assert output_proto.shape == (batch_size, style_value_dim), \
        f"Expected ({batch_size}, {style_value_dim}), got {output_proto.shape}"
    print(f"[OK] StyleTokenLayer (prototype) output shape: {output_proto.shape}")


# ============================================================
# Utils Tests
# ============================================================

def test_get_activation():
    """
    get_activationのテスト
    """
    print("\nTesting get_activation...")

    # Test various activation functions
    activations = ["relu", "gelu", "silu", "tanh", "sigmoid"]

    for act_name in activations:
        act_fn = get_activation(act_name)
        x = torch.randn(2, 10)
        output = act_fn(x)
        assert output.shape == x.shape, f"Activation {act_name} should preserve shape"

    print(f"[OK] get_activation tested for: {', '.join(activations)}")


def test_sequence_mask():
    """
    sequence_maskのテスト
    """
    print("\nTesting sequence_mask...")

    batch_size = 4
    max_len = 100

    # Lengths
    lengths = torch.tensor([50, 75, 100, 30])

    # Create mask
    mask = sequence_mask(lengths, max_len)

    # Check shape
    assert mask.shape == (batch_size, max_len), \
        f"Expected ({batch_size}, {max_len}), got {mask.shape}"

    # Check mask values
    for i, length in enumerate(lengths):
        assert mask[i, :length].all(), f"Mask should be True up to length {length}"
        if length < max_len:
            assert not mask[i, length:].any(), f"Mask should be False after length {length}"

    print(f"[OK] sequence_mask output shape: {mask.shape}")
    print(f"[OK] sequence_mask values correct for lengths: {lengths.tolist()}")


def test_init_weights():
    """
    init_weightsのテスト
    """
    print("\nTesting init_weights...")

    # Create a simple module
    import torch.nn as nn

    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.conv = nn.Conv1d(10, 10, 3)

        def forward(self, x):
            return x

    # Create and initialize
    model = SimpleModule()
    init_weights(model)

    # Check that weights are not all zeros
    linear_weight = model.linear.weight.data
    conv_weight = model.conv.weight.data

    assert not torch.allclose(linear_weight, torch.zeros_like(linear_weight)), \
        "Linear weights should be initialized"
    assert not torch.allclose(conv_weight, torch.zeros_like(conv_weight)), \
        "Conv weights should be initialized"

    print(f"[OK] init_weights successfully initialized model parameters")


# ============================================================
# Main Test Runner
# ============================================================

def main():
    """
    全テスト実行
    """
    print("=" * 60)
    print("Running Common Modules Unit Tests (Phase 1)")
    print("=" * 60)

    try:
        # Attention Tests
        print("\n" + "=" * 60)
        print("ATTENTION TESTS")
        print("=" * 60)
        test_larope()
        test_multi_head_attention()

        # ConvNeXt Tests
        print("\n" + "=" * 60)
        print("CONVNEXT TESTS")
        print("=" * 60)
        test_convnext_block()
        test_convnext_stack()
        test_initial_convnext()

        # Layers Tests
        print("\n" + "=" * 60)
        print("LAYERS TESTS")
        print("=" * 60)
        test_causal_conv1d()
        test_layer_scale()
        test_film()
        test_time_embedding()
        test_style_token_layer()

        # Utils Tests
        print("\n" + "=" * 60)
        print("UTILS TESTS")
        print("=" * 60)
        test_get_activation()
        test_sequence_mask()
        test_init_weights()

        print("\n" + "=" * 60)
        print("All tests passed! [SUCCESS]")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
