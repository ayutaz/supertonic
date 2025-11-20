"""
Duration Predictor Unit Tests

Duration Predictor関連モジュールのテスト
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dp import SentenceEncoder, StyleEncoderDP, DurationPredictor
from losses import DPLoss


def test_sentence_encoder():
    """
    Sentence Encoderのテスト
    """
    print("\nTesting SentenceEncoder...")

    batch_size = 2
    text_len = 50
    vocab_size = 1000

    # Model
    model = SentenceEncoder(
        vocab_size=vocab_size,
        char_emb_dim=64,
        convnext_idim=64,
        convnext_ksz=5,
        convnext_intermediate_dim=256,
        convnext_num_layers=6,
        attn_hidden_channels=64,
        attn_filter_channels=256,
        attn_n_heads=2,
        attn_n_layers=2,
        attn_p_dropout=0.0,
        proj_out_idim=64,
        proj_out_odim=64,
    )

    # Input
    text_ids = torch.randint(0, vocab_size, (batch_size, text_len))
    text_mask = torch.ones(batch_size, 1, text_len)

    # Forward
    output = model(text_ids, text_mask)

    # Check shape
    assert output.shape == (batch_size, 64, text_len), f"Expected (2, 64, 50), got {output.shape}"
    print(f"[OK] SentenceEncoder output shape: {output.shape}")


def test_style_encoder_dp():
    """
    Style Encoder (DP)のテスト
    """
    print("\nTesting StyleEncoderDP...")

    batch_size = 2
    ldim = 24
    chunk_compress_factor = 6
    latent_len = 100

    # Model
    model = StyleEncoderDP(
        ldim=ldim,
        chunk_compress_factor=chunk_compress_factor,
        proj_in_odim=64,
        convnext_idim=64,
        convnext_ksz=5,
        convnext_intermediate_dim=256,
        convnext_num_layers=4,
        style_input_dim=64,
        n_style=8,
        style_key_dim=0,
        style_value_dim=16,
        prototype_dim=64,
        n_units=64,
        n_heads=2,
    )

    # Input
    latent = torch.randn(batch_size, ldim * chunk_compress_factor, latent_len)
    mask = torch.ones(batch_size, 1, latent_len)

    # Forward
    output = model(latent, mask)

    # Check shape
    assert output.shape == (batch_size, 16), f"Expected (2, 16), got {output.shape}"
    print(f"[OK] StyleEncoderDP output shape: {output.shape}")


def test_duration_predictor():
    """
    Duration Predictorのテスト
    """
    print("\nTesting DurationPredictor...")

    batch_size = 2
    text_len = 50
    vocab_size = 1000
    ldim = 24
    chunk_compress_factor = 6
    ref_len = 100

    # Model config
    sentence_encoder_config = {
        "vocab_size": vocab_size,
        "char_emb_dim": 64,
        "convnext_idim": 64,
        "convnext_ksz": 5,
        "convnext_intermediate_dim": 256,
        "convnext_num_layers": 6,
        "attn_hidden_channels": 64,
        "attn_filter_channels": 256,
        "attn_n_heads": 2,
        "attn_n_layers": 2,
        "attn_p_dropout": 0.0,
        "proj_out_idim": 64,
        "proj_out_odim": 64,
    }

    style_encoder_config = {
        "ldim": ldim,
        "chunk_compress_factor": chunk_compress_factor,
        "proj_in_odim": 64,
        "convnext_idim": 64,
        "convnext_ksz": 5,
        "convnext_intermediate_dim": 256,
        "convnext_num_layers": 4,
        "style_input_dim": 64,
        "n_style": 8,
        "style_key_dim": 0,
        "style_value_dim": 16,
        "prototype_dim": 64,
        "n_units": 64,
        "n_heads": 2,
    }

    predictor_config = {
        "sentence_dim": 64,
        "style_dim": 16,
        "hdim": 128,
        "n_layer": 2,
    }

    # Model
    model = DurationPredictor(
        sentence_encoder_config=sentence_encoder_config,
        style_encoder_config=style_encoder_config,
        predictor_config=predictor_config,
    )

    # Input
    text_ids = torch.randint(0, vocab_size, (batch_size, text_len))
    text_mask = torch.ones(batch_size, 1, text_len)
    reference_latent = torch.randn(batch_size, ldim * chunk_compress_factor, ref_len)
    reference_mask = torch.ones(batch_size, 1, ref_len)

    # Forward
    duration = model(text_ids, reference_latent, text_mask, reference_mask)

    # Check shape
    assert duration.shape == (batch_size,), f"Expected (2,), got {duration.shape}"
    assert (duration >= 0).all(), "Duration should be non-negative"
    print(f"[OK] DurationPredictor output shape: {duration.shape}")
    print(f"[OK] Predicted durations: {duration.tolist()}")


def test_duration_predictor_inference():
    """
    Duration Predictor推論のテスト
    """
    print("\nTesting DurationPredictor.inference()...")

    batch_size = 2
    text_len = 50
    vocab_size = 1000
    ldim = 24
    chunk_compress_factor = 6
    ref_len = 100

    # Model config
    sentence_encoder_config = {
        "vocab_size": vocab_size,
        "char_emb_dim": 64,
        "convnext_idim": 64,
        "convnext_ksz": 5,
        "convnext_intermediate_dim": 256,
        "convnext_num_layers": 6,
        "attn_hidden_channels": 64,
        "attn_filter_channels": 256,
        "attn_n_heads": 2,
        "attn_n_layers": 2,
        "attn_p_dropout": 0.0,
        "proj_out_idim": 64,
        "proj_out_odim": 64,
    }

    style_encoder_config = {
        "ldim": ldim,
        "chunk_compress_factor": chunk_compress_factor,
        "proj_in_odim": 64,
        "convnext_idim": 64,
        "convnext_ksz": 5,
        "convnext_intermediate_dim": 256,
        "convnext_num_layers": 4,
        "style_input_dim": 64,
        "n_style": 8,
        "style_key_dim": 0,
        "style_value_dim": 16,
        "prototype_dim": 64,
        "n_units": 64,
        "n_heads": 2,
    }

    predictor_config = {
        "sentence_dim": 64,
        "style_dim": 16,
        "hdim": 128,
        "n_layer": 2,
    }

    # Model
    model = DurationPredictor(
        sentence_encoder_config=sentence_encoder_config,
        style_encoder_config=style_encoder_config,
        predictor_config=predictor_config,
    )
    model.eval()

    # Input
    text_ids = torch.randint(0, vocab_size, (batch_size, text_len))
    text_mask = torch.ones(batch_size, 1, text_len)
    reference_latent = torch.randn(batch_size, ldim * chunk_compress_factor, ref_len)
    reference_mask = torch.ones(batch_size, 1, ref_len)

    # Inference with different speeds
    with torch.no_grad():
        duration_normal = model.inference(text_ids, reference_latent, text_mask, reference_mask, speed=1.0)
        duration_fast = model.inference(text_ids, reference_latent, text_mask, reference_mask, speed=1.5)
        duration_slow = model.inference(text_ids, reference_latent, text_mask, reference_mask, speed=0.8)

    # Check shapes
    assert duration_normal.shape == (batch_size,), f"Expected (2,), got {duration_normal.shape}"
    assert duration_fast.shape == (batch_size,), f"Expected (2,), got {duration_fast.shape}"
    assert duration_slow.shape == (batch_size,), f"Expected (2,), got {duration_slow.shape}"

    # Check speed adjustment
    assert torch.all(duration_fast < duration_normal), "Fast speed should produce shorter duration"
    assert torch.all(duration_slow > duration_normal), "Slow speed should produce longer duration"

    print(f"[OK] Inference output shape: {duration_normal.shape}")
    print(f"[OK] Duration (speed=1.0): {duration_normal.tolist()}")
    print(f"[OK] Duration (speed=1.5): {duration_fast.tolist()}")
    print(f"[OK] Duration (speed=0.8): {duration_slow.tolist()}")


def test_dp_loss():
    """
    DP Lossのテスト
    """
    print("\nTesting DPLoss...")

    batch_size = 4

    # Loss function
    loss_fn = DPLoss()

    # Input
    predicted_duration = torch.randn(batch_size) + 3.0  # Mean around 3 seconds
    ground_truth_duration = torch.randn(batch_size) + 3.0

    # Forward
    loss, metrics = loss_fn(predicted_duration, ground_truth_duration)

    # Check
    assert loss.item() >= 0, "Loss should be non-negative"
    assert "mse_loss" in metrics, "Metrics should contain 'mse_loss'"
    assert "mean_predicted_duration" in metrics, "Metrics should contain 'mean_predicted_duration'"
    assert "mean_ground_truth_duration" in metrics, "Metrics should contain 'mean_ground_truth_duration'"

    print(f"[OK] DPLoss: {loss.item():.6f}")
    print(f"  - MSE Loss: {metrics['mse_loss']:.6f}")
    print(f"  - Mean Predicted Duration: {metrics['mean_predicted_duration']:.3f}s")
    print(f"  - Mean Ground Truth Duration: {metrics['mean_ground_truth_duration']:.3f}s")


def main():
    """
    全テスト実行
    """
    print("=" * 60)
    print("Running Duration Predictor Unit Tests")
    print("=" * 60)

    try:
        test_sentence_encoder()
        test_style_encoder_dp()
        test_duration_predictor()
        test_duration_predictor_inference()
        test_dp_loss()

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
