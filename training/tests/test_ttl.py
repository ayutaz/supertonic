"""
TTL (Text-to-Latent) Unit Tests

TTLモジュールの動作確認
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from models.ttl import (
    TextEncoder,
    StyleEncoder,
    VectorField,
    CrossAttentionLARoPE,
    TTLModel,
)
from losses import FlowMatchingLoss, TTLLoss


class TestTextEncoder:
    """TextEncoderのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        vocab_size = 1000
        batch_size = 2
        text_len = 50

        encoder = TextEncoder(
            vocab_size=vocab_size,
            char_emb_dim=256,
            convnext_num_layers=6,
            attn_n_layers=4,
            attn_n_heads=4,
        )

        # Input
        text_ids = torch.randint(0, vocab_size, (batch_size, text_len))
        text_mask = torch.ones(batch_size, 1, text_len)

        # Forward
        text_emb = encoder(text_ids, text_mask)

        # Expected: [batch, 256, text_len]
        assert text_emb.shape == (batch_size, 256, text_len), \
            f"Shape mismatch: {text_emb.shape} != (2, 256, 50)"

        print(f"[OK] TextEncoder output shape: {text_emb.shape}")


class TestStyleEncoder:
    """StyleEncoderのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        batch_size = 2
        latent_dim = 24
        chunk_compress_factor = 6
        latent_len = 100

        encoder = StyleEncoder(
            ldim=latent_dim,
            chunk_compress_factor=chunk_compress_factor,
            n_style=50,
            style_value_dim=256,
        )

        # Input: [batch, ldim * chunk_compress_factor, latent_len]
        latent = torch.randn(batch_size, latent_dim * chunk_compress_factor, latent_len)
        mask = torch.ones(batch_size, 1, latent_len)

        # Forward
        style_emb = encoder(latent, mask)

        # Expected: [batch, style_value_dim=256]
        assert style_emb.shape == (batch_size, 256), \
            f"Shape mismatch: {style_emb.shape} != (2, 256)"

        print(f"[OK] StyleEncoder output shape: {style_emb.shape}")


class TestCrossAttentionLARoPE:
    """CrossAttentionLARoPEのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        batch_size = 2
        q_dim = 512
        kv_dim = 256
        q_len = 100
        kv_len = 50

        attn = CrossAttentionLARoPE(
            q_dim=q_dim,
            kv_dim=kv_dim,
            n_heads=4,
        )

        # Input
        query = torch.randn(batch_size, q_dim, q_len)
        key_value = torch.randn(batch_size, kv_dim, kv_len)
        q_mask = torch.ones(batch_size, 1, q_len)
        kv_mask = torch.ones(batch_size, 1, kv_len)

        # Forward
        output = attn(query, key_value, q_mask, kv_mask, text_len=kv_len)

        # Expected: [batch, q_dim, q_len]
        assert output.shape == (batch_size, q_dim, q_len), \
            f"Shape mismatch: {output.shape} != (2, 512, 100)"

        print(f"[OK] CrossAttentionLARoPE output shape: {output.shape}")


class TestVectorField:
    """VectorFieldのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        batch_size = 2
        ldim = 24
        chunk_compress_factor = 6
        latent_len = 100
        text_len = 50

        vector_field = VectorField(
            ldim=ldim,
            chunk_compress_factor=chunk_compress_factor,
            n_blocks=4,
        )

        # Input
        noisy_latent = torch.randn(batch_size, ldim * chunk_compress_factor, latent_len)
        t = torch.rand(batch_size)
        text_emb = torch.randn(batch_size, 256, text_len)
        style_emb = torch.randn(batch_size, 256)
        latent_mask = torch.ones(batch_size, 1, latent_len)
        text_mask = torch.ones(batch_size, 1, text_len)

        # Forward
        velocity = vector_field(
            noisy_latent=noisy_latent,
            t=t,
            text_emb=text_emb,
            style_emb=style_emb,
            latent_mask=latent_mask,
            text_mask=text_mask,
        )

        # Expected: [batch, ldim * chunk_compress_factor, latent_len]
        expected_shape = (batch_size, ldim * chunk_compress_factor, latent_len)
        assert velocity.shape == expected_shape, \
            f"Shape mismatch: {velocity.shape} != {expected_shape}"

        print(f"[OK] VectorField output shape: {velocity.shape}")


class TestTTLModel:
    """TTLModelのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        vocab_size = 1000
        batch_size = 2
        text_len = 50
        latent_dim = 24
        chunk_compress_factor = 6
        latent_len = 100
        ref_len = 80

        model = TTLModel(
            vocab_size=vocab_size,
            text_encoder_config={
                "char_emb_dim": 256,
                "convnext_num_layers": 6,
                "convnext_dilation_lst": [1, 1, 1, 1, 1, 1],
                "attn_n_layers": 4,
                "attn_n_heads": 4,
            },
            style_encoder_config={
                "ldim": latent_dim,
                "chunk_compress_factor": chunk_compress_factor,
                "convnext_num_layers": 6,
                "convnext_dilation_lst": [1, 1, 1, 1, 1, 1],
            },
            vector_field_config={
                "ldim": latent_dim,
                "chunk_compress_factor": chunk_compress_factor,
                "n_blocks": 4,
            },
        )

        # Input
        noisy_latent = torch.randn(batch_size, latent_dim * chunk_compress_factor, latent_len)
        t = torch.rand(batch_size)
        text_ids = torch.randint(0, vocab_size, (batch_size, text_len))
        reference_latent = torch.randn(batch_size, latent_dim * chunk_compress_factor, ref_len)
        text_mask = torch.ones(batch_size, 1, text_len)
        latent_mask = torch.ones(batch_size, 1, latent_len)
        reference_mask = torch.ones(batch_size, 1, ref_len)

        # Forward
        velocity = model(
            noisy_latent=noisy_latent,
            t=t,
            text_ids=text_ids,
            reference_latent=reference_latent,
            text_mask=text_mask,
            latent_mask=latent_mask,
            reference_mask=reference_mask,
        )

        # Expected: [batch, latent_dim * chunk_compress_factor, latent_len]
        expected_shape = (batch_size, latent_dim * chunk_compress_factor, latent_len)
        assert velocity.shape == expected_shape, \
            f"Shape mismatch: {velocity.shape} != {expected_shape}"

        print(f"[OK] TTLModel output shape: {velocity.shape}")

    def test_inference_shape(self):
        """推論時の出力形状確認"""
        vocab_size = 1000
        batch_size = 2
        text_len = 50
        latent_dim = 24
        chunk_compress_factor = 6
        latent_len = 100
        ref_len = 80

        model = TTLModel(
            vocab_size=vocab_size,
            text_encoder_config={
                "char_emb_dim": 256,
                "convnext_num_layers": 6,
                "convnext_dilation_lst": [1, 1, 1, 1, 1, 1],
                "attn_n_layers": 4,
                "attn_n_heads": 4,
            },
            style_encoder_config={
                "ldim": latent_dim,
                "chunk_compress_factor": chunk_compress_factor,
                "convnext_num_layers": 6,
                "convnext_dilation_lst": [1, 1, 1, 1, 1, 1],
            },
            vector_field_config={
                "ldim": latent_dim,
                "chunk_compress_factor": chunk_compress_factor,
                "n_blocks": 4,
            },
        )
        model.eval()

        # Input
        text_ids = torch.randint(0, vocab_size, (batch_size, text_len))
        reference_latent = torch.randn(batch_size, latent_dim * chunk_compress_factor, ref_len)
        text_mask = torch.ones(batch_size, 1, text_len)
        reference_mask = torch.ones(batch_size, 1, ref_len)

        # Inference (Euler method)
        generated_latent = model.inference(
            text_ids=text_ids,
            reference_latent=reference_latent,
            latent_len=latent_len,
            text_mask=text_mask,
            reference_mask=reference_mask,
            total_steps=5,
        )

        # Expected: [batch, latent_dim * chunk_compress_factor, latent_len]
        expected_shape = (batch_size, latent_dim * chunk_compress_factor, latent_len)
        assert generated_latent.shape == expected_shape, \
            f"Shape mismatch: {generated_latent.shape} != {expected_shape}"

        print(f"[OK] TTLModel inference output shape: {generated_latent.shape}")


class TestFlowMatchingLoss:
    """FlowMatchingLossのテスト"""

    def test_forward(self):
        """順伝播の動作確認"""
        loss_fn = FlowMatchingLoss()

        batch_size = 2
        dim = 144
        seq_len = 100

        # Input
        v_pred = torch.randn(batch_size, dim, seq_len)
        v_target = torch.randn(batch_size, dim, seq_len)
        mask = torch.ones(batch_size, 1, seq_len)

        # Forward
        loss = loss_fn(v_pred, v_target, mask)

        # 損失は正の値
        assert loss > 0, f"Loss should be positive: {loss}"

        print(f"[OK] FlowMatchingLoss: {loss.item():.6f}")


class TestTTLLoss:
    """TTLLossのテスト"""

    def test_forward(self):
        """順伝播の動作確認"""
        loss_fn = TTLLoss()

        batch_size = 2
        ldim = 24
        chunk_compress_factor = 6
        latent_len = 100
        text_len = 50

        # Vector Field
        vector_field = VectorField(
            ldim=ldim,
            chunk_compress_factor=chunk_compress_factor,
            n_blocks=4,
        )

        # Input
        latent_gt = torch.randn(batch_size, ldim * chunk_compress_factor, latent_len)
        text_emb = torch.randn(batch_size, 256, text_len)
        style_emb = torch.randn(batch_size, 256)
        latent_mask = torch.ones(batch_size, 1, latent_len)
        text_mask = torch.ones(batch_size, 1, text_len)

        # Forward
        total_loss, metrics = loss_fn(
            vector_field=vector_field,
            latent_gt=latent_gt,
            text_emb=text_emb,
            style_emb=style_emb,
            latent_mask=latent_mask,
            text_mask=text_mask,
        )

        # 損失は正の値
        assert total_loss > 0, f"Loss should be positive: {total_loss}"
        assert "total_loss" in metrics
        assert "flow_matching_loss" in metrics

        print(f"[OK] TTLLoss: {total_loss.item():.6f}")
        print(f"  - Flow Matching Loss: {metrics['flow_matching_loss']:.6f}")


def run_all_tests():
    """全テストを実行"""
    print("=" * 60)
    print("Running TTL (Text-to-Latent) Unit Tests")
    print("=" * 60)
    print()

    # TextEncoder
    print("Testing TextEncoder...")
    test = TestTextEncoder()
    test.test_forward_shape()
    print()

    # StyleEncoder
    print("Testing StyleEncoder...")
    test = TestStyleEncoder()
    test.test_forward_shape()
    print()

    # CrossAttentionLARoPE
    print("Testing CrossAttentionLARoPE...")
    test = TestCrossAttentionLARoPE()
    test.test_forward_shape()
    print()

    # VectorField
    print("Testing VectorField...")
    test = TestVectorField()
    test.test_forward_shape()
    print()

    # TTLModel
    print("Testing TTLModel...")
    test = TestTTLModel()
    test.test_forward_shape()
    test.test_inference_shape()
    print()

    # FlowMatchingLoss
    print("Testing FlowMatchingLoss...")
    test = TestFlowMatchingLoss()
    test.test_forward()
    print()

    # TTLLoss
    print("Testing TTLLoss...")
    test = TestTTLLoss()
    test.test_forward()
    print()

    print("=" * 60)
    print("All tests passed! [SUCCESS]")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
