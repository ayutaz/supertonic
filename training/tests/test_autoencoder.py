"""
Speech Autoencoder Unit Tests

モデルの形状確認と基本動作テスト
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from models.autoencoder import (
    SpeechAutoencoder,
    MelSpectrogramProcessor,
    AcousticEncoder,
    AcousticDecoder,
    Vocoder,
    MultiScaleDiscriminator,
)


class TestMelSpectrogramProcessor:
    """MelSpectrogramProcessorのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        processor = MelSpectrogramProcessor(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=228,
        )

        # Input: [batch=2, time=44100] (1秒の音声)
        wav = torch.randn(2, 44100)

        # Forward
        mel = processor(wav)

        # Expected: [batch=2, n_mels*3=684, frames≈86]
        assert mel.shape[0] == 2, f"Batch size mismatch: {mel.shape[0]} != 2"
        assert mel.shape[1] == 228 * 3, f"Mel channels mismatch: {mel.shape[1]} != 684"
        # Frames ≈ 44100 / 512 ≈ 86
        assert 80 <= mel.shape[2] <= 90, f"Frames out of range: {mel.shape[2]}"

        print(f"[OK] MelSpectrogramProcessor output shape: {mel.shape}")


class TestAcousticEncoder:
    """AcousticEncoderのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        encoder = AcousticEncoder(
            idim=684,  # 228*3
            hdim=512,
            odim=24,
            num_layers=10,
        )

        # Input: [batch=2, idim=684, frames=86]
        mel = torch.randn(2, 684, 86)

        # Forward
        latent = encoder(mel)

        # Expected: [batch=2, odim=24, frames=86]
        assert latent.shape == (2, 24, 86), f"Shape mismatch: {latent.shape} != (2, 24, 86)"

        print(f"[OK] AcousticEncoder output shape: {latent.shape}")


class TestAcousticDecoder:
    """AcousticDecoderのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        decoder = AcousticDecoder(
            idim=24,
            hdim=512,
            odim=512,
            num_layers=10,
        )

        # Input: [batch=2, idim=24, frames=86]
        latent = torch.randn(2, 24, 86)

        # Forward
        decoded = decoder(latent)

        # Expected: [batch=2, odim=512, frames=86]
        assert decoded.shape == (2, 512, 86), f"Shape mismatch: {decoded.shape} != (2, 512, 86)"

        print(f"[OK] AcousticDecoder output shape: {decoded.shape}")


class TestVocoder:
    """Vocoderのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        vocoder = Vocoder(
            n_mels=228,
            n_fft=2048,
            hop_length=512,
            sample_rate=44100,
            n_iter=4,  # テストでは少なめ
        )

        # Input: [batch=2, n_mels=228, frames=86]
        mel = torch.randn(2, 228, 86)

        # Forward
        wav = vocoder(mel)

        # Expected: [batch=2, time≈44032] (86 frames * 512 hop)
        assert wav.shape[0] == 2, f"Batch size mismatch: {wav.shape[0]} != 2"
        expected_time = 86 * 512
        assert abs(wav.shape[1] - expected_time) < 1000, f"Time mismatch: {wav.shape[1]} != ~{expected_time}"

        print(f"[OK] Vocoder output shape: {wav.shape}")


class TestSpeechAutoencoder:
    """SpeechAutoencoderのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        model = SpeechAutoencoder(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=228,
            encoder_idim=684,
            encoder_hdim=256,  # 小さめ
            encoder_odim=24,
            encoder_num_layers=4,  # 少なめ
            encoder_dilation_lst=[1, 1, 1, 1],
            decoder_hdim=256,
            decoder_num_layers=4,
            decoder_dilation_lst=[1, 2, 4, 1],
            head_hdim=512,
            vocoder_n_iter=4,
        )

        # Input: [batch=2, time=44100] (1秒の音声)
        wav = torch.randn(2, 44100)

        # Forward
        reconstructed_wav, latent, mel_original, mel_reconstructed = model(wav)

        # Check shapes
        assert reconstructed_wav.shape[0] == 2, f"Batch size mismatch"
        assert latent.shape[0] == 2 and latent.shape[1] == 24, f"Latent shape mismatch: {latent.shape}"
        assert mel_original.shape[0] == 2 and mel_original.shape[1] == 684, f"Mel original shape mismatch"
        assert mel_reconstructed.shape[0] == 2 and mel_reconstructed.shape[1] == 228, f"Mel reconstructed shape mismatch"

        print(f"[OK] SpeechAutoencoder output shapes:")
        print(f"  - reconstructed_wav: {reconstructed_wav.shape}")
        print(f"  - latent: {latent.shape}")
        print(f"  - mel_original: {mel_original.shape}")
        print(f"  - mel_reconstructed: {mel_reconstructed.shape}")

    def test_encode_decode_separately(self):
        """Encode/Decodeを分離して実行"""
        model = SpeechAutoencoder(
            encoder_hdim=256,
            encoder_num_layers=4,
            encoder_dilation_lst=[1, 1, 1, 1],
            decoder_hdim=256,
            decoder_num_layers=4,
            decoder_dilation_lst=[1, 2, 4, 1],
            vocoder_n_iter=4,
        )

        wav = torch.randn(2, 44100)

        # Encode
        latent, mel = model.encode(wav)
        assert latent.shape[1] == 24, f"Latent dim mismatch"
        print(f"[OK] Encode: {wav.shape} -> {latent.shape}")

        # Decode
        reconstructed_wav, mel_reconstructed = model.decode(latent)
        assert reconstructed_wav.shape[0] == 2, f"Batch size mismatch"
        print(f"[OK] Decode: {latent.shape} -> {reconstructed_wav.shape}")


class TestMultiScaleDiscriminator:
    """MultiScaleDiscriminatorのテスト"""

    def test_forward_shape(self):
        """順伝播の出力形状確認"""
        discriminator = MultiScaleDiscriminator(
            in_channels=1,
            channels=64,
            num_scales=3,
        )

        # Input: [batch=2, channels=1, time=44100]
        wav = torch.randn(2, 1, 44100)

        # Forward
        scores, features_list = discriminator(wav)

        # Expected: 3 scales
        assert len(scores) == 3, f"Number of scales mismatch: {len(scores)} != 3"
        assert len(features_list) == 3, f"Number of feature lists mismatch"

        print(f"[OK] MultiScaleDiscriminator outputs:")
        for i, (score, features) in enumerate(zip(scores, features_list)):
            print(f"  Scale {i}: score shape={score.shape}, {len(features)} feature maps")


def run_all_tests():
    """全テストを実行"""
    print("=" * 60)
    print("Running Speech Autoencoder Unit Tests")
    print("=" * 60)
    print()

    # MelSpectrogramProcessor
    print("Testing MelSpectrogramProcessor...")
    test = TestMelSpectrogramProcessor()
    test.test_forward_shape()
    print()

    # AcousticEncoder
    print("Testing AcousticEncoder...")
    test = TestAcousticEncoder()
    test.test_forward_shape()
    print()

    # AcousticDecoder
    print("Testing AcousticDecoder...")
    test = TestAcousticDecoder()
    test.test_forward_shape()
    print()

    # Vocoder
    print("Testing Vocoder...")
    test = TestVocoder()
    test.test_forward_shape()
    print()

    # SpeechAutoencoder
    print("Testing SpeechAutoencoder...")
    test = TestSpeechAutoencoder()
    test.test_forward_shape()
    test.test_encode_decode_separately()
    print()

    # MultiScaleDiscriminator
    print("Testing MultiScaleDiscriminator...")
    test = TestMultiScaleDiscriminator()
    test.test_forward_shape()
    print()

    print("=" * 60)
    print("All tests passed! [SUCCESS]")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
