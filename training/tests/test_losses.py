"""
Loss Functions Unit Tests

損失関数の動作確認
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from losses import (
    MultiScaleSTFTLoss,
    MelSpectrogramLoss,
    FeatureMatchingLoss,
    AdversarialLoss,
    DiscriminatorLoss,
)


class TestMultiScaleSTFTLoss:
    """MultiScaleSTFTLossのテスト"""

    def test_forward(self):
        """順伝播の動作確認"""
        loss_fn = MultiScaleSTFTLoss(
            fft_sizes=[2048, 1024, 512],
        )

        # 同じ音声
        y_pred = torch.randn(2, 44100)
        y_true = y_pred.clone()

        # Forward
        mag_loss, phase_loss = loss_fn(y_pred, y_true)

        # 同じ音声なので損失は0に近いはず
        assert mag_loss < 0.1, f"Magnitude loss too high: {mag_loss}"
        assert phase_loss < 0.1, f"Phase loss too high: {phase_loss}"

        print(f"[OK] MultiScaleSTFTLoss (same audio):")
        print(f"  - Magnitude loss: {mag_loss.item():.6f}")
        print(f"  - Phase loss: {phase_loss.item():.6f}")

        # 異なる音声
        y_pred = torch.randn(2, 44100)
        y_true = torch.randn(2, 44100)

        mag_loss, phase_loss = loss_fn(y_pred, y_true)

        # 異なる音声なので損失は大きいはず
        assert mag_loss > 0.5, f"Magnitude loss too low: {mag_loss}"

        print(f"[OK] MultiScaleSTFTLoss (different audio):")
        print(f"  - Magnitude loss: {mag_loss.item():.6f}")
        print(f"  - Phase loss: {phase_loss.item():.6f}")


class TestMelSpectrogramLoss:
    """MelSpectrogramLossのテスト"""

    def test_forward(self):
        """順伝播の動作確認"""
        loss_fn = MelSpectrogramLoss(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=228,
        )

        # 同じ音声
        y_pred = torch.randn(2, 44100)
        y_true = y_pred.clone()

        # Forward
        loss = loss_fn(y_pred, y_true)

        # 同じ音声なので損失は0に近いはず
        assert loss < 0.1, f"Loss too high: {loss}"

        print(f"[OK] MelSpectrogramLoss (same audio): {loss.item():.6f}")

        # 異なる音声
        y_pred = torch.randn(2, 44100)
        y_true = torch.randn(2, 44100)

        loss = loss_fn(y_pred, y_true)

        # 異なる音声なので損失は大きいはず
        assert loss > 0.5, f"Loss too low: {loss}"

        print(f"[OK] MelSpectrogramLoss (different audio): {loss.item():.6f}")


class TestFeatureMatchingLoss:
    """FeatureMatchingLossのテスト"""

    def test_forward(self):
        """順伝播の動作確認"""
        loss_fn = FeatureMatchingLoss()

        # 模擬的な特徴量
        # 3スケール、各スケール5層
        features_real = [
            [torch.randn(2, 64, 100) for _ in range(5)] for _ in range(3)
        ]
        features_fake = [
            [torch.randn(2, 64, 100) for _ in range(5)] for _ in range(3)
        ]

        # Forward
        loss = loss_fn(features_real, features_fake)

        # 損失は正の値
        assert loss > 0, f"Loss should be positive: {loss}"

        print(f"[OK] FeatureMatchingLoss: {loss.item():.6f}")


class TestAdversarialLoss:
    """AdversarialLossのテスト"""

    def test_forward(self):
        """順伝播の動作確認"""
        loss_fn = AdversarialLoss(loss_type="hinge")

        # Discriminatorの出力（3スケール）
        disc_fake_outputs = [
            torch.randn(2, 1, 100),
            torch.randn(2, 1, 50),
            torch.randn(2, 1, 25),
        ]

        # Forward
        loss = loss_fn(disc_fake_outputs)

        print(f"[OK] AdversarialLoss: {loss.item():.6f}")


class TestDiscriminatorLoss:
    """DiscriminatorLossのテスト"""

    def test_forward(self):
        """順伝播の動作確認"""
        loss_fn = DiscriminatorLoss(loss_type="hinge")

        # Discriminatorの出力（3スケール）
        # Real: 正の値
        disc_real_outputs = [
            torch.ones(2, 1, 100),
            torch.ones(2, 1, 50),
            torch.ones(2, 1, 25),
        ]

        # Fake: 負の値
        disc_fake_outputs = [
            -torch.ones(2, 1, 100),
            -torch.ones(2, 1, 50),
            -torch.ones(2, 1, 25),
        ]

        # Forward
        loss = loss_fn(disc_real_outputs, disc_fake_outputs)

        # Hinge lossは0になるはず（理想的なケース）
        assert loss < 0.1, f"Loss should be near 0: {loss}"

        print(f"[OK] DiscriminatorLoss (ideal case): {loss.item():.6f}")

        # 逆の場合（学習が必要）
        disc_real_outputs = [
            -torch.ones(2, 1, 100),
            -torch.ones(2, 1, 50),
            -torch.ones(2, 1, 25),
        ]

        disc_fake_outputs = [
            torch.ones(2, 1, 100),
            torch.ones(2, 1, 50),
            torch.ones(2, 1, 25),
        ]

        loss = loss_fn(disc_real_outputs, disc_fake_outputs)

        # 損失は大きいはず
        assert loss > 1.0, f"Loss should be high: {loss}"

        print(f"[OK] DiscriminatorLoss (need training): {loss.item():.6f}")


def run_all_tests():
    """全テストを実行"""
    print("=" * 60)
    print("Running Loss Functions Unit Tests")
    print("=" * 60)
    print()

    # MultiScaleSTFTLoss
    print("Testing MultiScaleSTFTLoss...")
    test = TestMultiScaleSTFTLoss()
    test.test_forward()
    print()

    # MelSpectrogramLoss
    print("Testing MelSpectrogramLoss...")
    test = TestMelSpectrogramLoss()
    test.test_forward()
    print()

    # FeatureMatchingLoss
    print("Testing FeatureMatchingLoss...")
    test = TestFeatureMatchingLoss()
    test.test_forward()
    print()

    # AdversarialLoss
    print("Testing AdversarialLoss...")
    test = TestAdversarialLoss()
    test.test_forward()
    print()

    # DiscriminatorLoss
    print("Testing DiscriminatorLoss...")
    test = TestDiscriminatorLoss()
    test.test_forward()
    print()

    print("=" * 60)
    print("All tests passed! [SUCCESS]")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
