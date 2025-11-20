"""
Speech Autoencoder Loss Functions

- Multi-Scale STFT Loss
- Mel-Spectrogram Loss
- Feature Matching Loss
- Adversarial Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-Scale STFT Loss

    複数のFFTサイズでSTFTを計算し、Magnitude損失とPhase損失を計算

    論文の設定: [2048, 1024, 512, 256, 128]
    """

    def __init__(
        self,
        fft_sizes: List[int] = [2048, 1024, 512, 256, 128],
        hop_sizes: List[int] = None,
        win_sizes: List[int] = None,
    ):
        """
        引数:
            fft_sizes: FFTサイズのリスト
            hop_sizes: ホップサイズのリスト (Noneの場合はfft_size // 4)
            win_sizes: ウィンドウサイズのリスト (Noneの場合はfft_size)
        """
        super().__init__()

        self.fft_sizes = fft_sizes

        if hop_sizes is None:
            self.hop_sizes = [f // 4 for f in fft_sizes]
        else:
            self.hop_sizes = hop_sizes

        if win_sizes is None:
            self.win_sizes = fft_sizes
        else:
            self.win_sizes = win_sizes

        # Register windows
        for i, (fft_size, win_size) in enumerate(zip(fft_sizes, self.win_sizes)):
            window = torch.hann_window(win_size)
            self.register_buffer(f"window_{i}", window)

    def stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_size: int, window: torch.Tensor) -> torch.Tensor:
        """
        STFT計算

        引数:
            x: 入力波形 [batch, time]
            fft_size: FFTサイズ
            hop_size: ホップサイズ
            win_size: ウィンドウサイズ
            window: ウィンドウ関数

        戻り値:
            STFT [batch, freq, frames]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # STFT
        spec = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window,
            return_complex=True,
            center=True,
            normalized=False,
        )

        return spec

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播

        引数:
            y_pred: 予測波形 [batch, time]
            y_true: 真の波形 [batch, time]

        戻り値:
            (mag_loss, phase_loss): Magnitude損失とPhase損失
        """
        mag_loss = 0.0
        phase_loss = 0.0

        for i, (fft_size, hop_size, win_size) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_sizes)):
            window = getattr(self, f"window_{i}")

            # STFT
            spec_pred = self.stft(y_pred, fft_size, hop_size, win_size, window)
            spec_true = self.stft(y_true, fft_size, hop_size, win_size, window)

            # Magnitude
            mag_pred = torch.abs(spec_pred)
            mag_true = torch.abs(spec_true)

            # Magnitude loss (L1)
            mag_loss += F.l1_loss(mag_pred, mag_true)

            # Phase loss (cosine distance)
            phase_pred = torch.angle(spec_pred)
            phase_true = torch.angle(spec_true)

            # Cosine similarity of phase
            phase_diff = torch.cos(phase_pred - phase_true)
            phase_loss += (1 - phase_diff.mean())

        # Average over scales
        mag_loss /= len(self.fft_sizes)
        phase_loss /= len(self.fft_sizes)

        return mag_loss, phase_loss


class MelSpectrogramLoss(nn.Module):
    """
    Mel-Spectrogram Loss

    Mel-SpectrogramのL1損失

    重み: 45.0 (論文の設定)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        n_mels: int = 228,
    ):
        """
        引数:
            sample_rate: サンプリングレート
            n_fft: FFTサイズ
            hop_length: ホップ長
            win_length: ウィンドウ長
            n_mels: Melフィルターバンク数
        """
        super().__init__()

        # Mel filterbank
        import librosa
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("window", torch.hann_window(win_length))

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def compute_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Mel-Spectrogramを計算

        引数:
            wav: 音声波形 [batch, time]

        戻り値:
            Mel-Spectrogram [batch, n_mels, frames]
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # STFT
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

        # Power spectrogram
        spec = torch.abs(spec) ** 2

        # Mel filterbank
        mel = torch.matmul(self.mel_basis, spec)

        # Log scale
        mel = torch.log(mel + 1e-5)

        return mel

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            y_pred: 予測波形 [batch, time]
            y_true: 真の波形 [batch, time]

        戻り値:
            Mel-Spectrogram Loss
        """
        mel_pred = self.compute_mel(y_pred)
        mel_true = self.compute_mel(y_true)

        loss = F.l1_loss(mel_pred, mel_true)

        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss

    Discriminatorの中間層特徴量のマッチング

    重み: 2.0 (論文の設定)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features_real: List[List[torch.Tensor]],
        features_fake: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            features_real: 真の音声のDiscriminator中間層特徴
                [[scale0_layer0, scale0_layer1, ...], [scale1_layer0, ...], ...]
            features_fake: 生成音声のDiscriminator中間層特徴

        戻り値:
            Feature Matching Loss
        """
        loss = 0.0
        count = 0

        for feats_real_scale, feats_fake_scale in zip(features_real, features_fake):
            for feat_real, feat_fake in zip(feats_real_scale, feats_fake_scale):
                loss += F.l1_loss(feat_fake, feat_real.detach())
                count += 1

        loss = loss / count

        return loss


class AdversarialLoss(nn.Module):
    """
    Adversarial Loss (Generator側)

    Hinge Lossを使用

    重み: 1.0 (論文の設定)
    """

    def __init__(self, loss_type: str = "hinge"):
        """
        引数:
            loss_type: 損失タイプ ("hinge", "mse", "bce")
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, disc_fake_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        順伝播

        引数:
            disc_fake_outputs: Discriminatorの出力（生成音声）
                [scale0_output, scale1_output, ...]

        戻り値:
            Adversarial Loss
        """
        loss = 0.0

        for disc_output in disc_fake_outputs:
            if self.loss_type == "hinge":
                # Hinge Loss: min -D(G(z))
                loss += -disc_output.mean()
            elif self.loss_type == "mse":
                # MSE Loss
                loss += F.mse_loss(disc_output, torch.ones_like(disc_output))
            elif self.loss_type == "bce":
                # BCE Loss
                loss += F.binary_cross_entropy_with_logits(
                    disc_output, torch.ones_like(disc_output)
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss = loss / len(disc_fake_outputs)

        return loss


class DiscriminatorLoss(nn.Module):
    """
    Discriminator Loss

    真の音声と生成音声を識別

    重み: 1.0 (論文の設定)
    """

    def __init__(self, loss_type: str = "hinge"):
        """
        引数:
            loss_type: 損失タイプ ("hinge", "mse", "bce")
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_fake_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            disc_real_outputs: Discriminatorの出力（真の音声）
            disc_fake_outputs: Discriminatorの出力（生成音声）

        戻り値:
            Discriminator Loss
        """
        loss = 0.0

        for disc_real, disc_fake in zip(disc_real_outputs, disc_fake_outputs):
            if self.loss_type == "hinge":
                # Hinge Loss: max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
                loss += F.relu(1 - disc_real).mean() + F.relu(1 + disc_fake).mean()
            elif self.loss_type == "mse":
                # MSE Loss
                loss += F.mse_loss(disc_real, torch.ones_like(disc_real))
                loss += F.mse_loss(disc_fake, torch.zeros_like(disc_fake))
            elif self.loss_type == "bce":
                # BCE Loss
                loss += F.binary_cross_entropy_with_logits(
                    disc_real, torch.ones_like(disc_real)
                )
                loss += F.binary_cross_entropy_with_logits(
                    disc_fake, torch.zeros_like(disc_fake)
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss = loss / len(disc_real_outputs)

        return loss
