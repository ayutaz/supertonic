"""
Speech Autoencoder Encoder

音声波形からMel-Spectrogramを抽出し、24次元の潜在表現に圧縮
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..common import ConvNeXtStack, InitialConvNeXt
from ...data.preprocessing import MelSpectrogramExtractor


class MelSpectrogramProcessor(nn.Module):
    """
    Mel-Spectrogram抽出とDelta特徴量の計算

    論文の設定:
    - n_fft: 2048
    - hop_length: 512
    - n_mels: 228
    - Delta, Delta-Delta特徴量を追加
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        eps: float = 1e-5,
        norm_mean: float = 0.0,
        norm_std: float = 1.0,
    ):
        """
        引数:
            sample_rate: サンプリングレート
            n_fft: FFTサイズ
            win_length: ウィンドウ長
            hop_length: ホップ長
            n_mels: Melフィルターバンク数
            eps: 数値安定化のための定数
            norm_mean: 正規化の平均
            norm_std: 正規化の標準偏差
        """
        super().__init__()

        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            eps=eps,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    def compute_delta(self, x: torch.Tensor, order: int = 1) -> torch.Tensor:
        """
        Delta特徴量を計算

        引数:
            x: 入力 [batch, n_mels, frames]
            order: Delta次数 (1: Delta, 2: Delta-Delta)

        戻り値:
            Delta特徴量 [batch, n_mels, frames]
        """
        if order == 0:
            return x

        # 簡単な差分計算（隣接フレームとの差）
        delta = x[:, :, 1:] - x[:, :, :-1]
        # 最初のフレームは0で埋める
        delta = F.pad(delta, (1, 0), mode="constant", value=0)

        if order > 1:
            return self.compute_delta(delta, order - 1)

        return delta

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            wav: 音声波形 [batch, time]

        戻り値:
            拡張Mel-Spectrogram [batch, n_mels*3, frames]
            (Mel + Delta + Delta-Delta)
        """
        # Mel-Spectrogram抽出
        mel = self.mel_extractor(wav)  # [batch, n_mels, frames]

        # Delta特徴量
        delta = self.compute_delta(mel, order=1)  # [batch, n_mels, frames]
        delta_delta = self.compute_delta(mel, order=2)  # [batch, n_mels, frames]

        # 結合
        mel_extended = torch.cat([mel, delta, delta_delta], dim=1)  # [batch, n_mels*3, frames]

        return mel_extended


class AcousticEncoder(nn.Module):
    """
    Acoustic Encoder

    拡張Mel-Spectrogramから24次元の潜在表現に圧縮

    アーキテクチャ:
    1. 初期Conv (kernel=7)
    2. ConvNeXt Stack (10 layers)
    3. Projection to latent (24-dim)
    """

    def __init__(
        self,
        idim: int = 1253,  # 228*3 + context = 684, but tts.json says 1253
        hdim: int = 512,
        odim: int = 24,
        ksz_init: int = 7,
        ksz: int = 7,
        num_layers: int = 10,
        dilation_lst: Optional[list] = None,
        intermediate_dim: int = 2048,
        dropout: float = 0.0,
    ):
        """
        引数:
            idim: 入力次元
            hdim: 隠れ層次元
            odim: 出力次元（潜在次元）
            ksz_init: 初期カーネルサイズ
            ksz: ConvNeXtブロックのカーネルサイズ
            num_layers: ConvNeXtブロックの数
            dilation_lst: 各レイヤーのDilation
            intermediate_dim: ConvNeXt中間層の次元数
            dropout: Dropout率
        """
        super().__init__()

        if dilation_lst is None:
            dilation_lst = [1] * num_layers

        # Input projection
        self.input_proj = nn.Conv1d(idim, hdim, kernel_size=1)

        # ConvNeXt Stack
        self.convnext = InitialConvNeXt(
            in_channels=hdim,
            out_channels=hdim,
            ksz_init=ksz_init,
            ksz=ksz,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            dilation_lst=dilation_lst,
            dropout=dropout,
        )

        # Output projection to latent
        self.output_proj = nn.Conv1d(hdim, odim, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            mel: 拡張Mel-Spectrogram [batch, idim, frames]

        戻り値:
            潜在表現 [batch, odim, frames]
        """
        # Input projection
        x = self.input_proj(mel)  # [batch, hdim, frames]

        # ConvNeXt processing
        x = self.convnext(x)  # [batch, hdim, frames]

        # Output projection
        latent = self.output_proj(x)  # [batch, odim, frames]

        return latent


class EncoderWithMelProcessor(nn.Module):
    """
    Mel-Spectrogram処理とEncoderの統合

    音声波形から直接潜在表現を抽出
    """

    def __init__(
        self,
        # Mel-Spectrogram設定
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        eps: float = 1e-5,
        norm_mean: float = 0.0,
        norm_std: float = 1.0,
        # Encoder設定
        encoder_idim: int = 1253,
        encoder_hdim: int = 512,
        encoder_odim: int = 24,
        ksz_init: int = 7,
        ksz: int = 7,
        num_layers: int = 10,
        dilation_lst: Optional[list] = None,
        intermediate_dim: int = 2048,
        dropout: float = 0.0,
    ):
        """
        引数:
            Mel-Spectrogram処理の引数とEncoderの引数を統合
        """
        super().__init__()

        self.mel_processor = MelSpectrogramProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            eps=eps,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.encoder = AcousticEncoder(
            idim=encoder_idim,
            hdim=encoder_hdim,
            odim=encoder_odim,
            ksz_init=ksz_init,
            ksz=ksz,
            num_layers=num_layers,
            dilation_lst=dilation_lst,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
        )

        # Note: encoder_idimが1253なのに対し、mel_processorの出力は228*3=684
        # これはtts.jsonの設定に従っているが、実際には追加の前処理が必要かもしれない
        # （コンテキストフレーム、その他の特徴量など）
        # 簡易的には、projectionで調整する
        if n_mels * 3 != encoder_idim:
            self.feature_projection = nn.Conv1d(n_mels * 3, encoder_idim, kernel_size=1)
        else:
            self.feature_projection = nn.Identity()

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播

        引数:
            wav: 音声波形 [batch, time]

        戻り値:
            (latent, mel): 潜在表現とMel-Spectrogram
        """
        # Mel-Spectrogram抽出
        mel = self.mel_processor(wav)  # [batch, n_mels*3, frames]

        # Feature projection (if needed)
        mel_projected = self.feature_projection(mel)  # [batch, encoder_idim, frames]

        # Encoding
        latent = self.encoder(mel_projected)  # [batch, odim, frames]

        return latent, mel
