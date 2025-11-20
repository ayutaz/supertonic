"""
Speech Autoencoder Decoder

24次元の潜在表現から音声波形を復元
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..common import InitialConvNeXt


class AcousticDecoder(nn.Module):
    """
    Acoustic Decoder

    24次元の潜在表現からMel-Spectrogramを復元

    アーキテクチャ:
    1. Input projection
    2. ConvNeXt Stack (10 layers)
    3. Decoder Head (3層MLP)
    """

    def __init__(
        self,
        idim: int = 24,
        hdim: int = 512,
        odim: int = 512,  # Decoder headへの入力次元
        ksz_init: int = 7,
        ksz: int = 7,
        num_layers: int = 10,
        dilation_lst: Optional[list] = None,
        intermediate_dim: int = 2048,
        dropout: float = 0.0,
    ):
        """
        引数:
            idim: 入力次元（潜在次元）
            hdim: 隠れ層次元
            odim: 出力次元（Decoder headへ）
            ksz_init: 初期カーネルサイズ
            ksz: ConvNeXtブロックのカーネルサイズ
            num_layers: ConvNeXtブロックの数
            dilation_lst: 各レイヤーのDilation
            intermediate_dim: ConvNeXt中間層の次元数
            dropout: Dropout率
        """
        super().__init__()

        if dilation_lst is None:
            dilation_lst = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]

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

        # Output projection
        self.output_proj = nn.Conv1d(hdim, odim, kernel_size=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            latent: 潜在表現 [batch, idim, frames]

        戻り値:
            デコード特徴 [batch, odim, frames]
        """
        # Input projection
        x = self.input_proj(latent)  # [batch, hdim, frames]

        # ConvNeXt processing
        x = self.convnext(x)  # [batch, hdim, frames]

        # Output projection
        decoded = self.output_proj(x)  # [batch, odim, frames]

        return decoded


class DecoderHead(nn.Module):
    """
    Decoder Head

    デコードされた特徴からMel-Spectrogramを生成

    tts.jsonの設定:
    - idim: 512
    - hdim: 2048
    - odim: 512
    - ksz: 3
    """

    def __init__(
        self,
        idim: int = 512,
        hdim: int = 2048,
        odim: int = 512,
        ksz: int = 3,
    ):
        """
        引数:
            idim: 入力次元
            hdim: 隠れ層次元
            odim: 出力次元
            ksz: カーネルサイズ
        """
        super().__init__()

        # 3層のConvolution
        padding = (ksz - 1) // 2

        self.conv1 = nn.Conv1d(idim, hdim, kernel_size=ksz, padding=padding)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(hdim, hdim, kernel_size=ksz, padding=padding)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv1d(hdim, odim, kernel_size=ksz, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            x: 入力 [batch, idim, frames]

        戻り値:
            出力 [batch, odim, frames]
        """
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        return x


class Vocoder(nn.Module):
    """
    Vocoder

    Mel-Spectrogramから音声波形を生成

    現在はGriffin-Limアルゴリズムを使用
    将来的にはNeural Vocoder (HiFi-GAN等) に置き換え可能
    """

    def __init__(
        self,
        n_mels: int = 228,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        sample_rate: int = 44100,
        n_iter: int = 32,
    ):
        """
        引数:
            n_mels: Melフィルターバンク数
            n_fft: FFTサイズ
            hop_length: ホップ長
            win_length: ウィンドウ長
            sample_rate: サンプリングレート
            n_iter: Griffin-Limのイテレーション数
        """
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_iter = n_iter

        # Mel filterbank (inverse)
        import librosa
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        mel_basis_inv = torch.pinverse(torch.from_numpy(mel_basis)).float()
        self.register_buffer("mel_basis_inv", mel_basis_inv)

        # Window
        self.register_buffer("window", torch.hann_window(win_length))

    def mel_to_linear(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Mel-Spectrogram → Linear Spectrogram

        引数:
            mel: Mel-Spectrogram [batch, n_mels, frames]

        戻り値:
            Linear Spectrogram [batch, n_fft//2+1, frames]
        """
        # Denormalize (if normalized during training)
        # mel = mel * norm_std + norm_mean
        # mel = torch.exp(mel) - eps

        # Mel to Linear
        linear = torch.matmul(self.mel_basis_inv, mel)

        return linear

    def griffin_lim(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Griffin-Lim Algorithm

        引数:
            mag: Magnitude Spectrogram [batch, freq, frames]

        戻り値:
            Waveform [batch, time]
        """
        batch_size = mag.size(0)

        # ランダム位相で初期化
        phase = torch.randn_like(mag) * 2 * torch.pi
        complex_spec = mag * torch.exp(1j * phase)

        # イテレーション
        for _ in range(self.n_iter):
            # ISTFT
            wav = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=False,
            )

            # STFT
            complex_spec = torch.stft(
                wav,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )

            # Magnitudeを元のものに置き換え
            phase = torch.angle(complex_spec)
            complex_spec = mag * torch.exp(1j * phase)

        # 最終的な波形
        wav = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False,
        )

        return wav

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            mel: Mel-Spectrogram [batch, n_mels, frames]

        戻り値:
            Waveform [batch, time]
        """
        # Mel to Linear
        linear = self.mel_to_linear(mel)

        # Griffin-Lim
        wav = self.griffin_lim(linear)

        return wav


class DecoderWithVocoder(nn.Module):
    """
    Decoder + Head + Vocoder の統合

    潜在表現から音声波形を直接生成
    """

    def __init__(
        self,
        # Decoder設定
        decoder_idim: int = 24,
        decoder_hdim: int = 512,
        decoder_odim: int = 512,
        decoder_ksz_init: int = 7,
        decoder_ksz: int = 7,
        decoder_num_layers: int = 10,
        decoder_dilation_lst: Optional[list] = None,
        decoder_intermediate_dim: int = 2048,
        # Decoder Head設定
        head_idim: int = 512,
        head_hdim: int = 2048,
        head_odim: int = 512,
        head_ksz: int = 3,
        # Vocoder設定
        n_mels: int = 228,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        sample_rate: int = 44100,
        n_iter: int = 32,
    ):
        """
        引数:
            Decoder、Decoder Head、Vocoderの各設定
        """
        super().__init__()

        self.decoder = AcousticDecoder(
            idim=decoder_idim,
            hdim=decoder_hdim,
            odim=decoder_odim,
            ksz_init=decoder_ksz_init,
            ksz=decoder_ksz,
            num_layers=decoder_num_layers,
            dilation_lst=decoder_dilation_lst,
            intermediate_dim=decoder_intermediate_dim,
        )

        self.head = DecoderHead(
            idim=head_idim,
            hdim=head_hdim,
            odim=head_odim,
            ksz=head_ksz,
        )

        # Head outputからMel-Spectrogramへの変換
        self.to_mel = nn.Conv1d(head_odim, n_mels, kernel_size=1)

        self.vocoder = Vocoder(
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            sample_rate=sample_rate,
            n_iter=n_iter,
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播

        引数:
            latent: 潜在表現 [batch, idim, frames]

        戻り値:
            (wav, mel): 音声波形とMel-Spectrogram
        """
        # Decoder
        decoded = self.decoder(latent)  # [batch, decoder_odim, frames]

        # Decoder Head
        features = self.head(decoded)  # [batch, head_odim, frames]

        # To Mel-Spectrogram
        mel = self.to_mel(features)  # [batch, n_mels, frames]

        # Vocoder
        wav = self.vocoder(mel)  # [batch, time]

        return wav, mel
