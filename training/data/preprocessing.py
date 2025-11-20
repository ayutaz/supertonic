"""
Audio preprocessing utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from typing import Optional, Tuple, Union


class MelSpectrogramExtractor(nn.Module):
    """
    Mel-Spectrogram抽出器

    音声波形からMel-Spectrogramを抽出
    論文の設定に基づく
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
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
            fmin: 最小周波数
            fmax: 最大周波数 (Noneの場合は sample_rate / 2)
            eps: 数値安定化のための定数
            norm_mean: 正規化の平均
            norm_std: 正規化の標準偏差
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2
        self.eps = eps
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # Mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=self.fmax,
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()

        # Hann window
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        順伝播

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
        )

        # Power spectrogram
        spec = torch.abs(spec) ** 2

        # Mel filterbank
        mel_basis = self.mel_basis.to(spec.device)
        mel = torch.matmul(mel_basis, spec)

        # Log scale
        mel = torch.log(mel + self.eps)

        # Normalize
        if self.norm_std != 1.0 or self.norm_mean != 0.0:
            mel = (mel - self.norm_mean) / self.norm_std

        return mel


class AudioPreprocessor:
    """
    音声前処理クラス

    音声ファイルの読み込み、リサンプリング、正規化などを行う
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        normalize: bool = True,
        trim_silence: bool = False,
        top_db: float = 60.0,
    ):
        """
        引数:
            sample_rate: 目標サンプリングレート
            normalize: 正規化を行うか
            trim_silence: 無音部分を削除するか
            top_db: 無音判定の閾値 (dB)
        """
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.top_db = top_db

    def load_audio(
        self,
        path: str,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        音声ファイルを読み込み

        引数:
            path: ファイルパス
            offset: 開始位置 (秒)
            duration: 読み込む長さ (秒、Noneの場合は全体)

        戻り値:
            (wav, sample_rate): 音声波形とサンプリングレート
        """
        # librosaで読み込み
        wav, sr = librosa.load(
            path,
            sr=self.sample_rate,
            mono=True,
            offset=offset,
            duration=duration,
        )

        return wav, sr

    def process(
        self,
        wav: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        音声波形を前処理

        引数:
            wav: 音声波形
            sample_rate: 元のサンプリングレート (Noneの場合はリサンプリングしない)

        戻り値:
            前処理後の音声波形
        """
        # リサンプリング
        if sample_rate is not None and sample_rate != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=self.sample_rate)

        # 無音削除
        if self.trim_silence:
            wav, _ = librosa.effects.trim(wav, top_db=self.top_db)

        # 正規化
        if self.normalize:
            wav = wav / (np.abs(wav).max() + 1e-8)

        return wav

    def load_and_process(
        self,
        path: str,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        音声ファイルを読み込んで前処理

        引数:
            path: ファイルパス
            offset: 開始位置 (秒)
            duration: 読み込む長さ (秒)

        戻り値:
            前処理後の音声波形
        """
        wav, sr = self.load_audio(path, offset, duration)
        wav = self.process(wav, sr)
        return wav


def dynamic_range_compression(
    x: torch.Tensor,
    C: float = 1.0,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    """
    Dynamic Range Compression (DRC)

    引数:
        x: 入力
        C: 圧縮係数
        clip_val: クリップ値

    戻り値:
        圧縮後の値
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(
    x: torch.Tensor,
    C: float = 1.0,
) -> torch.Tensor:
    """
    Dynamic Range Decompression

    引数:
        x: 入力
        C: 圧縮係数

    戻り値:
        解凍後の値
    """
    return torch.exp(x) / C


def griffin_lim(
    mag: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int = 32,
) -> torch.Tensor:
    """
    Griffin-Lim Algorithm

    Magnitude SpectrogramからWaveformを復元

    引数:
        mag: Magnitude Spectrogram [batch, freq, frames]
        n_fft: FFTサイズ
        hop_length: ホップ長
        win_length: ウィンドウ長
        n_iter: イテレーション数

    戻り値:
        復元された波形 [batch, time]
    """
    # ランダム位相で初期化
    phase = torch.randn_like(mag) * 2 * np.pi
    complex_spec = mag * torch.exp(1j * phase)

    # イテレーション
    window = torch.hann_window(win_length).to(mag.device)

    for _ in range(n_iter):
        # ISTFT
        wav = torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

        # STFT
        complex_spec = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

        # Magnitudeを元のものに置き換え
        phase = torch.angle(complex_spec)
        complex_spec = mag * torch.exp(1j * phase)

    # 最終的な波形
    wav = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )

    return wav
