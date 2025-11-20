"""
Speech Autoencoder - 統合モデル

Encoder + Decoder の統合
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .encoder import EncoderWithMelProcessor
from .decoder import DecoderWithVocoder


class SpeechAutoencoder(nn.Module):
    """
    Speech Autoencoder

    音声波形 → 24次元の潜在表現 → 音声波形

    完全な再構成モデル
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
        encoder_ksz_init: int = 7,
        encoder_ksz: int = 7,
        encoder_num_layers: int = 10,
        encoder_dilation_lst: Optional[list] = None,
        encoder_intermediate_dim: int = 2048,
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
        vocoder_n_iter: int = 32,
    ):
        """
        引数:
            各コンポーネントの設定パラメータ
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.latent_dim = encoder_odim

        # Encoder (with Mel processor)
        self.encoder = EncoderWithMelProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            eps=eps,
            norm_mean=norm_mean,
            norm_std=norm_std,
            encoder_idim=encoder_idim,
            encoder_hdim=encoder_hdim,
            encoder_odim=encoder_odim,
            ksz_init=encoder_ksz_init,
            ksz=encoder_ksz,
            num_layers=encoder_num_layers,
            dilation_lst=encoder_dilation_lst,
            intermediate_dim=encoder_intermediate_dim,
        )

        # Decoder (with Vocoder)
        self.decoder = DecoderWithVocoder(
            decoder_idim=decoder_idim,
            decoder_hdim=decoder_hdim,
            decoder_odim=decoder_odim,
            decoder_ksz_init=decoder_ksz_init,
            decoder_ksz=decoder_ksz,
            decoder_num_layers=decoder_num_layers,
            decoder_dilation_lst=decoder_dilation_lst,
            decoder_intermediate_dim=decoder_intermediate_dim,
            head_idim=head_idim,
            head_hdim=head_hdim,
            head_odim=head_odim,
            head_ksz=head_ksz,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            sample_rate=sample_rate,
            n_iter=vocoder_n_iter,
        )

    def forward(
        self,
        wav: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播（完全な再構成）

        引数:
            wav: 入力音声波形 [batch, time]

        戻り値:
            (reconstructed_wav, latent, mel_original, mel_reconstructed)
        """
        # Encode
        latent, mel_original = self.encoder(wav)  # [batch, latent_dim, frames]

        # Decode
        reconstructed_wav, mel_reconstructed = self.decoder(latent)

        return reconstructed_wav, latent, mel_original, mel_reconstructed

    def encode(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodeのみ

        引数:
            wav: 入力音声波形 [batch, time]

        戻り値:
            (latent, mel): 潜在表現とMel-Spectrogram
        """
        return self.encoder(wav)

    def decode(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodeのみ

        引数:
            latent: 潜在表現 [batch, latent_dim, frames]

        戻り値:
            (wav, mel): 音声波形とMel-Spectrogram
        """
        return self.decoder(latent)

    @classmethod
    def from_config(cls, config: dict):
        """
        設定辞書からモデルを構築

        引数:
            config: 設定辞書（YAMLから読み込んだもの）

        戻り値:
            SpeechAutoencoderインスタンス
        """
        ae_config = config.get("model", {})

        # Extract parameters
        sample_rate = ae_config.get("sample_rate", 44100)
        n_delay = ae_config.get("n_delay", 0)
        base_chunk_size = ae_config.get("base_chunk_size", 512)
        ldim = ae_config.get("ldim", 24)

        # Encoder config
        encoder_config = ae_config.get("encoder", {})
        spec_config = encoder_config.get("spec_processor", {})

        # Decoder config
        decoder_config = ae_config.get("decoder", {})
        head_config = decoder_config.get("head", {})

        return cls(
            # Mel-Spectrogram
            sample_rate=spec_config.get("sample_rate", sample_rate),
            n_fft=spec_config.get("n_fft", 2048),
            win_length=spec_config.get("win_length", 2048),
            hop_length=spec_config.get("hop_length", 512),
            n_mels=spec_config.get("n_mels", 228),
            eps=spec_config.get("eps", 1e-5),
            norm_mean=spec_config.get("norm_mean", 0.0),
            norm_std=spec_config.get("norm_std", 1.0),
            # Encoder
            encoder_idim=encoder_config.get("idim", 1253),
            encoder_hdim=encoder_config.get("hdim", 512),
            encoder_odim=encoder_config.get("odim", ldim),
            encoder_ksz_init=encoder_config.get("ksz_init", 7),
            encoder_ksz=encoder_config.get("ksz", 7),
            encoder_num_layers=encoder_config.get("num_layers", 10),
            encoder_dilation_lst=encoder_config.get("dilation_lst"),
            encoder_intermediate_dim=encoder_config.get("intermediate_dim", 2048),
            # Decoder
            decoder_idim=decoder_config.get("idim", ldim),
            decoder_hdim=decoder_config.get("hdim", 512),
            decoder_odim=decoder_config.get("odim", 512),
            decoder_ksz_init=decoder_config.get("ksz_init", 7),
            decoder_ksz=decoder_config.get("ksz", 7),
            decoder_num_layers=decoder_config.get("num_layers", 10),
            decoder_dilation_lst=decoder_config.get("dilation_lst"),
            decoder_intermediate_dim=decoder_config.get("intermediate_dim", 2048),
            # Decoder Head
            head_idim=head_config.get("idim", 512),
            head_hdim=head_config.get("hdim", 2048),
            head_odim=head_config.get("odim", 512),
            head_ksz=head_config.get("ksz", 3),
        )
