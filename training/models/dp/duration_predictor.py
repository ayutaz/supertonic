"""
Duration Predictor

テキストと音声スタイルから発話全体の長さを予測

アーキテクチャ (tts.json の dp):
1. Sentence Encoder: テキスト → センテンスエンベディング
2. Style Encoder: 参照音声 → スタイルエンベディング
3. Predictor: センテンス+スタイル → duration (秒単位)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .sentence_encoder import SentenceEncoder
from .style_encoder import StyleEncoderDP


class Predictor(nn.Module):
    """
    Duration Predictor Head

    センテンスエンベディングとスタイルエンベディングから
    発話全体の長さ（秒単位）を予測
    """

    def __init__(
        self,
        sentence_dim: int = 64,
        style_dim: int = 16,
        hdim: int = 128,
        n_layer: int = 2,
    ):
        """
        引数:
            sentence_dim: センテンスエンベディング次元
            style_dim: スタイルエンベディング次元
            hdim: MLP隠れ層次元
            n_layer: MLP層数
        """
        super().__init__()
        self.sentence_dim = sentence_dim
        self.style_dim = style_dim
        self.hdim = hdim
        self.n_layer = n_layer

        # Input dimension
        input_dim = sentence_dim + style_dim

        # MLP layers
        layers = []
        for i in range(n_layer):
            if i == 0:
                layers.append(nn.Linear(input_dim, hdim))
            else:
                layers.append(nn.Linear(hdim, hdim))

            # 最後の層以外はReLU
            if i < n_layer - 1:
                layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hdim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        sentence_emb: torch.Tensor,
        style_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            sentence_emb: センテンスエンベディング [batch, sentence_dim]
            style_emb: スタイルエンベディング [batch, style_dim]

        戻り値:
            duration: 発話長（秒単位） [batch]
        """
        # Concatenate
        x = torch.cat([sentence_emb, style_emb], dim=-1)  # [batch, sentence_dim + style_dim]

        # MLP
        x = self.mlp(x)  # [batch, 1]

        # Squeeze to [batch]
        duration = x.squeeze(-1)

        # Ensure positive duration
        duration = F.softplus(duration)

        return duration


class DurationPredictor(nn.Module):
    """
    Duration Predictor

    テキストと参照音声から発話全体の長さを予測

    アーキテクチャ:
    1. Sentence Encoder: テキスト → センテンスエンベディング [batch, 64, seq_len]
    2. Global Average Pooling: [batch, 64, seq_len] → [batch, 64]
    3. Style Encoder: 参照音声 → スタイルエンベディング [batch, 16]
    4. Predictor: センテンス+スタイル → duration [batch]
    """

    def __init__(
        self,
        # Sentence Encoder
        sentence_encoder_config: dict,
        # Style Encoder
        style_encoder_config: dict,
        # Predictor
        predictor_config: dict,
    ):
        """
        引数:
            sentence_encoder_config: Sentence Encoderの設定
            style_encoder_config: Style Encoderの設定
            predictor_config: Predictorの設定
        """
        super().__init__()

        # 1. Sentence Encoder
        self.sentence_encoder = SentenceEncoder(**sentence_encoder_config)

        # 2. Style Encoder
        self.style_encoder = StyleEncoderDP(**style_encoder_config)

        # 3. Predictor
        self.predictor = Predictor(**predictor_config)

    def encode_sentence(
        self,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        センテンスエンコーディング

        引数:
            text_ids: Text IDs [batch, seq_len]
            text_mask: マスク [batch, 1, seq_len]

        戻り値:
            センテンスエンベディング [batch, sentence_dim]
        """
        # Sentence Encoder
        sentence_emb = self.sentence_encoder(text_ids, text_mask)  # [batch, sentence_dim, seq_len]

        # Global Average Pooling
        if text_mask is not None:
            # マスクされた部分を除外して平均
            mask = text_mask.squeeze(1)  # [batch, seq_len]
            masked_emb = sentence_emb * text_mask  # [batch, sentence_dim, seq_len]
            sentence_emb_pooled = masked_emb.sum(dim=2) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            sentence_emb_pooled = sentence_emb.mean(dim=2)  # [batch, sentence_dim]

        return sentence_emb_pooled

    def encode_style(
        self,
        reference_latent: torch.Tensor,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        スタイルエンコーディング

        引数:
            reference_latent: 参照音声の潜在表現 [batch, ldim × chunk_compress_factor, seq_len]
            reference_mask: マスク [batch, 1, seq_len]

        戻り値:
            スタイルエンベディング [batch, style_dim]
        """
        style_emb = self.style_encoder(reference_latent, reference_mask)
        return style_emb

    def forward(
        self,
        text_ids: torch.Tensor,
        reference_latent: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            text_ids: Text IDs [batch, seq_len]
            reference_latent: 参照音声の潜在表現 [batch, ldim × chunk_compress_factor, ref_len]
            text_mask: テキストマスク [batch, 1, seq_len]
            reference_mask: 参照音声マスク [batch, 1, ref_len]

        戻り値:
            duration: 発話長（秒単位） [batch]
        """
        # 1. Sentence Encoding
        sentence_emb = self.encode_sentence(text_ids, text_mask)  # [batch, sentence_dim]

        # 2. Style Encoding
        style_emb = self.encode_style(reference_latent, reference_mask)  # [batch, style_dim]

        # 3. Predict Duration
        duration = self.predictor(sentence_emb, style_emb)  # [batch]

        return duration

    def inference(
        self,
        text_ids: torch.Tensor,
        reference_latent: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
        speed: float = 1.0,
    ) -> torch.Tensor:
        """
        推論時の順伝播（速度調整付き）

        引数:
            text_ids: Text IDs [batch, seq_len]
            reference_latent: 参照音声の潜在表現 [batch, ldim × chunk_compress_factor, ref_len]
            text_mask: テキストマスク [batch, 1, seq_len]
            reference_mask: 参照音声マスク [batch, 1, ref_len]
            speed: 速度係数（1.0より大きいと速く、小さいと遅く）

        戻り値:
            duration: 調整後の発話長（秒単位） [batch]
        """
        # Duration prediction
        duration = self.forward(text_ids, reference_latent, text_mask, reference_mask)

        # Speed adjustment
        duration = duration / speed

        return duration
