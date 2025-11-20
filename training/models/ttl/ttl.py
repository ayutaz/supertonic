"""
Text-to-Latent (TTL) Model

テキストから音声の潜在表現を生成するモデル

統合されたコンポーネント:
1. Text Encoder: テキスト → テキストエンベディング
2. Style Encoder: 音声潜在表現 → スタイルエンベディング
3. Vector Field: Flow Matchingで潜在表現を生成
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .text_encoder import TextEncoder
from .style_encoder import StyleEncoder
from .vector_field import VectorField


class TTLModel(nn.Module):
    """
    Text-to-Latent (TTL) Model

    テキストから音声の潜在表現を生成

    学習:
    1. Text Encoderでテキストエンベディングを生成
    2. Style Encoderで参照音声からスタイルを抽出
    3. Flow Matchingで潜在表現を学習

    推論:
    1. Text Encoderでテキストエンベディングを生成
    2. Style Encoderで参照音声からスタイルを抽出
    3. Euler法で潜在表現を生成
    """

    def __init__(
        self,
        # Vocabulary
        vocab_size: int,
        # Text Encoder
        text_encoder_config: Optional[dict] = None,
        # Style Encoder
        style_encoder_config: Optional[dict] = None,
        # Vector Field
        vector_field_config: Optional[dict] = None,
    ):
        """
        引数:
            vocab_size: 語彙サイズ
            text_encoder_config: Text Encoderの設定
            style_encoder_config: Style Encoderの設定
            vector_field_config: Vector Fieldの設定
        """
        super().__init__()

        # デフォルト設定
        if text_encoder_config is None:
            text_encoder_config = {}
        if style_encoder_config is None:
            style_encoder_config = {}
        if vector_field_config is None:
            vector_field_config = {}

        # Text Encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            **text_encoder_config,
        )

        # Style Encoder
        self.style_encoder = StyleEncoder(
            **style_encoder_config,
        )

        # Vector Field
        self.vector_field = VectorField(
            **vector_field_config,
        )

    def encode_text(
        self,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        テキストエンベディングを生成

        引数:
            text_ids: Text IDs [batch, seq_len]
            text_mask: マスク [batch, 1, seq_len]

        戻り値:
            テキストエンベディング [batch, text_dim, seq_len]
        """
        return self.text_encoder(text_ids, text_mask)

    def encode_style(
        self,
        reference_latent: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        スタイルエンベディングを生成

        引数:
            reference_latent: 参照音声の潜在表現 [batch, latent_dim, latent_len]
            latent_mask: マスク [batch, 1, latent_len]

        戻り値:
            スタイルエンベディング [batch, style_dim]
        """
        return self.style_encoder(reference_latent, latent_mask)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        t: torch.Tensor,
        text_ids: torch.Tensor,
        reference_latent: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        latent_mask: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        順伝播 (学習用)

        引数:
            noisy_latent: ノイズの多い潜在表現 [batch, latent_dim, latent_len]
            t: 時刻 [batch] or [batch, 1], 範囲 [0, 1]
            text_ids: Text IDs [batch, text_len]
            reference_latent: 参照音声の潜在表現 [batch, latent_dim, ref_len]
            text_mask: テキストマスク [batch, 1, text_len]
            latent_mask: Latentマスク [batch, 1, latent_len]
            reference_mask: 参照音声マスク [batch, 1, ref_len]

        戻り値:
            予測速度場 [batch, latent_dim, latent_len]
        """
        # Text Encoder
        text_emb = self.text_encoder(text_ids, text_mask)

        # Style Encoder
        style_emb = self.style_encoder(reference_latent, reference_mask)

        # Vector Field
        velocity = self.vector_field(
            noisy_latent=noisy_latent,
            t=t,
            text_emb=text_emb,
            style_emb=style_emb,
            latent_mask=latent_mask,
            text_mask=text_mask,
        )

        return velocity

    @torch.no_grad()
    def inference(
        self,
        text_ids: torch.Tensor,
        reference_latent: torch.Tensor,
        latent_len: int,
        text_mask: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
        total_steps: int = 5,
    ) -> torch.Tensor:
        """
        推論 (Euler法)

        引数:
            text_ids: Text IDs [batch, text_len]
            reference_latent: 参照音声の潜在表現 [batch, latent_dim, ref_len]
            latent_len: 生成する潜在表現の長さ
            text_mask: テキストマスク [batch, 1, text_len]
            reference_mask: 参照音声マスク [batch, 1, ref_len]
            total_steps: Euler法のステップ数

        戻り値:
            生成された潜在表現 [batch, latent_dim, latent_len]
        """
        batch_size = text_ids.size(0)
        device = text_ids.device
        latent_dim = reference_latent.size(1)

        # Text Encoder
        text_emb = self.text_encoder(text_ids, text_mask)

        # Style Encoder
        style_emb = self.style_encoder(reference_latent, reference_mask)

        # 初期ノイズ: x_0 ~ N(0, I)
        x = torch.randn(batch_size, latent_dim, latent_len, device=device)

        # Latentマスク（全て有効）
        latent_mask = torch.ones(batch_size, 1, latent_len, device=device)

        # Euler法
        dt = 1.0 / total_steps
        for step in range(total_steps):
            t = torch.full((batch_size,), step / total_steps, device=device)

            # 速度場を予測
            velocity = self.vector_field(
                noisy_latent=x,
                t=t,
                text_emb=text_emb,
                style_emb=style_emb,
                latent_mask=latent_mask,
                text_mask=text_mask,
            )

            # Euler更新: x = x + v * dt
            x = x + velocity * dt

        return x
