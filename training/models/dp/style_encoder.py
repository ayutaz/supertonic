"""
Style Encoder for Duration Predictor

音声の潜在表現からスタイル情報を抽出（Duration Predictor用）

アーキテクチャ (tts.json の dp.style_encoder):
1. Input Projection: latent (24次元 × chunk_compress_factor) → 64次元
2. ConvNeXt: 4層の処理
3. Style Token Layer: 8個のスタイルトークンによる集約（16次元出力）
"""

import torch
import torch.nn as nn
from typing import Optional

from models.common import ConvNeXtStack, StyleTokenLayer


class StyleEncoderDP(nn.Module):
    """
    Style Encoder for Duration Predictor

    音声の潜在表現からスタイルエンベディングを抽出

    アーキテクチャ:
    1. Input Projection: latent → 64次元
    2. ConvNeXt: 4層の処理
    3. Style Token Layer: 8個のスタイルトークン、16次元出力
    """

    def __init__(
        self,
        # Input Projection
        ldim: int = 24,
        chunk_compress_factor: int = 6,
        proj_in_odim: int = 64,
        # ConvNeXt
        convnext_idim: int = 64,
        convnext_ksz: int = 5,
        convnext_intermediate_dim: int = 256,
        convnext_num_layers: int = 4,
        convnext_dilation_lst: Optional[list] = None,
        # Style Token Layer
        style_input_dim: int = 64,
        n_style: int = 8,
        style_key_dim: int = 0,
        style_value_dim: int = 16,
        prototype_dim: int = 64,
        n_units: int = 64,
        n_heads: int = 2,
    ):
        """
        引数:
            ldim: 潜在表現の次元
            chunk_compress_factor: チャンク圧縮係数
            proj_in_odim: Input Projectionの出力次元
            convnext_idim: ConvNeXtの入力次元
            convnext_ksz: ConvNeXtのカーネルサイズ
            convnext_intermediate_dim: ConvNeXtの中間次元
            convnext_num_layers: ConvNeXt層の数
            convnext_dilation_lst: ConvNeXtのDilationリスト
            style_input_dim: Style Token Layerの入力次元
            n_style: スタイルトークン数
            style_key_dim: スタイルキー次元
            style_value_dim: スタイル値次元
            prototype_dim: プロトタイプ次元
            n_units: ユニット数
            n_heads: ヘッド数
        """
        super().__init__()

        # Dilation listのデフォルト値
        if convnext_dilation_lst is None:
            convnext_dilation_lst = [1] * convnext_num_layers

        # 1. Input Projection
        # latent [ldim × chunk_compress_factor, seq_len] → [proj_in_odim, seq_len]
        input_dim = ldim * chunk_compress_factor
        self.proj_in = nn.Conv1d(input_dim, proj_in_odim, kernel_size=1)

        # 2. ConvNeXt処理
        self.convnext = ConvNeXtStack(
            idim=convnext_idim,
            ksz=convnext_ksz,
            intermediate_dim=convnext_intermediate_dim,
            num_layers=convnext_num_layers,
            dilation_lst=convnext_dilation_lst,
        )

        # 3. Style Token Layer
        self.style_token_layer = StyleTokenLayer(
            input_dim=style_input_dim,
            n_style=n_style,
            style_key_dim=style_key_dim,
            style_value_dim=style_value_dim,
            prototype_dim=prototype_dim,
            n_units=n_units,
            n_heads=n_heads,
        )

    def forward(
        self,
        latent: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            latent: 音声の潜在表現 [batch, ldim × chunk_compress_factor, seq_len]
            mask: マスク [batch, 1, seq_len]
                  有効部分=1.0、パディング=0.0

        戻り値:
            スタイルエンベディング [batch, style_value_dim]
        """
        # 1. Input Projection
        # [batch, ldim × chunk_compress_factor, seq_len] -> [batch, proj_in_odim, seq_len]
        x = self.proj_in(latent)

        # 2. ConvNeXt処理
        # [batch, proj_in_odim, seq_len] -> [batch, proj_in_odim, seq_len]
        x = self.convnext(x)

        # マスクの適用
        if mask is not None:
            x = x * mask

        # 3. Style Token Layer
        # Attention mask (batch_size次元を削除したマスク)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.squeeze(1)  # [batch, seq_len]

        # [batch, proj_in_odim, seq_len] -> [batch, style_value_dim]
        style_emb = self.style_token_layer(x, mask=attn_mask)

        return style_emb
