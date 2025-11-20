"""
Text Encoder for SupertonicTTS

テキストからエンベディングを生成するモジュール

アーキテクチャ (tts.json):
1. Text Embedder: Unicode IDs → 256次元
2. ConvNeXt: 6層の前処理
3. Attention Encoder: 4層のTransformer (LARoPE使用)
4. Output Projection: 256 → 256
"""

import torch
import torch.nn as nn
from typing import Optional

from models.common import ConvNeXtStack, MultiHeadAttention


class TextEmbedder(nn.Module):
    """
    Text Embedding Layer

    Unicode Text IDsを256次元のエンベディングに変換
    """

    def __init__(
        self,
        vocab_size: int,
        char_emb_dim: int = 256,
        padding_idx: int = 0,
    ):
        """
        引数:
            vocab_size: 語彙サイズ
            char_emb_dim: 文字エンベディング次元
            padding_idx: パディングのインデックス
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.char_emb_dim = char_emb_dim

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=char_emb_dim,
            padding_idx=padding_idx,
        )

        # Initialize weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            nn.init.constant_(self.embedding.weight[padding_idx], 0.0)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            text_ids: Text IDs [batch, seq_len]

        戻り値:
            エンベディング [batch, char_emb_dim, seq_len]
        """
        # Embedding
        emb = self.embedding(text_ids)  # [batch, seq_len, char_emb_dim]

        # [batch, seq_len, char_emb_dim] -> [batch, char_emb_dim, seq_len]
        emb = emb.transpose(1, 2)

        return emb


class TextEncoder(nn.Module):
    """
    Text Encoder

    テキストIDsからテキストエンベディングを生成

    アーキテクチャ:
    1. Text Embedder: Unicode IDs → 256次元
    2. ConvNeXt: 6層の前処理
    3. Attention Encoder: 4層のTransformer (LARoPE使用)
    4. Output Projection: 256 → 256
    """

    def __init__(
        self,
        # Text Embedder
        vocab_size: int,
        char_emb_dim: int = 256,
        # ConvNeXt
        convnext_idim: int = 256,
        convnext_ksz: int = 5,
        convnext_intermediate_dim: int = 1024,
        convnext_num_layers: int = 6,
        convnext_dilation_lst: Optional[list] = None,
        # Attention Encoder
        attn_hidden_channels: int = 256,
        attn_filter_channels: int = 1024,
        attn_n_heads: int = 4,
        attn_n_layers: int = 4,
        attn_p_dropout: float = 0.0,
        # RoPE settings
        use_rope: bool = True,
        rotary_base: int = 10000,
        rotary_scale: float = 10.0,
        # Output Projection
        proj_out_idim: int = 256,
        proj_out_odim: int = 256,
    ):
        """
        引数:
            vocab_size: 語彙サイズ
            char_emb_dim: 文字エンベディング次元
            convnext_idim: ConvNeXtの入力次元
            convnext_ksz: ConvNeXtのカーネルサイズ
            convnext_intermediate_dim: ConvNeXtの中間次元
            convnext_num_layers: ConvNeXt層の数
            convnext_dilation_lst: ConvNeXtのDilationリスト
            attn_hidden_channels: Attentionの隠れ層チャンネル数
            attn_filter_channels: Attentionのフィルターチャンネル数
            attn_n_heads: Attentionのヘッド数
            attn_n_layers: Attention層の数
            attn_p_dropout: Attentionのドロップアウト率
            use_rope: RoPEを使用するか
            rotary_base: RoPEのベース値
            rotary_scale: RoPEのスケーリング係数
            proj_out_idim: Output Projectionの入力次元
            proj_out_odim: Output Projectionの出力次元
        """
        super().__init__()

        # Dilation listのデフォルト値
        if convnext_dilation_lst is None:
            convnext_dilation_lst = [1] * convnext_num_layers

        # 1. Text Embedder
        self.text_embedder = TextEmbedder(
            vocab_size=vocab_size,
            char_emb_dim=char_emb_dim,
        )

        # 2. ConvNeXt前処理
        self.convnext = ConvNeXtStack(
            idim=convnext_idim,
            ksz=convnext_ksz,
            intermediate_dim=convnext_intermediate_dim,
            num_layers=convnext_num_layers,
            dilation_lst=convnext_dilation_lst,
        )

        # 3. Attention Encoder (with LARoPE)
        self.attn_encoder = MultiHeadAttention(
            hidden_channels=attn_hidden_channels,
            filter_channels=attn_filter_channels,
            n_heads=attn_n_heads,
            n_layers=attn_n_layers,
            p_dropout=attn_p_dropout,
            use_rope=use_rope,
            rotary_base=rotary_base,
            rotary_scale=rotary_scale,
        )

        # 4. Output Projection
        self.proj_out = nn.Conv1d(proj_out_idim, proj_out_odim, kernel_size=1)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            text_ids: Text IDs [batch, seq_len]
            text_mask: マスク [batch, 1, seq_len]
                       有効部分=1.0、パディング=0.0

        戻り値:
            テキストエンベディング [batch, odim, seq_len]
        """
        # 1. Text Embedder
        # [batch, seq_len] -> [batch, char_emb_dim, seq_len]
        emb = self.text_embedder(text_ids)

        # 2. ConvNeXt前処理
        # [batch, char_emb_dim, seq_len] -> [batch, char_emb_dim, seq_len]
        emb = self.convnext(emb)

        # 3. Attention Encoder
        # テキスト長を計算（LARoPE用）
        if text_mask is not None:
            text_len = text_mask.squeeze(1).sum(dim=1).max().int().item()
        else:
            text_len = text_ids.size(1)

        # Attention mask (batch_size次元を削除したマスク)
        attn_mask = None
        if text_mask is not None:
            attn_mask = text_mask.squeeze(1)  # [batch, seq_len]

        # [batch, char_emb_dim, seq_len] -> [batch, char_emb_dim, seq_len]
        emb = self.attn_encoder(emb, mask=attn_mask, text_len=text_len)

        # 4. Output Projection
        # [batch, char_emb_dim, seq_len] -> [batch, odim, seq_len]
        emb = self.proj_out(emb)

        # マスクの適用（パディング部分をゼロにする）
        if text_mask is not None:
            emb = emb * text_mask

        return emb
