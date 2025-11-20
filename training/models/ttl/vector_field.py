"""
Vector Field (Vector Estimator) for Flow Matching

Flow Matchingの速度場を推定するモデル

アーキテクチャ (tts.json):
1. Input Projection: latent (24×6=144次元) → 512次元
2. Time Encoder: 時刻埋め込み (64次元)
3. Main Blocks (4ブロック):
   - Time Conditioning (FiLM)
   - Style Conditioning (FiLM)
   - Text Cross-Attention (LARoPE)
   - ConvNeXt Block 0, 1, 2
4. Last ConvNeXt: 4層
5. Output Projection: 512次元 → latent (24×6=144次元)
"""

import torch
import torch.nn as nn
from typing import Optional

from models.common import ConvNeXtStack, FiLM, TimeEmbedding
from .cross_attention import CrossAttentionLARoPE


class TimeEncoder(nn.Module):
    """
    Time Encoder

    時刻 t ∈ [0, 1] を埋め込みベクトルに変換し、MLPで処理
    """

    def __init__(self, time_dim: int = 64, hdim: int = 256):
        """
        引数:
            time_dim: Time Embeddingの次元
            hdim: MLPの隠れ層次元
        """
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            t: 時刻 [batch] or [batch, 1], 範囲 [0, 1]

        戻り値:
            時刻エンベディング [batch, hdim]
        """
        emb = self.time_emb(t)  # [batch, time_dim]
        emb = self.mlp(emb)  # [batch, hdim]
        return emb


class MainBlock(nn.Module):
    """
    Vector Fieldのメインブロック

    1つのブロックは以下で構成:
    - Time Conditioning (FiLM)
    - Style Conditioning (FiLM)
    - Text Cross-Attention (LARoPE)
    - ConvNeXt Block 0 (4層, dilation=[1,2,4,8])
    - ConvNeXt Block 1 (1層, dilation=[1])
    - ConvNeXt Block 2 (1層, dilation=[1])
    """

    def __init__(
        self,
        idim: int = 512,
        time_dim: int = 64,
        style_dim: int = 256,
        text_dim: int = 256,
        n_heads: int = 4,
        use_residual: bool = True,
        rotary_base: int = 10000,
        rotary_scale: float = 10.0,
        # ConvNeXt settings
        ksz: int = 5,
        intermediate_dim: int = 1024,
        convnext_0_num_layers: int = 4,
        convnext_0_dilation_lst: Optional[list] = None,
        convnext_1_num_layers: int = 1,
        convnext_1_dilation_lst: Optional[list] = None,
        convnext_2_num_layers: int = 1,
        convnext_2_dilation_lst: Optional[list] = None,
    ):
        """
        引数:
            idim: 入力次元
            time_dim: 時刻エンベディング次元
            style_dim: スタイルエンベディング次元
            text_dim: テキストエンベディング次元
            n_heads: Cross-Attentionのヘッド数
            use_residual: Cross-Attentionでのresidual接続
            rotary_base: RoPEのベース値
            rotary_scale: RoPEのスケーリング係数
            ksz: ConvNeXtのカーネルサイズ
            intermediate_dim: ConvNeXtの中間次元
            convnext_0_num_layers: ConvNeXt Block 0の層数
            convnext_0_dilation_lst: ConvNeXt Block 0のDilationリスト
            convnext_1_num_layers: ConvNeXt Block 1の層数
            convnext_1_dilation_lst: ConvNeXt Block 1のDilationリスト
            convnext_2_num_layers: ConvNeXt Block 2の層数
            convnext_2_dilation_lst: ConvNeXt Block 2のDilationリスト
        """
        super().__init__()

        # Dilation listsのデフォルト値
        if convnext_0_dilation_lst is None:
            convnext_0_dilation_lst = [1, 2, 4, 8]
        if convnext_1_dilation_lst is None:
            convnext_1_dilation_lst = [1]
        if convnext_2_dilation_lst is None:
            convnext_2_dilation_lst = [1]

        # Time Conditioning (FiLM)
        self.time_cond = FiLM(time_dim, idim)

        # Style Conditioning (FiLM)
        self.style_cond = FiLM(style_dim, idim)

        # Text Cross-Attention (LARoPE)
        self.text_attn = CrossAttentionLARoPE(
            q_dim=idim,
            kv_dim=text_dim,
            n_heads=n_heads,
            use_residual=use_residual,
            rotary_base=rotary_base,
            rotary_scale=rotary_scale,
        )

        # ConvNeXt Block 0
        self.convnext_0 = ConvNeXtStack(
            idim=idim,
            ksz=ksz,
            intermediate_dim=intermediate_dim,
            num_layers=convnext_0_num_layers,
            dilation_lst=convnext_0_dilation_lst,
        )

        # ConvNeXt Block 1
        self.convnext_1 = ConvNeXtStack(
            idim=idim,
            ksz=ksz,
            intermediate_dim=intermediate_dim,
            num_layers=convnext_1_num_layers,
            dilation_lst=convnext_1_dilation_lst,
        )

        # ConvNeXt Block 2
        self.convnext_2 = ConvNeXtStack(
            idim=idim,
            ksz=ksz,
            intermediate_dim=intermediate_dim,
            num_layers=convnext_2_num_layers,
            dilation_lst=convnext_2_dilation_lst,
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        style_emb: torch.Tensor,
        text_emb: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        text_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            x: Latent [batch, idim, latent_len]
            time_emb: 時刻エンベディング [batch, time_dim]
            style_emb: スタイルエンベディング [batch, style_dim]
            text_emb: テキストエンベディング [batch, text_dim, text_len]
            latent_mask: Latentマスク [batch, 1, latent_len]
            text_mask: テキストマスク [batch, 1, text_len]
            text_len: テキスト長 (LARoPE用)

        戻り値:
            出力 [batch, idim, latent_len]
        """
        # Time Conditioning (FiLM)
        x = self.time_cond(x, time_emb)

        # Style Conditioning (FiLM)
        x = self.style_cond(x, style_emb)

        # Text Cross-Attention (LARoPE)
        x = self.text_attn(
            query=x,
            key_value=text_emb,
            query_mask=latent_mask,
            kv_mask=text_mask,
            text_len=text_len,
        )

        # ConvNeXt Block 0
        x = self.convnext_0(x)

        # ConvNeXt Block 1
        x = self.convnext_1(x)

        # ConvNeXt Block 2
        x = self.convnext_2(x)

        return x


class VectorField(nn.Module):
    """
    Vector Field (Vector Estimator)

    Flow Matchingの速度場 u_θ(x_t, t, c) を推定

    アーキテクチャ:
    1. Input Projection
    2. Time Encoder
    3. Main Blocks (n_blocks個)
    4. Last ConvNeXt
    5. Output Projection
    """

    def __init__(
        self,
        # Input Projection
        ldim: int = 24,
        chunk_compress_factor: int = 6,
        proj_in_odim: int = 512,
        # Time Encoder
        time_dim: int = 64,
        time_hdim: int = 256,
        # Main Blocks
        n_blocks: int = 4,
        idim: int = 512,
        style_dim: int = 256,
        text_dim: int = 256,
        n_heads: int = 4,
        use_residual: bool = True,
        rotary_base: int = 10000,
        rotary_scale: float = 10.0,
        # ConvNeXt settings
        ksz: int = 5,
        intermediate_dim: int = 1024,
        # Last ConvNeXt
        last_convnext_num_layers: int = 4,
        last_convnext_dilation_lst: Optional[list] = None,
    ):
        """
        引数:
            ldim: 潜在表現の次元
            chunk_compress_factor: チャンク圧縮係数
            proj_in_odim: Input Projectionの出力次元
            time_dim: 時刻エンベディング次元
            time_hdim: Time Encoderの隠れ層次元
            n_blocks: Main Blocksの数
            idim: Main Blocksの次元
            style_dim: スタイルエンベディング次元
            text_dim: テキストエンベディング次元
            n_heads: Cross-Attentionのヘッド数
            use_residual: Cross-Attentionでのresidual接続
            rotary_base: RoPEのベース値
            rotary_scale: RoPEのスケーリング係数
            ksz: ConvNeXtのカーネルサイズ
            intermediate_dim: ConvNeXtの中間次元
            last_convnext_num_layers: Last ConvNeXtの層数
            last_convnext_dilation_lst: Last ConvNeXtのDilationリスト
        """
        super().__init__()
        self.ldim = ldim
        self.chunk_compress_factor = chunk_compress_factor

        # Dilation listsのデフォルト値
        if last_convnext_dilation_lst is None:
            last_convnext_dilation_lst = [1] * last_convnext_num_layers

        # 1. Input Projection
        input_dim = ldim * chunk_compress_factor
        self.proj_in = nn.Conv1d(input_dim, proj_in_odim, kernel_size=1)

        # 2. Time Encoder
        self.time_encoder = TimeEncoder(time_dim, time_hdim)

        # 3. Main Blocks
        self.main_blocks = nn.ModuleList([
            MainBlock(
                idim=idim,
                time_dim=time_hdim,
                style_dim=style_dim,
                text_dim=text_dim,
                n_heads=n_heads,
                use_residual=use_residual,
                rotary_base=rotary_base,
                rotary_scale=rotary_scale,
                ksz=ksz,
                intermediate_dim=intermediate_dim,
                convnext_0_num_layers=4,
                convnext_0_dilation_lst=[1, 2, 4, 8],
                convnext_1_num_layers=1,
                convnext_1_dilation_lst=[1],
                convnext_2_num_layers=1,
                convnext_2_dilation_lst=[1],
            )
            for _ in range(n_blocks)
        ])

        # 4. Last ConvNeXt
        self.last_convnext = ConvNeXtStack(
            idim=idim,
            ksz=ksz,
            intermediate_dim=intermediate_dim,
            num_layers=last_convnext_num_layers,
            dilation_lst=last_convnext_dilation_lst,
        )

        # 5. Output Projection
        output_dim = ldim * chunk_compress_factor
        self.proj_out = nn.Conv1d(idim, output_dim, kernel_size=1)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        style_emb: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            noisy_latent: ノイズの多い潜在表現 [batch, ldim × chunk_compress_factor, latent_len]
            t: 時刻 [batch] or [batch, 1], 範囲 [0, 1]
            text_emb: テキストエンベディング [batch, text_dim, text_len]
            style_emb: スタイルエンベディング [batch, style_dim]
            latent_mask: Latentマスク [batch, 1, latent_len]
            text_mask: テキストマスク [batch, 1, text_len]

        戻り値:
            速度場 [batch, ldim × chunk_compress_factor, latent_len]
        """
        # 1. Input Projection
        x = self.proj_in(noisy_latent)  # [batch, proj_in_odim, latent_len]

        # 2. Time Encoder
        time_emb = self.time_encoder(t)  # [batch, time_hdim]

        # テキスト長を計算（LARoPE用）
        if text_mask is not None:
            text_len = text_mask.squeeze(1).sum(dim=1).max().int().item()
        else:
            text_len = text_emb.size(2)

        # 3. Main Blocks
        for block in self.main_blocks:
            x = block(
                x=x,
                time_emb=time_emb,
                style_emb=style_emb,
                text_emb=text_emb,
                latent_mask=latent_mask,
                text_mask=text_mask,
                text_len=text_len,
            )

        # 4. Last ConvNeXt
        x = self.last_convnext(x)

        # 5. Output Projection
        velocity = self.proj_out(x)  # [batch, ldim × chunk_compress_factor, latent_len]

        # マスクの適用
        if latent_mask is not None:
            velocity = velocity * latent_mask

        return velocity
