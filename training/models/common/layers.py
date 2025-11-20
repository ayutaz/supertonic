"""
Common layer implementations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution (因果的畳み込み)

    未来の情報を参照しない畳み込み層
    AutoregressiveモデルやSequence-to-Sequenceモデルで使用
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        引数:
            in_channels: 入力チャンネル数
            out_channels: 出力チャンネル数
            kernel_size: カーネルサイズ
            dilation: Dilation (膨張率)
            groups: グループ数 (Depthwise Convolutionの場合 = in_channels)
            bias: バイアス項を使用するか
        """
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            x: 入力 [batch, in_channels, seq_len]

        戻り値:
            出力 [batch, out_channels, seq_len]
        """
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]  # 右側のパディングを削除
        return x


class LayerScale(nn.Module):
    """
    Layer Scale (ConvNeXtで使用)

    各層の出力をスケーリングして学習を安定化
    """

    def __init__(self, dim: int, init_value: float = 1e-6):
        """
        引数:
            dim: 次元数
            init_value: 初期値
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            x: 入力 [batch, dim, seq_len]

        戻り値:
            出力 [batch, dim, seq_len]
        """
        return x * self.gamma.unsqueeze(0).unsqueeze(-1)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)

    条件付き情報でfeatureをアフィン変換
    """

    def __init__(self, cond_dim: int, feature_dim: int):
        """
        引数:
            cond_dim: 条件の次元数
            feature_dim: 特徴の次元数
        """
        super().__init__()
        self.scale = nn.Linear(cond_dim, feature_dim)
        self.shift = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            x: 入力特徴 [batch, feature_dim, seq_len]
            cond: 条件 [batch, cond_dim] or [batch, cond_dim, 1]

        戻り値:
            出力 [batch, feature_dim, seq_len]
        """
        # condが [batch, cond_dim] の場合
        if cond.dim() == 2:
            cond = cond.unsqueeze(-1)

        # scale, shift を計算
        scale = self.scale(cond.transpose(1, 2))  # [batch, seq_len, feature_dim]
        shift = self.shift(cond.transpose(1, 2))

        scale = scale.transpose(1, 2)  # [batch, feature_dim, seq_len]
        shift = shift.transpose(1, 2)

        return x * (1 + scale) + shift


class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding (Diffusion/Flow Matchingで使用)

    時刻 t ∈ [0, 1] を埋め込みベクトルに変換
    """

    def __init__(self, dim: int):
        """
        引数:
            dim: 埋め込み次元数
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            t: 時刻 [batch] or [batch, 1]

        戻り値:
            埋め込み [batch, dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb


class StyleTokenLayer(nn.Module):
    """
    Style Token Layer (Global Style Tokens)

    入力特徴から学習可能なスタイルトークンへのアテンションで
    スタイル表現を抽出
    """

    def __init__(
        self,
        input_dim: int,
        n_style: int,
        style_key_dim: int,
        style_value_dim: int,
        prototype_dim: int,
        n_units: int,
        n_heads: int,
    ):
        """
        引数:
            input_dim: 入力次元
            n_style: スタイルトークン数
            style_key_dim: スタイルキー次元 (0の場合はキーなし)
            style_value_dim: スタイル値次元
            prototype_dim: プロトタイプ次元
            n_units: ユニット数
            n_heads: ヘッド数
        """
        super().__init__()
        self.n_style = n_style
        self.style_key_dim = style_key_dim
        self.style_value_dim = style_value_dim
        self.n_heads = n_heads

        # Query projection
        self.query = nn.Linear(input_dim, n_units)

        # Style embeddings
        if style_key_dim > 0:
            # Key-value style tokens
            self.style_keys = nn.Parameter(torch.randn(n_style, style_key_dim))
            self.style_values = nn.Parameter(torch.randn(n_style, style_value_dim))
        else:
            # Value-only style tokens (simplified version)
            self.style_prototypes = nn.Parameter(torch.randn(n_style, prototype_dim))
            self.style_values = nn.Parameter(torch.randn(n_style, style_value_dim))

        # Output projection
        self.output = nn.Linear(style_value_dim, style_value_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        順伝播

        引数:
            x: 入力特徴 [batch, input_dim, seq_len]
            mask: マスク [batch, seq_len]

        戻り値:
            スタイル表現 [batch, style_value_dim]
        """
        batch_size = x.size(0)

        # Query: [batch, seq_len, n_units]
        q = self.query(x.transpose(1, 2))

        if self.style_key_dim > 0:
            # Multi-head attention with style keys
            # Compute attention scores
            attn = torch.matmul(q, self.style_keys.t())  # [batch, seq_len, n_style]
            attn = attn / math.sqrt(self.style_key_dim)

            # Apply mask if provided
            if mask is not None:
                attn = attn.masked_fill(~mask.unsqueeze(-1), float('-inf'))

            # Softmax over sequence length
            attn = F.softmax(attn, dim=1)  # [batch, seq_len, n_style]

            # Weighted sum over sequence
            attn = attn.mean(dim=1)  # [batch, n_style]

            # Apply to style values
            style = torch.matmul(attn, self.style_values)  # [batch, style_value_dim]
        else:
            # Simplified version: prototype-based
            # Compute similarity to prototypes
            q_pooled = q.mean(dim=1)  # [batch, n_units]

            # Expand to match prototype dimension
            attn = torch.matmul(q_pooled, self.style_prototypes.t())  # [batch, n_style]
            attn = F.softmax(attn, dim=-1)

            # Apply to style values
            style = torch.matmul(attn, self.style_values)  # [batch, style_value_dim]

        # Output projection
        style = self.output(style)

        return style
