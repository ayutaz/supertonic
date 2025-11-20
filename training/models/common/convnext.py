"""
ConvNeXt Block Implementation

論文で使用されるConvNeXtブロックの実装
Causal Convolution + Inverted Bottleneck + Layer Scale
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .layers import CausalConv1d, LayerScale


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (1D version)

    アーキテクチャ:
    1. Depthwise Causal Convolution (7x7)
    2. LayerNorm
    3. Inverted Bottleneck (1x1 conv → GELU → 1x1 conv)
    4. Layer Scale
    5. Residual Connection
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        expansion_factor: int = 4,
        layer_scale_init: float = 1e-6,
        dropout: float = 0.0,
    ):
        """
        引数:
            dim: 入力・出力チャンネル数
            kernel_size: カーネルサイズ
            dilation: Dilation (膨張率)
            expansion_factor: Inverted Bottleneckの拡張率
            layer_scale_init: Layer Scaleの初期値
            dropout: Dropout率
        """
        super().__init__()

        # Depthwise Causal Convolution
        self.dwconv = CausalConv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=dim,  # Depthwise
            bias=True,
        )

        # Normalization
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Inverted Bottleneck (Pointwise Convolutions)
        intermediate_dim = dim * expansion_factor
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        # Layer Scale
        self.layer_scale = LayerScale(dim, init_value=layer_scale_init)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            x: 入力 [batch, dim, seq_len]

        戻り値:
            出力 [batch, dim, seq_len]
        """
        residual = x

        # Depthwise Conv
        x = self.dwconv(x)

        # Transpose for LayerNorm and Linear layers
        x = x.transpose(1, 2)  # [batch, seq_len, dim]

        # LayerNorm
        x = self.norm(x)

        # Inverted Bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Transpose back
        x = x.transpose(1, 2)  # [batch, dim, seq_len]

        # Layer Scale
        x = self.layer_scale(x)

        # Dropout
        x = self.dropout(x)

        # Residual
        x = x + residual

        return x


class ConvNeXtStack(nn.Module):
    """
    ConvNeXt Block のスタック

    複数のConvNeXtブロックを積み重ねたもの
    各ブロックは異なるdilationを持つことができる
    """

    def __init__(
        self,
        idim: int,
        ksz: int = 7,
        intermediate_dim: int = 1024,
        num_layers: int = 6,
        dilation_lst: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        """
        引数:
            idim: 入力・出力次元
            ksz: カーネルサイズ
            intermediate_dim: 中間層の次元数
            num_layers: レイヤー数
            dilation_lst: 各レイヤーのDilation (Noneの場合は全て1)
            dropout: Dropout率
        """
        super().__init__()

        if dilation_lst is None:
            dilation_lst = [1] * num_layers

        assert len(dilation_lst) == num_layers, (
            f"dilation_lst length ({len(dilation_lst)}) must match "
            f"num_layers ({num_layers})"
        )

        # expansion_factor を計算
        expansion_factor = intermediate_dim // idim

        # ConvNeXt Blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=idim,
                kernel_size=ksz,
                dilation=dilation,
                expansion_factor=expansion_factor,
                dropout=dropout,
            )
            for dilation in dilation_lst
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            x: 入力 [batch, idim, seq_len]

        戻り値:
            出力 [batch, idim, seq_len]
        """
        for block in self.blocks:
            x = block(x)
        return x


class InitialConvNeXt(nn.Module):
    """
    Initial ConvNeXt (入力処理用)

    初期の畳み込み層 + ConvNeXtスタック
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksz_init: int = 7,
        ksz: int = 7,
        intermediate_dim: int = 1024,
        num_layers: int = 6,
        dilation_lst: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        """
        引数:
            in_channels: 入力チャンネル数
            out_channels: 出力チャンネル数
            ksz_init: 初期カーネルサイズ
            ksz: ConvNeXtブロックのカーネルサイズ
            intermediate_dim: 中間層の次元数
            num_layers: ConvNeXtブロックの数
            dilation_lst: 各レイヤーのDilation
            dropout: Dropout率
        """
        super().__init__()

        # Initial Convolution
        self.init_conv = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=ksz_init,
            bias=True,
        )

        # ConvNeXt Stack
        self.convnext_stack = ConvNeXtStack(
            idim=out_channels,
            ksz=ksz,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            dilation_lst=dilation_lst,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        引数:
            x: 入力 [batch, in_channels, seq_len]

        戻り値:
            出力 [batch, out_channels, seq_len]
        """
        x = self.init_conv(x)
        x = self.convnext_stack(x)
        return x
