"""
Utility functions for models
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


def get_activation(activation: str) -> nn.Module:
    """
    活性化関数を取得

    引数:
        activation: 活性化関数名 ("relu", "gelu", "silu", etc.)

    戻り値:
        活性化関数モジュール
    """
    activation = activation.lower()

    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu" or activation == "swish":
        return nn.SiLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leakyrelu":
        return nn.LeakyReLU(0.2)
    else:
        raise ValueError(f"Unknown activation: {activation}")


def init_weights(module: nn.Module, scale: float = 1.0):
    """
    重みの初期化

    引数:
        module: 初期化対象のモジュール
        scale: 初期化スケール
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=0.02 * scale)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """
    シーケンスマスクを生成

    引数:
        length: 各シーケンスの長さ [batch_size]
        max_length: 最大長 (Noneの場合はlength.max()を使用)

    戻り値:
        マスク [batch_size, max_length]
    """
    if max_length is None:
        max_length = length.max().item()

    batch_size = length.size(0)
    seq_range = torch.arange(0, max_length, dtype=length.dtype, device=length.device)
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_length)

    mask = seq_range < length.unsqueeze(1)
    return mask


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Duration Alignmentパスを生成 (Monotonic Alignment Search用)

    引数:
        duration: 各フレームの継続時間 [batch, text_len]
        mask: テキストマスク [batch, text_len]

    戻り値:
        アライメントパス [batch, mel_len, text_len]
    """
    batch, text_len = duration.size()

    # 累積和でフレーム位置を計算
    cum_duration = torch.cumsum(duration, dim=1)
    cum_duration_prev = F.pad(cum_duration[:, :-1], (1, 0))

    mel_len = cum_duration.max().int().item()

    # パスを生成
    path = torch.zeros(batch, mel_len, text_len, dtype=duration.dtype, device=duration.device)

    for b in range(batch):
        for t in range(text_len):
            if mask[b, t]:
                start = int(cum_duration_prev[b, t].item())
                end = int(cum_duration[b, t].item())
                path[b, start:end, t] = 1.0

    return path


def convert_pad_shape(pad_shape: list) -> list:
    """
    パディング形状を変換 (TensorFlowスタイル -> PyTorchスタイル)

    引数:
        pad_shape: [[before, after], ...] 形式のリスト

    戻り値:
        [before_last, after_last, before_second, after_second, ...] 形式のリスト
    """
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


def subsequent_mask(size: int, device: torch.device = None) -> torch.Tensor:
    """
    Subsequent Mask (下三角行列) を生成 (Transformer Decoder用)

    引数:
        size: マスクサイズ
        device: デバイス

    戻り値:
        マスク [size, size]
    """
    return torch.tril(torch.ones(size, size, device=device)).bool()


def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor,
                                     n_channels: int) -> torch.Tensor:
    """
    WaveNet-style gated activation

    引数:
        input_a: 入力テンソルA
        input_b: 入力テンソルB
        n_channels: チャンネル数

    戻り値:
        出力テンソル
    """
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts
