"""
Speech Autoencoder Discriminator

Multi-Scale Discriminator for adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DiscriminatorBlock(nn.Module):
    """
    Discriminatorの基本ブロック

    PatchGAN構造
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 64,
        kernel_sizes: List[int] = [15, 41, 41, 41, 41, 5],
        strides: List[int] = [1, 2, 2, 4, 4, 1],
        groups: List[int] = [1, 4, 16, 16, 16, 1],
    ):
        """
        引数:
            in_channels: 入力チャンネル数
            channels: 基本チャンネル数
            kernel_sizes: 各層のカーネルサイズ
            strides: 各層のストライド
            groups: 各層のグループ数
        """
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 各層の構築
        current_channels = in_channels
        for i, (ksz, stride, group) in enumerate(zip(kernel_sizes, strides, groups)):
            out_channels = channels * (2 ** i) if i < len(kernel_sizes) - 1 else channels

            padding = (ksz - 1) // 2

            # Convolution
            self.convs.append(
                nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size=ksz,
                    stride=stride,
                    padding=padding,
                    groups=group if current_channels >= group else 1,
                )
            )

            # GroupNorm
            if i < len(kernel_sizes) - 1:
                self.norms.append(nn.GroupNorm(min(32, out_channels // 4), out_channels))
            else:
                self.norms.append(nn.Identity())

            current_channels = out_channels

        # 最終層（判別スコア）
        self.final_conv = nn.Conv1d(current_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        順伝播

        引数:
            x: 入力 [batch, in_channels, time]

        戻り値:
            (score, features): 判別スコアと中間層特徴のリスト
        """
        features = []

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = norm(x)
            x = F.leaky_relu(x, 0.2)
            features.append(x)

        # 最終層
        score = self.final_conv(x)

        return score, features


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator

    複数のスケールで音声を判別
    - Full resolution
    - 1/2 resolution (average pooling)
    - 1/4 resolution (average pooling)
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 64,
        num_scales: int = 3,
    ):
        """
        引数:
            in_channels: 入力チャンネル数
            channels: 基本チャンネル数
            num_scales: スケール数
        """
        super().__init__()

        self.num_scales = num_scales

        # 各スケールのDiscriminator
        self.discriminators = nn.ModuleList([
            DiscriminatorBlock(
                in_channels=in_channels,
                channels=channels,
            )
            for _ in range(num_scales)
        ])

        # Downsampling (Average Pooling)
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
            for _ in range(num_scales - 1)
        ])

    def forward(self, x: torch.Tensor) -> tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        順伝播

        引数:
            x: 入力音声 [batch, in_channels, time]

        戻り値:
            (scores, features_list): 各スケールの判別スコアと中間層特徴のリスト
        """
        scores = []
        features_list = []

        # Full resolution
        score, features = self.discriminators[0](x)
        scores.append(score)
        features_list.append(features)

        # Downsampled resolutions
        for i in range(1, self.num_scales):
            x = self.downsamplers[i - 1](x)
            score, features = self.discriminators[i](x)
            scores.append(score)
            features_list.append(features)

        return scores, features_list


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator

    異なる周期で音声を判別
    HiFi-GANスタイル

    注: これはオプションで、論文に明示されていないため、
    必要に応じて使用
    """

    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        channels: int = 64,
    ):
        """
        引数:
            periods: 判別周期のリスト
            channels: 基本チャンネル数
        """
        super().__init__()

        self.periods = periods

        # 各周期のDiscriminator
        self.discriminators = nn.ModuleList([
            PeriodDiscriminatorBlock(period=p, channels=channels)
            for p in periods
        ])

    def forward(self, x: torch.Tensor) -> tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        順伝播

        引数:
            x: 入力音声 [batch, 1, time]

        戻り値:
            (scores, features_list): 各周期の判別スコアと中間層特徴のリスト
        """
        scores = []
        features_list = []

        for disc in self.discriminators:
            score, features = disc(x)
            scores.append(score)
            features_list.append(features)

        return scores, features_list


class PeriodDiscriminatorBlock(nn.Module):
    """
    Period Discriminator Block

    特定の周期で音声を判別
    """

    def __init__(
        self,
        period: int,
        channels: int = 64,
        kernel_size: int = 5,
        stride: int = 3,
    ):
        """
        引数:
            period: 判別周期
            channels: 基本チャンネル数
            kernel_size: カーネルサイズ
            stride: ストライド
        """
        super().__init__()

        self.period = period

        # Convolution layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(2, 0)),
            nn.Conv2d(channels, channels * 2, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(2, 0)),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(2, 0)),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(2, 0)),
            nn.Conv2d(channels * 8, channels * 8, kernel_size=(kernel_size, 1), stride=1, padding=(2, 0)),
        ])

        # Final convolution
        self.final_conv = nn.Conv2d(channels * 8, 1, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        順伝播

        引数:
            x: 入力音声 [batch, 1, time]

        戻り値:
            (score, features): 判別スコアと中間層特徴のリスト
        """
        features = []

        batch, channels, time = x.shape

        # Reshape to (batch, 1, time // period, period)
        if time % self.period != 0:
            pad_amount = self.period - (time % self.period)
            x = F.pad(x, (0, pad_amount), mode="reflect")
            time = x.shape[2]

        x = x.view(batch, channels, time // self.period, self.period)

        # Convolutions
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            features.append(x)

        # Final layer
        score = self.final_conv(x)
        score = torch.flatten(score, 1, -1)

        return score, features
