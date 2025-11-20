"""
Duration Predictor Loss Functions

Duration Predictor用の損失関数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPLoss(nn.Module):
    """
    Duration Predictor Loss

    MSE Loss: ||predicted_duration - ground_truth_duration||^2
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predicted_duration: torch.Tensor,
        ground_truth_duration: torch.Tensor,
    ) -> tuple:
        """
        順伝播

        引数:
            predicted_duration: 予測された発話長 [batch]
            ground_truth_duration: 正解の発話長 [batch]

        戻り値:
            loss: MSE Loss
            metrics: メトリクス辞書
        """
        # MSE Loss
        loss = F.mse_loss(predicted_duration, ground_truth_duration)

        # Metrics
        metrics = {
            "mse_loss": loss.item(),
            "mean_predicted_duration": predicted_duration.mean().item(),
            "mean_ground_truth_duration": ground_truth_duration.mean().item(),
        }

        return loss, metrics
