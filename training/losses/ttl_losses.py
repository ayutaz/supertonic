"""
Text-to-Latent (TTL) Loss Functions

Flow Matchingの損失関数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FlowMatchingLoss(nn.Module):
    """
    Conditional Flow Matching Loss

    Flow Matchingの学習目標:
    L_cfm = E_{t,x_0,x_1,c} [||u_θ(x_t, t, c) - (x_1 - x_0)||^2]

    ここで:
    - x_0 ~ N(0, I): ガウシアンノイズ
    - x_1: 真の潜在表現 (Ground truth)
    - x_t = t * x_1 + (1 - t) * x_0: Linear interpolation
    - u_θ(x_t, t, c): Vector Fieldが予測する速度場
    - (x_1 - x_0): 目標速度場

    論文: "Flow Matching for Generative Modeling" (ICLR 2023)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            v_pred: 予測速度場 [batch, dim, seq_len]
            v_target: 目標速度場 [batch, dim, seq_len]
            mask: マスク [batch, 1, seq_len]
                  有効部分=1.0、パディング=0.0

        戻り値:
            Flow Matching Loss (スカラー)
        """
        # MSE Loss
        loss = F.mse_loss(v_pred, v_target, reduction="none")

        # マスクの適用
        if mask is not None:
            loss = loss * mask
            # 有効な要素数で正規化
            num_valid = mask.sum()
            loss = loss.sum() / (num_valid + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def compute_with_sampling(
        self,
        vector_field: nn.Module,
        latent_gt: torch.Tensor,
        text_emb: torch.Tensor,
        style_emb: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Flow Matching Lossをノイズサンプリングから計算

        引数:
            vector_field: Vector Fieldモデル
            latent_gt: 真の潜在表現 (x_1) [batch, dim, seq_len]
            text_emb: テキストエンベディング [batch, text_dim, text_len]
            style_emb: スタイルエンベディング [batch, style_dim]
            latent_mask: Latentマスク [batch, 1, seq_len]
            text_mask: テキストマスク [batch, 1, text_len]

        戻り値:
            (loss, metrics): 損失と計測値の辞書
        """
        batch_size = latent_gt.size(0)
        device = latent_gt.device

        # 1. 時刻tをサンプリング: t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)

        # 2. ガウシアンノイズをサンプリング: x_0 ~ N(0, I)
        x_0 = torch.randn_like(latent_gt)

        # 3. Linear interpolation: x_t = t * x_1 + (1 - t) * x_0
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * latent_gt + (1 - t_expanded) * x_0

        # マスクの適用
        if latent_mask is not None:
            x_t = x_t * latent_mask

        # 4. Vector Fieldで速度場を予測
        v_pred = vector_field(
            noisy_latent=x_t,
            t=t,
            text_emb=text_emb,
            style_emb=style_emb,
            latent_mask=latent_mask,
            text_mask=text_mask,
        )

        # 5. 目標速度場: v_target = x_1 - x_0
        v_target = latent_gt - x_0

        # 6. Flow Matching Loss
        loss = self.forward(v_pred, v_target, mask=latent_mask)

        # メトリクス
        metrics = {
            "flow_matching_loss": loss.item(),
            "avg_time": t.mean().item(),
        }

        return loss, metrics


class TTLLoss(nn.Module):
    """
    Text-to-Latent (TTL) の総合損失

    Flow Matching Lossのみで構成
    （将来的に追加の損失を含めることも可能）
    """

    def __init__(self):
        super().__init__()
        self.flow_matching_loss = FlowMatchingLoss()

    def forward(
        self,
        vector_field: nn.Module,
        latent_gt: torch.Tensor,
        text_emb: torch.Tensor,
        style_emb: torch.Tensor,
        latent_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        順伝播

        引数:
            vector_field: Vector Fieldモデル
            latent_gt: 真の潜在表現 [batch, dim, seq_len]
            text_emb: テキストエンベディング [batch, text_dim, text_len]
            style_emb: スタイルエンベディング [batch, style_dim]
            latent_mask: Latentマスク [batch, 1, seq_len]
            text_mask: テキストマスク [batch, 1, text_len]

        戻り値:
            (total_loss, metrics): 総合損失と計測値の辞書
        """
        # Flow Matching Loss
        fm_loss, fm_metrics = self.flow_matching_loss.compute_with_sampling(
            vector_field=vector_field,
            latent_gt=latent_gt,
            text_emb=text_emb,
            style_emb=style_emb,
            latent_mask=latent_mask,
            text_mask=text_mask,
        )

        # 総合損失
        total_loss = fm_loss

        # メトリクス
        metrics = {
            "total_loss": total_loss.item(),
            **fm_metrics,
        }

        return total_loss, metrics
