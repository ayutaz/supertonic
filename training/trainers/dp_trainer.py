"""
Duration Predictor Trainer

Duration Predictorの学習を管理するTrainerクラス
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict

from models.autoencoder import SpeechAutoencoder
from models.dp import DurationPredictor
from losses import DPLoss


class DPTrainer:
    """
    Duration Predictor Trainer

    Duration Predictorの学習を管理
    """

    def __init__(
        self,
        dp_model: DurationPredictor,
        autoencoder: SpeechAutoencoder,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: Optional[float] = 1.0,
        log_dir: Optional[str] = None,
        sample_rate: int = 44100,
    ):
        """
        引数:
            dp_model: Duration Predictorモデル
            autoencoder: Frozen Autoencoder
            optimizer: オプティマイザー
            scheduler: 学習率スケジューラー（オプション）
            device: デバイス（'cuda' or 'cpu'）
            use_amp: Automatic Mixed Precisionを使用するか
            grad_clip: Gradient clippingの閾値（Noneの場合は無効）
            log_dir: TensorBoardログディレクトリ（Noneの場合は無効）
            sample_rate: サンプリングレート（duration計算用）
        """
        self.dp_model = dp_model.to(device)
        self.autoencoder = autoencoder.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.sample_rate = sample_rate

        # Freeze Autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()

        # Loss function
        self.loss_fn = DPLoss()

        # AMP
        self.scaler = GradScaler() if use_amp else None

        # TensorBoard
        self.writer = SummaryWriter(log_dir) if log_dir is not None else None
        self.global_step = 0

    def train_step(self, batch: dict) -> dict:
        """
        1ステップの学習

        引数:
            batch: バッチデータ
                - text_ids: Text IDs [batch, seq_len]
                - text_mask: Text mask [batch, 1, seq_len]
                - wav: 音声波形 [batch, wav_len]
                - reference_wav: 参照音声波形 [batch, ref_len]

        戻り値:
            metrics: メトリクス辞書
        """
        self.dp_model.train()

        # デバイスに移動
        text_ids = batch["text_ids"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)
        wav = batch["wav"].to(self.device)
        reference_wav = batch["reference_wav"].to(self.device)

        # 1. Frozen Autoencoderで参照音声の潜在表現を取得
        with torch.no_grad():
            reference_latent, _ = self.autoencoder.encode(reference_wav)
            reference_latent = reference_latent.detach()

        # Reference mask（全て有効と仮定）
        batch_size, _, ref_len = reference_latent.shape
        reference_mask = torch.ones(batch_size, 1, ref_len, device=self.device)

        # 2. Ground Truth Duration
        # duration = wav_length / sample_rate
        wav_lengths = batch.get("wav_lengths", torch.tensor([wav.size(1)] * batch_size))
        ground_truth_duration = wav_lengths.float() / self.sample_rate
        ground_truth_duration = ground_truth_duration.to(self.device)

        # 3. Predict Duration
        with autocast(enabled=self.use_amp):
            predicted_duration = self.dp_model(
                text_ids=text_ids,
                reference_latent=reference_latent,
                text_mask=text_mask,
                reference_mask=reference_mask,
            )

            # 4. Loss
            loss, metrics = self.loss_fn(predicted_duration, ground_truth_duration)

        # 5. Backward
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.dp_model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.dp_model.parameters(), self.grad_clip)
            self.optimizer.step()

        # 6. Scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        # 7. Logging
        self.global_step += 1
        if self.writer is not None:
            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.writer.add_scalar("train/mse_loss", metrics["mse_loss"], self.global_step)
            self.writer.add_scalar(
                "train/mean_predicted_duration",
                metrics["mean_predicted_duration"],
                self.global_step,
            )
            self.writer.add_scalar(
                "train/mean_ground_truth_duration",
                metrics["mean_ground_truth_duration"],
                self.global_step,
            )
            if self.scheduler is not None:
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)

        return metrics

    @torch.no_grad()
    def validation_step(self, batch: dict) -> dict:
        """
        1ステップの検証

        引数:
            batch: バッチデータ

        戻り値:
            metrics: メトリクス辞書
        """
        self.dp_model.eval()

        # デバイスに移動
        text_ids = batch["text_ids"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)
        wav = batch["wav"].to(self.device)
        reference_wav = batch["reference_wav"].to(self.device)

        # 1. Frozen Autoencoderで参照音声の潜在表現を取得
        reference_latent, _ = self.autoencoder.encode(reference_wav)

        # Reference mask
        batch_size, _, ref_len = reference_latent.shape
        reference_mask = torch.ones(batch_size, 1, ref_len, device=self.device)

        # 2. Ground Truth Duration
        wav_lengths = batch.get("wav_lengths", torch.tensor([wav.size(1)] * batch_size))
        ground_truth_duration = wav_lengths.float() / self.sample_rate
        ground_truth_duration = ground_truth_duration.to(self.device)

        # 3. Predict Duration
        with autocast(enabled=self.use_amp):
            predicted_duration = self.dp_model(
                text_ids=text_ids,
                reference_latent=reference_latent,
                text_mask=text_mask,
                reference_mask=reference_mask,
            )

            # 4. Loss
            loss, metrics = self.loss_fn(predicted_duration, ground_truth_duration)

        # 5. Logging
        if self.writer is not None:
            self.writer.add_scalar("val/loss", loss.item(), self.global_step)
            self.writer.add_scalar("val/mse_loss", metrics["mse_loss"], self.global_step)

        return metrics

    def save_checkpoint(self, checkpoint_path: str):
        """
        チェックポイント保存

        引数:
            checkpoint_path: 保存先パス
        """
        checkpoint = {
            "model_state_dict": self.dp_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        チェックポイント読み込み

        引数:
            checkpoint_path: 読み込み元パス
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.dp_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")
