"""
TTL (Text-to-Latent) Trainer

Flow Matchingによる学習
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
from pathlib import Path

from models.ttl import TTLModel
from models.autoencoder import SpeechAutoencoder
from losses import TTLLoss


class TTLTrainer:
    """
    TTL Trainer

    Flow Matchingの学習を管理

    学習フロー:
    1. Frozen Autoencoderで音声→潜在表現に変換
    2. Text Encoderでテキスト→テキストエンベディングに変換
    3. Style Encoderで参照音声→スタイルエンベディングに抽出
    4. Vector FieldでFlow Matchingを学習
    """

    def __init__(
        self,
        ttl_model: TTLModel,
        autoencoder: SpeechAutoencoder,
        optimizer: torch.optim.Optimizer,
        loss_fn: TTLLoss,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # Training settings
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        # Logging
        log_dir: Optional[str] = None,
        log_interval: int = 10,
        # Checkpointing
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 1000,
    ):
        """
        引数:
            ttl_model: TTLモデル
            autoencoder: Speech Autoencoder (Frozen)
            optimizer: Optimizer
            loss_fn: TTL Loss関数
            device: デバイス
            use_amp: Automatic Mixed Precision
            gradient_clip: Gradient clipping値
            log_dir: TensorBoardログディレクトリ
            log_interval: ログ出力間隔
            checkpoint_dir: チェックポイント保存ディレクトリ
            checkpoint_interval: チェックポイント保存間隔
        """
        self.ttl_model = ttl_model.to(device)
        self.autoencoder = autoencoder.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.device = device

        # Training settings
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip

        # Gradient scaler for AMP
        self.scaler = GradScaler(enabled=use_amp)

        # Freeze autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()

        # Logging
        self.log_interval = log_interval
        self.writer = None
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0

    def train_step(self, batch: dict) -> dict:
        """
        1ステップの学習

        引数:
            batch: バッチデータ
                - wav: 音声波形 [batch, time]
                - text_ids: Text IDs [batch, text_len]
                - text_mask: テキストマスク [batch, 1, text_len]
                - reference_wav: 参照音声 (オプション)
                - reference_latent: 参照潜在表現 (オプション、事前計算済み)
                - reference_mask: 参照マスク (オプション)

        戻り値:
            metrics: 計測値の辞書
        """
        self.ttl_model.train()
        self.optimizer.zero_grad()

        # デバイスに転送
        wav = batch["wav"].to(self.device)
        text_ids = batch["text_ids"].to(self.device)
        text_mask = batch.get("text_mask", None)
        if text_mask is not None:
            text_mask = text_mask.to(self.device)

        # 1. Frozen Autoencoderで潜在表現を取得
        with torch.no_grad():
            latent_gt, mel_gt = self.autoencoder.encode(wav)

        # Latentマスクを作成
        latent_mask = torch.ones_like(latent_gt[:, :1, :])

        # 2. 参照音声の潜在表現を取得
        if "reference_latent" in batch:
            # 事前計算済みの場合
            reference_latent = batch["reference_latent"].to(self.device)
            reference_mask = batch.get("reference_mask", None)
            if reference_mask is not None:
                reference_mask = reference_mask.to(self.device)
        elif "reference_wav" in batch:
            # 参照音声から計算
            reference_wav = batch["reference_wav"].to(self.device)
            with torch.no_grad():
                reference_latent, _ = self.autoencoder.encode(reference_wav)
            reference_mask = torch.ones_like(reference_latent[:, :1, :])
        else:
            # 参照音声がない場合は、同じ音声を使用
            reference_latent = latent_gt
            reference_mask = latent_mask

        # 3. Text Encoderでテキストエンベディングを生成
        with autocast(enabled=self.use_amp):
            text_emb = self.ttl_model.encode_text(text_ids, text_mask)

            # 4. Style Encoderでスタイルエンベディングを生成
            style_emb = self.ttl_model.encode_style(reference_latent, reference_mask)

            # 5. Flow Matching Loss
            loss, metrics = self.loss_fn(
                vector_field=self.ttl_model.vector_field,
                latent_gt=latent_gt,
                text_emb=text_emb,
                style_emb=style_emb,
                latent_mask=latent_mask,
                text_mask=text_mask,
            )

        # Backward
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.ttl_model.parameters(), self.gradient_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update global step
        self.global_step += 1

        # Logging
        if self.writer is not None and self.global_step % self.log_interval == 0:
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, self.global_step)

            # Gradient norm
            total_norm = 0.0
            for p in self.ttl_model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.writer.add_scalar("train/grad_norm", total_norm, self.global_step)

        # Checkpointing
        if self.checkpoint_dir is not None and self.global_step % self.checkpoint_interval == 0:
            self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

        return metrics

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        """
        検証

        引数:
            val_loader: 検証データローダー

        戻り値:
            avg_metrics: 平均計測値の辞書
        """
        self.ttl_model.eval()

        total_metrics = {}
        num_batches = 0

        for batch in val_loader:
            # デバイスに転送
            wav = batch["wav"].to(self.device)
            text_ids = batch["text_ids"].to(self.device)
            text_mask = batch.get("text_mask", None)
            if text_mask is not None:
                text_mask = text_mask.to(self.device)

            # 1. Frozen Autoencoderで潜在表現を取得
            latent_gt, mel_gt = self.autoencoder.encode(wav)

            # Latentマスクを作成
            latent_mask = torch.ones_like(latent_gt[:, :1, :])

            # 2. 参照音声の潜在表現を取得
            if "reference_latent" in batch:
                reference_latent = batch["reference_latent"].to(self.device)
                reference_mask = batch.get("reference_mask", None)
                if reference_mask is not None:
                    reference_mask = reference_mask.to(self.device)
            elif "reference_wav" in batch:
                reference_wav = batch["reference_wav"].to(self.device)
                reference_latent, _ = self.autoencoder.encode(reference_wav)
                reference_mask = torch.ones_like(reference_latent[:, :1, :])
            else:
                reference_latent = latent_gt
                reference_mask = latent_mask

            # 3. Text Encoderでテキストエンベディングを生成
            text_emb = self.ttl_model.encode_text(text_ids, text_mask)

            # 4. Style Encoderでスタイルエンベディングを生成
            style_emb = self.ttl_model.encode_style(reference_latent, reference_mask)

            # 5. Flow Matching Loss
            loss, metrics = self.loss_fn(
                vector_field=self.ttl_model.vector_field,
                latent_gt=latent_gt,
                text_emb=text_emb,
                style_emb=style_emb,
                latent_mask=latent_mask,
                text_mask=text_mask,
            )

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value

            num_batches += 1

        # Average metrics
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

        # Logging
        if self.writer is not None:
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, self.global_step)

        return avg_metrics

    def save_checkpoint(self, filename: str):
        """
        チェックポイント保存

        引数:
            filename: ファイル名
        """
        if self.checkpoint_dir is None:
            return

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "ttl_model": self.ttl_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"[INFO] Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        チェックポイント読み込み

        引数:
            checkpoint_path: チェックポイントパス
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.ttl_model.load_state_dict(checkpoint["ttl_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

        print(f"[INFO] Checkpoint loaded: {checkpoint_path}")
        print(f"[INFO] Resumed from step {self.global_step}, epoch {self.epoch}")
