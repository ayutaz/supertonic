"""
Speech Autoencoder Trainer

Autoencoder + Discriminatorの学習ループ
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional
import wandb
from tqdm import tqdm

from models.autoencoder import SpeechAutoencoder, MultiScaleDiscriminator
from losses import (
    MultiScaleSTFTLoss,
    MelSpectrogramLoss,
    FeatureMatchingLoss,
    AdversarialLoss,
    DiscriminatorLoss,
)


class AutoencoderTrainer:
    """
    Speech Autoencoder Trainer

    Generator (Autoencoder) と Discriminator を交互に学習
    """

    def __init__(
        self,
        model: SpeechAutoencoder,
        discriminator: MultiScaleDiscriminator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        """
        引数:
            model: Speech Autoencoderモデル
            discriminator: Discriminator
            train_loader: 学習データローダー
            val_loader: 検証データローダー
            config: 設定辞書
            device: デバイス
        """
        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training config
        training_config = config.get("training", {})
        self.total_steps = training_config.get("total_steps", 1500000)
        self.discriminator_start_step = training_config.get("discriminator_start_step", 10000)
        self.val_every_n_steps = training_config.get("val_every_n_steps", 1000)
        self.save_every_n_steps = training_config.get("save_every_n_steps", 5000)
        self.log_every_n_steps = training_config.get("log_every_n_steps", 100)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)
        self.use_amp = training_config.get("use_amp", True)

        # Loss config
        loss_config = config.get("loss", {})
        self.stft_loss_weight = loss_config.get("stft_loss_weight", 1.0)
        self.mel_loss_weight = loss_config.get("mel_loss_weight", 45.0)
        self.feature_matching_weight = loss_config.get("feature_matching_weight", 2.0)
        self.adversarial_loss_weight = loss_config.get("adversarial_loss_weight", 1.0)
        self.discriminator_loss_weight = loss_config.get("discriminator_loss_weight", 1.0)

        # Loss functions
        self.stft_loss = MultiScaleSTFTLoss().to(device)
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=model.sample_rate,
            hop_length=model.hop_length,
        ).to(device)
        self.feature_matching_loss = FeatureMatchingLoss().to(device)
        self.adversarial_loss = AdversarialLoss().to(device)
        self.discriminator_loss = DiscriminatorLoss().to(device)

        # Optimizers
        opt_config = training_config.get("optimizer", {})
        lr = opt_config.get("lr", 2e-4)
        betas = opt_config.get("betas", [0.8, 0.99])
        weight_decay = opt_config.get("weight_decay", 0.01)

        self.optimizer_g = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        self.optimizer_d = optim.AdamW(
            discriminator.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        # Scheduler
        scheduler_config = training_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "ExponentialLR")

        if scheduler_type == "ExponentialLR":
            gamma = scheduler_config.get("gamma", 0.999875)
            self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=gamma)
            self.scheduler_d = optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=gamma)
        else:
            self.scheduler_g = None
            self.scheduler_d = None

        # Gradient scaler for AMP
        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)

        # Checkpointing
        self.checkpoint_dir = Path(training_config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        log_config = training_config.get("wandb", {})
        self.use_wandb = log_config.get("enabled", False)

        if self.use_wandb:
            wandb.init(
                project=log_config.get("project", "supertonic-autoencoder"),
                entity=log_config.get("entity"),
                config=config,
            )

        # State
        self.global_step = 0
        self.epoch = 0

    def train_step_generator(self, batch: dict) -> dict:
        """
        Generatorの学習ステップ

        引数:
            batch: バッチデータ

        戻り値:
            損失の辞書
        """
        wav = batch["wav"].to(self.device)

        self.optimizer_g.zero_grad()

        with autocast(enabled=self.use_amp):
            # Forward
            reconstructed_wav, latent, mel_original, mel_reconstructed = self.model(wav)

            # Reconstruction losses
            stft_mag_loss, stft_phase_loss = self.stft_loss(reconstructed_wav, wav)
            stft_loss = stft_mag_loss + stft_phase_loss

            mel_loss = self.mel_loss(reconstructed_wav, wav)

            # Total reconstruction loss
            recon_loss = (
                self.stft_loss_weight * stft_loss +
                self.mel_loss_weight * mel_loss
            )

            # Adversarial loss (if discriminator is active)
            if self.global_step >= self.discriminator_start_step:
                # Discriminator forward (for generated audio)
                disc_fake_scores, disc_fake_features = self.discriminator(
                    reconstructed_wav.unsqueeze(1)
                )

                # Discriminator forward (for real audio)
                with torch.no_grad():
                    disc_real_scores, disc_real_features = self.discriminator(
                        wav.unsqueeze(1)
                    )

                # Adversarial loss
                adv_loss = self.adversarial_loss(disc_fake_scores)

                # Feature matching loss
                fm_loss = self.feature_matching_loss(disc_real_features, disc_fake_features)

                # Total generator loss
                total_loss = (
                    recon_loss +
                    self.adversarial_loss_weight * adv_loss +
                    self.feature_matching_weight * fm_loss
                )
            else:
                adv_loss = torch.tensor(0.0)
                fm_loss = torch.tensor(0.0)
                total_loss = recon_loss

        # Backward
        self.scaler_g.scale(total_loss).backward()
        self.scaler_g.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        if self.scheduler_g is not None:
            self.scheduler_g.step()

        return {
            "loss/generator": total_loss.item(),
            "loss/stft": stft_loss.item(),
            "loss/mel": mel_loss.item(),
            "loss/adversarial": adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            "loss/feature_matching": fm_loss.item() if isinstance(fm_loss, torch.Tensor) else fm_loss,
        }

    def train_step_discriminator(self, batch: dict) -> dict:
        """
        Discriminatorの学習ステップ

        引数:
            batch: バッチデータ

        戻り値:
            損失の辞書
        """
        if self.global_step < self.discriminator_start_step:
            return {"loss/discriminator": 0.0}

        wav = batch["wav"].to(self.device)

        self.optimizer_d.zero_grad()

        with autocast(enabled=self.use_amp):
            # Generate fake audio
            with torch.no_grad():
                reconstructed_wav, _, _, _ = self.model(wav)

            # Discriminator forward
            disc_real_scores, _ = self.discriminator(wav.unsqueeze(1))
            disc_fake_scores, _ = self.discriminator(reconstructed_wav.unsqueeze(1).detach())

            # Discriminator loss
            disc_loss = self.discriminator_loss(disc_real_scores, disc_fake_scores)
            disc_loss = self.discriminator_loss_weight * disc_loss

        # Backward
        self.scaler_d.scale(disc_loss).backward()
        self.scaler_d.unscale_(self.optimizer_d)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip)
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()

        if self.scheduler_d is not None:
            self.scheduler_d.step()

        return {
            "loss/discriminator": disc_loss.item(),
        }

    def validate(self) -> dict:
        """
        検証

        戻り値:
            検証損失の辞書
        """
        self.model.eval()
        self.discriminator.eval()

        total_stft_loss = 0.0
        total_mel_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                wav = batch["wav"].to(self.device)

                # Forward
                reconstructed_wav, _, _, _ = self.model(wav)

                # Losses
                stft_mag_loss, stft_phase_loss = self.stft_loss(reconstructed_wav, wav)
                stft_loss = stft_mag_loss + stft_phase_loss
                mel_loss = self.mel_loss(reconstructed_wav, wav)

                total_stft_loss += stft_loss.item()
                total_mel_loss += mel_loss.item()
                num_batches += 1

        self.model.train()
        self.discriminator.train()

        return {
            "val/stft_loss": total_stft_loss / num_batches,
            "val/mel_loss": total_mel_loss / num_batches,
        }

    def save_checkpoint(self, name: str = "checkpoint"):
        """
        チェックポイントを保存

        引数:
            name: チェックポイント名
        """
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d": self.scaler_d.state_dict(),
        }

        if self.scheduler_g is not None:
            checkpoint["scheduler_g"] = self.scheduler_g.state_dict()
            checkpoint["scheduler_d"] = self.scheduler_d.state_dict()

        save_path = self.checkpoint_dir / f"{name}_step{self.global_step}.pt"
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

    def train(self):
        """
        学習ループ
        """
        print(f"Starting training for {self.total_steps} steps...")
        print(f"Discriminator will start at step {self.discriminator_start_step}")

        self.model.train()
        self.discriminator.train()

        pbar = tqdm(total=self.total_steps, initial=self.global_step)

        while self.global_step < self.total_steps:
            for batch in self.train_loader:
                # Generator step
                losses_g = self.train_step_generator(batch)

                # Discriminator step
                losses_d = self.train_step_discriminator(batch)

                # Merge losses
                losses = {**losses_g, **losses_d}

                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    pbar.set_postfix(losses)
                    if self.use_wandb:
                        wandb.log(losses, step=self.global_step)

                # Validation
                if self.global_step % self.val_every_n_steps == 0:
                    val_losses = self.validate()
                    print(f"\nValidation at step {self.global_step}: {val_losses}")
                    if self.use_wandb:
                        wandb.log(val_losses, step=self.global_step)

                # Checkpointing
                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint("autoencoder")

                self.global_step += 1
                pbar.update(1)

                if self.global_step >= self.total_steps:
                    break

            self.epoch += 1

        pbar.close()

        # Final checkpoint
        self.save_checkpoint("autoencoder_final")
        print("Training completed!")

        if self.use_wandb:
            wandb.finish()
