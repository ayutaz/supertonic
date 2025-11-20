"""
SupertonicTTS 統合学習パイプライン

3段階学習の自動化:
- Stage 1: Speech Autoencoder学習
- Stage 2: TTL (Text-to-Latent)学習
- Stage 3: Duration Predictor学習
"""

import argparse
import json
import yaml
import torch
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models
from models import autoencoder as ae_module
from models import ttl as ttl_module
from models import dp as dp_module

# Import trainers
from trainers import autoencoder_trainer as ae_trainer_module
from trainers import ttl_trainer as ttl_trainer_module
from trainers import dp_trainer as dp_trainer_module

# Import data
from data import datasets as data_module
from data import preprocessing as preprocessing_module

# Specific imports
SpeechAutoencoder = ae_module.SpeechAutoencoder
MultiScaleDiscriminator = ae_module.MultiScaleDiscriminator
TTLModel = ttl_module.TTLModel
DurationPredictor = dp_module.DurationPredictor
AutoencoderTrainer = ae_trainer_module.AutoencoderTrainer
TTLTrainer = ttl_trainer_module.TTLTrainer
DPTrainer = dp_trainer_module.DPTrainer
AudioDataset = data_module.AudioDataset
TTSDataset = data_module.TTSDataset
collate_fn = data_module.collate_fn
MelSpectrogramExtractor = preprocessing_module.MelSpectrogramExtractor
from torch.utils.data import DataLoader, random_split


class PipelineManager:
    """
    3段階学習パイプライン管理クラス

    Stage 1: Autoencoder → Stage 2: TTL → Stage 3: DP
    """

    def __init__(
        self,
        config_path: str,
        data_dir: str,
        metadata_files: list,
        output_dir: str = "outputs",
        device: str = "cuda",
        skip_stages: Optional[list] = None,
    ):
        """
        引数:
            config_path: tts.json設定ファイルパス
            data_dir: データディレクトリ
            metadata_files: メタデータファイルのリスト
            output_dir: 出力ディレクトリ
            device: デバイス ('cuda' or 'cpu')
            skip_stages: スキップする段階のリスト (e.g., ['autoencoder'])
        """
        self.config_path = config_path
        self.data_dir = Path(data_dir)
        self.metadata_files = metadata_files
        self.output_dir = Path(output_dir)
        self.device = device
        self.skip_stages = skip_stages or []

        # 設定読み込み
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # ディレクトリ作成
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("SupertonicTTS 統合学習パイプライン")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Output directory: {output_dir}")
        print(f"Config: {config_path}")
        print(f"Data directory: {data_dir}")
        print(f"Skip stages: {skip_stages}")
        print("=" * 60)

    def get_checkpoint_path(self, stage: str, name: str = "best") -> Path:
        """
        チェックポイントパスを取得

        引数:
            stage: 段階名 ('autoencoder', 'ttl', 'dp')
            name: チェックポイント名 ('best', 'latest', 'final')

        戻り値:
            チェックポイントパス
        """
        return self.checkpoint_dir / f"{stage}_{name}.pt"

    def stage1_train_autoencoder(
        self,
        num_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        segment_length: float = 2.0,  # seconds
        val_ratio: float = 0.05,
    ):
        """
        Stage 1: Speech Autoencoder学習

        引数:
            num_epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            segment_length: 音声セグメント長（秒）
            val_ratio: 検証セット比率
        """
        if "autoencoder" in self.skip_stages:
            print("\n[SKIP] Stage 1: Autoencoder (skipped)")
            return

        print("\n" + "=" * 60)
        print("Stage 1: Speech Autoencoder Training")
        print("=" * 60)

        # Mel-Spectrogram extractor
        ae_config = self.config["ae"]
        spec_config = ae_config["encoder"]["spec_processor"]

        mel_extractor = MelSpectrogramExtractor(
            sample_rate=spec_config["sample_rate"],
            n_fft=spec_config["n_fft"],
            win_length=spec_config.get("win_length", spec_config["n_fft"]),
            hop_length=spec_config["hop_length"],
            n_mels=spec_config["n_mels"],
            eps=spec_config.get("eps", 1e-5),
            norm_mean=spec_config.get("norm_mean", 0.0),
            norm_std=spec_config.get("norm_std", 1.0),
        )

        # Load audio file list
        audio_files = self._load_audio_files(self.metadata_files)
        print(f"Total audio files: {len(audio_files)}")

        # Dataset
        segment_samples = int(segment_length * spec_config["sample_rate"])

        full_dataset = AudioDataset(
            data_dir=self.data_dir,
            audio_files=audio_files,
            mel_extractor=mel_extractor,
            sample_rate=spec_config["sample_rate"],
            segment_length=segment_samples,
            normalize=True,
            trim_silence=False,
        )

        # Train/Val split
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"Train dataset: {len(train_dataset)}")
        print(f"Val dataset: {len(val_dataset)}")

        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Model
        autoencoder = SpeechAutoencoder(
            sample_rate=spec_config["sample_rate"],
            n_fft=spec_config["n_fft"],
            hop_length=spec_config["hop_length"],
            n_mels=spec_config["n_mels"],
            encoder_idim=ae_config["encoder"]["idim"],
            encoder_hdim=ae_config["encoder"]["hdim"],
            encoder_odim=ae_config["encoder"]["odim"],
            encoder_ksz_init=ae_config["encoder"]["ksz_init"],
            encoder_ksz=ae_config["encoder"]["ksz"],
            encoder_num_layers=ae_config["encoder"]["num_layers"],
            encoder_dilation_lst=ae_config["encoder"]["dilation_lst"],
            encoder_intermediate_dim=ae_config["encoder"]["intermediate_dim"],
            decoder_idim=ae_config["decoder"]["idim"],
            decoder_hdim=ae_config["decoder"]["hdim"],
            decoder_odim=ae_config["decoder"]["head"]["odim"],
            decoder_ksz_init=ae_config["decoder"]["ksz_init"],
            decoder_ksz=ae_config["decoder"]["ksz"],
            decoder_num_layers=ae_config["decoder"]["num_layers"],
            decoder_dilation_lst=ae_config["decoder"]["dilation_lst"],
            decoder_intermediate_dim=ae_config["decoder"]["intermediate_dim"],
            head_idim=ae_config["decoder"]["head"]["idim"],
            head_hdim=ae_config["decoder"]["head"]["hdim"],
            head_odim=ae_config["decoder"]["head"]["odim"],
            head_ksz=ae_config["decoder"]["head"]["ksz"],
            vocoder_n_iter=32,
        )

        # Discriminator
        discriminator = MultiScaleDiscriminator(
            in_channels=1,
            channels=64,
            num_scales=3,
        )

        # Trainer config
        trainer_config = {
            "training": {
                "total_steps": num_epochs * len(train_loader),
                "discriminator_start_step": 10000,
                "val_every_n_steps": len(train_loader),
                "save_every_n_steps": len(train_loader) * 5,
                "log_every_n_steps": 100,
                "gradient_clip": 1.0,
                "use_amp": True,
                "batch_size": batch_size,
                "optimizer": {
                    "lr": learning_rate,
                    "betas": [0.9, 0.999],
                    "weight_decay": 0.01,
                },
                "scheduler": {
                    "type": "ExponentialLR",
                    "gamma": 0.999875,
                },
                "checkpoint_dir": str(self.checkpoint_dir / "autoencoder"),
                "wandb": {
                    "enabled": False,
                },
            },
            "loss": {
                "stft_loss_weight": 1.0,
                "mel_loss_weight": 45.0,
                "feature_matching_weight": 2.0,
                "adversarial_loss_weight": 1.0,
                "discriminator_loss_weight": 1.0,
            },
            "model": ae_config,
        }

        # Trainer
        trainer = AutoencoderTrainer(
            model=autoencoder,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            device=self.device,
        )

        # Training
        print("\nStarting Autoencoder training...")
        trainer.train()

        # Save final checkpoint
        checkpoint_path = self.get_checkpoint_path("autoencoder", "final")
        trainer.save_checkpoint("autoencoder_final")
        print(f"\n[Stage 1 Complete] Checkpoint saved: {checkpoint_path}")

    def stage2_train_ttl(
        self,
        num_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        val_ratio: float = 0.05,
        autoencoder_checkpoint: Optional[str] = None,
    ):
        """
        Stage 2: TTL (Text-to-Latent)学習

        引数:
            num_epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            val_ratio: 検証セット比率
            autoencoder_checkpoint: Autoencoderチェックポイントパス（Noneの場合は自動検出）
        """
        if "ttl" in self.skip_stages:
            print("\n[SKIP] Stage 2: TTL (skipped)")
            return

        print("\n" + "=" * 60)
        print("Stage 2: TTL (Text-to-Latent) Training")
        print("=" * 60)

        # Get config
        ae_config = self.config["ae"]
        ttl_config = self.config["ttl"]
        spec_config = ae_config["encoder"]["spec_processor"]

        # 1. Unicode Processor setup
        # Note: unicode_indexer.json should be in the same directory as config
        unicode_indexer_path = None
        config_dir = Path(self.config_path).parent
        potential_indexer_path = config_dir / "unicode_indexer.json"
        if potential_indexer_path.exists():
            unicode_indexer_path = str(potential_indexer_path)
            print(f"[INFO] Using unicode indexer: {unicode_indexer_path}")
        else:
            print("[WARNING] unicode_indexer.json not found, using Unicode values directly")

        unicode_processor = UnicodeProcessor(unicode_indexer_path=unicode_indexer_path)

        # 2. Mel-Spectrogram Extractor setup
        mel_extractor = MelSpectrogramExtractor(
            sample_rate=spec_config["sample_rate"],
            n_fft=spec_config["n_fft"],
            hop_length=spec_config["hop_length"],
            win_length=spec_config["win_length"],
            n_mels=spec_config["n_mels"],
            f_min=spec_config.get("f_min", 0.0),
            f_max=spec_config.get("f_max", None),
            power=spec_config.get("power", 2.0),
        )

        # 3. Dataset creation
        print("\n[INFO] Creating TTSDataset...")
        full_dataset = TTSDataset(
            data_dir=self.data_dir,
            metadata_file=self.metadata_files[0],  # Use first metadata file
            unicode_processor=unicode_processor,
            mel_extractor=mel_extractor,
            sample_rate=spec_config["sample_rate"],
            max_wav_length=None,
            max_text_length=None,
            normalize=True,
            trim_silence=False,
        )

        # Train/Val split
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_ratio)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        print(f"[INFO] Dataset split: Train={train_size}, Val={val_size}")

        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # 4. Load frozen Autoencoder from Stage 1 checkpoint
        print("\n[INFO] Loading frozen Autoencoder...")
        if autoencoder_checkpoint is None:
            # Auto-detect latest checkpoint
            ae_ckpt_dir = self.checkpoint_dir / "autoencoder"
            if ae_ckpt_dir.exists():
                checkpoints = list(ae_ckpt_dir.glob("autoencoder_*.pt"))
                if checkpoints:
                    # Use the final checkpoint if available, otherwise latest
                    final_ckpt = [c for c in checkpoints if "final" in c.name]
                    if final_ckpt:
                        autoencoder_checkpoint = str(final_ckpt[0])
                    else:
                        autoencoder_checkpoint = str(max(checkpoints, key=lambda x: x.stat().st_mtime))
                    print(f"[INFO] Auto-detected checkpoint: {autoencoder_checkpoint}")
                else:
                    print("[ERROR] No autoencoder checkpoint found!")
                    return
            else:
                print("[ERROR] Autoencoder checkpoint directory not found!")
                print("[INFO] Please run Stage 1 first or provide --autoencoder-checkpoint path")
                return

        # Create and load Autoencoder
        autoencoder = SpeechAutoencoder(
            encoder_config=ae_config["encoder"],
            decoder_config=ae_config["decoder"],
            sample_rate=ae_config["sample_rate"],
            hop_length=ae_config.get("hop_length", 256),
        )

        checkpoint = torch.load(autoencoder_checkpoint, map_location=self.device)
        autoencoder.load_state_dict(checkpoint["model"])
        autoencoder.to(self.device)
        autoencoder.eval()

        # Freeze autoencoder
        for param in autoencoder.parameters():
            param.requires_grad = False

        print(f"[INFO] Loaded frozen Autoencoder from {autoencoder_checkpoint}")

        # 5. Create TTL Model
        print("\n[INFO] Creating TTL model...")
        ttl_model = TTLModel(config=ttl_config)
        ttl_model.to(self.device)

        print(f"[INFO] TTL Model created with {sum(p.numel() for p in ttl_model.parameters())} parameters")

        # 6. Optimizer
        optimizer = torch.optim.AdamW(
            ttl_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # 7. Loss function
        loss_fn = TTLLoss()

        # 8. Trainer
        trainer = TTLTrainer(
            ttl_model=ttl_model,
            autoencoder=autoencoder,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device(self.device),
            use_amp=True,
            gradient_clip=1.0,
            log_dir=str(self.log_dir / "ttl"),
            log_interval=10,
            checkpoint_dir=str(self.checkpoint_dir / "ttl"),
            checkpoint_interval=1000,
        )

        # 9. Training loop
        print("\n[INFO] Starting TTL training...")
        print(f"[INFO] Total epochs: {num_epochs}")
        print(f"[INFO] Batch size: {batch_size}")
        print(f"[INFO] Learning rate: {learning_rate}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                # Prepare batch (add text_ids key from text)
                batch["text_ids"] = batch["text"]

                # Train step
                metrics = trainer.train_step(batch)
                total_loss += metrics["loss"]
                num_batches += 1

                # Logging
                if trainer.global_step % trainer.log_interval == 0:
                    print(f"  Step {trainer.global_step}: Loss={metrics['loss']:.4f}")

                # Checkpointing
                if trainer.global_step % trainer.checkpoint_interval == 0:
                    trainer.save_checkpoint(f"ttl_step{trainer.global_step}")

                trainer.global_step += 1

            # Epoch summary
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

            # Validation
            if val_loader is not None and len(val_loader) > 0:
                val_loss = 0.0
                val_batches = 0

                ttl_model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch["text_ids"] = batch["text"]
                        metrics = trainer.train_step(batch)
                        val_loss += metrics["loss"]
                        val_batches += 1

                avg_val_loss = val_loss / val_batches
                print(f"  Validation Loss: {avg_val_loss:.4f}")

                ttl_model.train()

            trainer.epoch += 1

        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "ttl" / "ttl_final.pt"
        trainer.save_checkpoint("ttl_final")
        print(f"\n[Stage 2 Complete] Checkpoint saved: {checkpoint_path}")

    def stage3_train_dp(
        self,
        num_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        val_ratio: float = 0.05,
        autoencoder_checkpoint: Optional[str] = None,
    ):
        """
        Stage 3: Duration Predictor学習

        引数:
            num_epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            val_ratio: 検証セット比率
            autoencoder_checkpoint: Autoencoderチェックポイントパス（Noneの場合は自動検出）
        """
        if "dp" in self.skip_stages:
            print("\n[SKIP] Stage 3: DP (skipped)")
            return

        print("\n" + "=" * 60)
        print("Stage 3: Duration Predictor Training")
        print("=" * 60)

        # Get config
        ae_config = self.config["ae"]
        dp_config = self.config["dp"]
        spec_config = ae_config["encoder"]["spec_processor"]

        # 1. Unicode Processor setup
        unicode_indexer_path = None
        config_dir = Path(self.config_path).parent
        potential_indexer_path = config_dir / "unicode_indexer.json"
        if potential_indexer_path.exists():
            unicode_indexer_path = str(potential_indexer_path)
            print(f"[INFO] Using unicode indexer: {unicode_indexer_path}")
        else:
            print("[WARNING] unicode_indexer.json not found, using Unicode values directly")

        unicode_processor = UnicodeProcessor(unicode_indexer_path=unicode_indexer_path)

        # 2. Mel-Spectrogram Extractor setup
        mel_extractor = MelSpectrogramExtractor(
            sample_rate=spec_config["sample_rate"],
            n_fft=spec_config["n_fft"],
            hop_length=spec_config["hop_length"],
            win_length=spec_config["win_length"],
            n_mels=spec_config["n_mels"],
            f_min=spec_config.get("f_min", 0.0),
            f_max=spec_config.get("f_max", None),
            power=spec_config.get("power", 2.0),
        )

        # 3. Dataset creation
        print("\n[INFO] Creating TTSDataset...")
        full_dataset = TTSDataset(
            data_dir=self.data_dir,
            metadata_file=self.metadata_files[0],
            unicode_processor=unicode_processor,
            mel_extractor=mel_extractor,
            sample_rate=spec_config["sample_rate"],
            max_wav_length=None,
            max_text_length=None,
            normalize=True,
            trim_silence=False,
        )

        # Train/Val split
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_ratio)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        print(f"[INFO] Dataset split: Train={train_size}, Val={val_size}")

        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # 4. Load frozen Autoencoder from Stage 1 checkpoint
        print("\n[INFO] Loading frozen Autoencoder...")
        if autoencoder_checkpoint is None:
            # Auto-detect latest checkpoint
            ae_ckpt_dir = self.checkpoint_dir / "autoencoder"
            if ae_ckpt_dir.exists():
                checkpoints = list(ae_ckpt_dir.glob("autoencoder_*.pt"))
                if checkpoints:
                    final_ckpt = [c for c in checkpoints if "final" in c.name]
                    if final_ckpt:
                        autoencoder_checkpoint = str(final_ckpt[0])
                    else:
                        autoencoder_checkpoint = str(max(checkpoints, key=lambda x: x.stat().st_mtime))
                    print(f"[INFO] Auto-detected checkpoint: {autoencoder_checkpoint}")
                else:
                    print("[ERROR] No autoencoder checkpoint found!")
                    return
            else:
                print("[ERROR] Autoencoder checkpoint directory not found!")
                print("[INFO] Please run Stage 1 first or provide --autoencoder-checkpoint path")
                return

        # Create and load Autoencoder
        autoencoder = SpeechAutoencoder(
            encoder_config=ae_config["encoder"],
            decoder_config=ae_config["decoder"],
            sample_rate=ae_config["sample_rate"],
            hop_length=ae_config.get("hop_length", 256),
        )

        checkpoint = torch.load(autoencoder_checkpoint, map_location=self.device)
        autoencoder.load_state_dict(checkpoint["model"])
        autoencoder.to(self.device)
        autoencoder.eval()

        # Freeze autoencoder
        for param in autoencoder.parameters():
            param.requires_grad = False

        print(f"[INFO] Loaded frozen Autoencoder from {autoencoder_checkpoint}")

        # 5. Create DP Model
        print("\n[INFO] Creating Duration Predictor model...")
        dp_model = DurationPredictor(config=dp_config)
        dp_model.to(self.device)

        print(f"[INFO] DP Model created with {sum(p.numel() for p in dp_model.parameters())} parameters")

        # 6. Optimizer
        optimizer = torch.optim.AdamW(
            dp_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # 7. Scheduler (optional)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * len(train_loader),
            eta_min=learning_rate * 0.1,
        )

        # 8. Trainer
        trainer = DPTrainer(
            dp_model=dp_model,
            autoencoder=autoencoder,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            use_amp=True,
            grad_clip=1.0,
            log_dir=str(self.log_dir / "dp"),
            sample_rate=spec_config["sample_rate"],
        )

        # 9. Training loop
        print("\n[INFO] Starting DP training...")
        print(f"[INFO] Total epochs: {num_epochs}")
        print(f"[INFO] Batch size: {batch_size}")
        print(f"[INFO] Learning rate: {learning_rate}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                # Prepare batch
                # DP requires text_ids, text_mask, wav, reference_wav
                batch["text_ids"] = batch["text"]
                batch["reference_wav"] = batch["wav"]  # Use same audio as reference for simplicity

                # Create text mask from text_lengths
                text_lengths = batch["text_lengths"]
                max_text_len = batch["text"].shape[1]
                text_mask = torch.arange(max_text_len)[None, :] < text_lengths[:, None]
                text_mask = text_mask.unsqueeze(1).float()  # [batch, 1, text_len]
                batch["text_mask"] = text_mask

                # Train step
                metrics = trainer.train_step(batch)
                total_loss += metrics["loss"]
                num_batches += 1

                # Logging
                if trainer.global_step % 10 == 0:
                    print(f"  Step {trainer.global_step}: Loss={metrics['loss']:.4f}, "
                          f"MSE={metrics['mse_loss']:.4f}, "
                          f"Pred Duration={metrics['mean_predicted_duration']:.2f}s, "
                          f"GT Duration={metrics['mean_ground_truth_duration']:.2f}s")

                # Checkpointing (every 1000 steps)
                if trainer.global_step % 1000 == 0:
                    checkpoint_path = self.checkpoint_dir / "dp" / f"dp_step{trainer.global_step}.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model": dp_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "global_step": trainer.global_step,
                        "epoch": epoch,
                    }, checkpoint_path)
                    print(f"  [Checkpoint saved: {checkpoint_path}]")

            # Epoch summary
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

            # Validation
            if val_loader is not None and len(val_loader) > 0:
                val_loss = 0.0
                val_batches = 0

                dp_model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch["text_ids"] = batch["text"]
                        batch["reference_wav"] = batch["wav"]

                        text_lengths = batch["text_lengths"]
                        max_text_len = batch["text"].shape[1]
                        text_mask = torch.arange(max_text_len)[None, :] < text_lengths[:, None]
                        text_mask = text_mask.unsqueeze(1).float()
                        batch["text_mask"] = text_mask

                        metrics = trainer.train_step(batch)
                        val_loss += metrics["loss"]
                        val_batches += 1

                avg_val_loss = val_loss / val_batches
                print(f"  Validation Loss: {avg_val_loss:.4f}")

                dp_model.train()

        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "dp" / "dp_final.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": dp_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "global_step": trainer.global_step,
            "epoch": num_epochs,
        }, checkpoint_path)
        print(f"\n[Stage 3 Complete] Checkpoint saved: {checkpoint_path}")

    def run(
        self,
        ae_epochs: int = 100,
        ttl_epochs: int = 100,
        dp_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
    ):
        """
        完全なパイプライン実行

        引数:
            ae_epochs: Autoencoderエポック数
            ttl_epochs: TTLエポック数
            dp_epochs: DPエポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
        """
        print("\n" + "=" * 60)
        print("Starting 3-Stage Training Pipeline")
        print("=" * 60)
        print(f"Stage 1: Autoencoder ({ae_epochs} epochs)")
        print(f"Stage 2: TTL ({ttl_epochs} epochs)")
        print(f"Stage 3: DP ({dp_epochs} epochs)")
        print("=" * 60)

        # Stage 1: Autoencoder
        self.stage1_train_autoencoder(
            num_epochs=ae_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Stage 2: TTL
        self.stage2_train_ttl(
            num_epochs=ttl_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Stage 3: DP
        self.stage3_train_dp(
            num_epochs=dp_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"Logs saved to: {self.log_dir}")

    def _load_audio_files(self, metadata_files: list) -> list:
        """
        メタデータから音声ファイルリスト読み込み

        引数:
            metadata_files: メタデータファイルパスのリスト

        戻り値:
            音声ファイルパスのリスト
        """
        audio_files = []

        for metadata_file in metadata_files:
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) >= 1:
                        audio_files.append(parts[0])

        return audio_files


def main():
    parser = argparse.ArgumentParser(description="SupertonicTTS 3-Stage Training Pipeline")

    # Data
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to tts.json config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to metadata file(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    # Training
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=100,
        help="Autoencoder training epochs",
    )
    parser.add_argument(
        "--ttl-epochs",
        type=int,
        default=100,
        help="TTL training epochs",
    )
    parser.add_argument(
        "--dp-epochs",
        type=int,
        default=100,
        help="DP training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-stages",
        type=str,
        nargs="*",
        default=[],
        help="Stages to skip (e.g., autoencoder ttl dp)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Run only specific stage (1=AE, 2=TTL, 3=DP)",
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = PipelineManager(
        config_path=args.config,
        data_dir=args.data_dir,
        metadata_files=args.metadata,
        output_dir=args.output_dir,
        device=args.device,
        skip_stages=args.skip_stages,
    )

    # Run specific stage or full pipeline
    if args.stage == 1:
        pipeline.stage1_train_autoencoder(
            num_epochs=args.ae_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    elif args.stage == 2:
        pipeline.stage2_train_ttl(
            num_epochs=args.ttl_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    elif args.stage == 3:
        pipeline.stage3_train_dp(
            num_epochs=args.dp_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    else:
        # Run full pipeline
        pipeline.run(
            ae_epochs=args.ae_epochs,
            ttl_epochs=args.ttl_epochs,
            dp_epochs=args.dp_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
